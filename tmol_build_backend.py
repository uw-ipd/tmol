"""PEP 517 backend wrapper with optional prebuilt wheel auto-fetch.

This backend mirrors FlashAttention-style UX:
1) Try to fetch a matching prebuilt wheel from GitHub Releases.
2) Fall back to a normal local build when no match is found.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_skbuild_import_error: ModuleNotFoundError | None = None

try:
    from scikit_build_core import build as _skbuild_backend
except ModuleNotFoundError as _import_error:
    _skbuild_import_error = _import_error

    class _MissingSkbuildBackend:
        @staticmethod
        def _raise() -> None:
            raise ModuleNotFoundError(
                "scikit_build_core is required for tmol build backend operations. "
                "Install it with `pip install scikit-build-core`."
            ) from _skbuild_import_error

        def build_wheel(self, *args, **kwargs):
            self._raise()

        def build_editable(self, *args, **kwargs):
            self._raise()

        def build_sdist(self, *args, **kwargs):
            self._raise()

        def get_requires_for_build_wheel(self, *args, **kwargs):
            self._raise()

        def get_requires_for_build_editable(self, *args, **kwargs):
            self._raise()

        def get_requires_for_build_sdist(self, *args, **kwargs):
            self._raise()

        def prepare_metadata_for_build_wheel(self, *args, **kwargs):
            self._raise()

        def prepare_metadata_for_build_editable(self, *args, **kwargs):
            self._raise()

    _skbuild_backend = _MissingSkbuildBackend()

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_RELEASE_BASE_URL = "https://github.com/uw-ipd/tmol/releases/download"
_PYPROJECT_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')
_SUPPORTED_ARCHES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64"}
_RELEASE_URL_ENV = "TMOL_WHEEL_RELEASE_BASE_URL"
_RELEASE_TAG_ENV = "TMOL_WHEEL_RELEASE_TAG"
_LOCAL_TAG_ENV = "TMOL_WHEEL_LOCAL_TAG"
_FORCE_BUILD_ENV = "TMOL_FORCE_BUILD"
_DISABLE_FETCH_ENV = "TMOL_DISABLE_WHEEL_FETCH"
_ALLOW_CPU_FALLBACK_ENV = "TMOL_WHEEL_ALLOW_CPU_FALLBACK"
_ALLOW_SOURCE_BUILD_ENV = "TMOL_ALLOW_SOURCE_BUILD"
_GITHUB_REPO = "uw-ipd/tmol"
# Published wheel CUDA tags per torch minor (pip index tags, not always torch.version.cuda).
_TORCH_CUDA_ALIASES: dict[str, list[str]] = {
    "2.12": ["cu132", "cu131", "cu130"],
    "2.11": ["cu131", "cu130"],
    "2.10": ["cu128", "cu131"],
    "2.9": ["cu130", "cu128"],
    "2.8": ["cu128", "cu129"],
}
_RELEASE_ASSETS_CACHE: dict[str, list[str]] = {}
_ENABLE_LOCAL_FETCH_ENV = "TMOL_ENABLE_LOCAL_FETCH"
_FETCH_TIMEOUT_ENV = "TMOL_WHEEL_FETCH_TIMEOUT_S"
_FETCH_RETRIES_ENV = "TMOL_WHEEL_FETCH_RETRIES"
_FETCH_BACKOFF_ENV = "TMOL_WHEEL_FETCH_BACKOFF_S"


def _log(message: str) -> None:
    print(f"[tmol-build] {message}", file=sys.stderr)


def _env_true(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(parsed, minimum)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(parsed, minimum)


def _read_project_version() -> str:
    pyproject_path = _PROJECT_ROOT / "pyproject.toml"
    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        match = _PYPROJECT_VERSION_RE.match(raw_line.strip())
        if match:
            return match.group(1)
    raise RuntimeError("Could not read project version from pyproject.toml")


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _linux_arch_tag() -> str | None:
    machine = platform.machine().lower()
    return _SUPPORTED_ARCHES.get(machine)


def _torch_major_minor() -> str | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    version = str(getattr(torch, "__version__", ""))
    match = re.match(r"^(\d+\.\d+)", version)
    return match.group(1) if match else None


def _torch_cuda_tag() -> str | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    cuda = getattr(torch.version, "cuda", None)
    if not cuda:
        return None
    digits = re.sub(r"[^0-9]", "", str(cuda))
    if len(digits) < 3:
        return None
    return f"cu{digits[:3]}"


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _cuda_tag_variants(cuda_tag: str, torch_mm: str) -> list[str]:
    """CUDA wheel tags compatible with the detected runtime (driver-level forward compat)."""
    variants: list[str] = []
    for tag in _TORCH_CUDA_ALIASES.get(torch_mm, []):
        if tag not in variants:
            variants.append(tag)
    if cuda_tag not in variants:
        variants.append(cuda_tag)
    if cuda_tag.startswith("cu13"):
        for suffix in ("2", "1", "0"):
            alt = f"cu13{suffix}"
            if alt not in variants:
                variants.append(alt)
    elif cuda_tag.startswith("cu12"):
        for suffix in ("8", "9"):
            alt = f"cu12{suffix}"
            if alt not in variants:
                variants.append(alt)
    return variants


def _candidate_local_tags() -> list[str]:
    override = os.environ.get(_LOCAL_TAG_ENV, "").strip()
    if override:
        return [override]

    torch_mm = _torch_major_minor()
    cuda_tag = _torch_cuda_tag()
    candidates: list[str] = []
    if torch_mm and cuda_tag:
        for cuda_var in _cuda_tag_variants(cuda_tag, torch_mm):
            candidates.append(f"{cuda_var}torch{torch_mm}")
    elif torch_mm and not cuda_tag:
        candidates.append("cpu")
    elif _env_true(_ALLOW_CPU_FALLBACK_ENV):
        candidates.append("cpu")

    return _dedupe_keep_order(candidates)


def _candidate_wheel_filenames() -> list[str]:
    if platform.system().lower() != "linux":
        return []
    arch = _linux_arch_tag()
    if not arch:
        return []

    version = _read_project_version()
    py_tag = _python_tag()
    platform_tags: list[str]
    if arch == "x86_64":
        # Prefer manylinux (auditwheel-repaired) when published; linux_x86_64 for
        # legacy NGC-native release wheels.
        platform_tags = ["manylinux_2_28_x86_64", "linux_x86_64"]
    else:
        # aarch64 releases are currently native Linux; keep manylinux fallback
        # for forward compatibility if ARM auditwheel lanes are enabled.
        platform_tags = ["linux_aarch64", "manylinux_2_34_aarch64"]

    candidates: list[str] = []
    for local_tag in _candidate_local_tags():
        for platform_tag in platform_tags:
            candidates.append(
                f"tmol-{version}+{local_tag}-{py_tag}-{py_tag}-{platform_tag}.whl"
            )
    return _dedupe_keep_order(candidates)


def _release_download_base() -> str:
    return os.environ.get(_RELEASE_URL_ENV, _DEFAULT_RELEASE_BASE_URL).rstrip("/")


def _release_tag() -> str:
    return os.environ.get(_RELEASE_TAG_ENV, f"v{_read_project_version()}").strip()


def _is_repo_checkout() -> bool:
    return (_PROJECT_ROOT / ".git").exists()


def _is_isolated_build_environment() -> bool:
    haystack = " ".join(
        value
        for value in (sys.prefix, sys.executable, os.environ.get("VIRTUAL_ENV", ""))
        if value
    ).lower()
    return "pip-build-env-" in haystack


def _build_context_summary() -> str:
    return (
        f"platform={platform.system().lower()}, arch={platform.machine().lower()}, "
        f"python={_python_tag()}, torch={_torch_major_minor() or 'none'}, "
        f"cuda={_torch_cuda_tag() or 'none'}"
    )


def _download_to_path(url: str, out_path: Path) -> bool:
    timeout_s = _env_float(_FETCH_TIMEOUT_ENV, 20.0, minimum=1.0)
    retries = _env_int(_FETCH_RETRIES_ENV, 2, minimum=0)
    backoff_s = _env_float(_FETCH_BACKOFF_ENV, 1.5, minimum=0.0)
    total_attempts = retries + 1
    last_error: Exception | None = None

    request = Request(url, headers={"User-Agent": "tmol-build-backend/1"})
    if out_path.exists():
        out_path.unlink()

    for attempt in range(1, total_attempts + 1):
        if attempt > 1:
            _log(f"Retrying download ({attempt}/{total_attempts}): {url}")
        try:
            with urlopen(request, timeout=timeout_s) as response:
                if getattr(response, "status", 200) != 200:
                    _log(f"Wheel probe returned HTTP {response.status}: {url}")
                    return False
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    delete=False, dir=str(out_path.parent)
                ) as tmp:
                    shutil.copyfileobj(response, tmp)
                    tmp_path = Path(tmp.name)
            try:
                tmp_path.replace(out_path)
            except OSError:
                # uv/pip wheel dirs may live on a different mount than /tmp.
                shutil.move(str(tmp_path), str(out_path))
            return True
        except HTTPError as error:
            if error.code == 404:
                return False
            last_error = error
        except (URLError, TimeoutError) as error:
            last_error = error
        except Exception as error:
            last_error = error

        if attempt < total_attempts and backoff_s > 0.0:
            time.sleep(backoff_s * attempt)

    if last_error is not None:
        _log(f"Failed to fetch {url} after {total_attempts} attempts: {last_error}")
    return False


def _release_asset_names(tag: str) -> list[str]:
    if tag in _RELEASE_ASSETS_CACHE:
        return _RELEASE_ASSETS_CACHE[tag]

    api_url = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/tags/{tag}"
    request = Request(
        api_url,
        headers={
            "User-Agent": "tmol-build-backend/1",
            "Accept": "application/vnd.github+json",
        },
    )
    timeout_s = _env_float(_FETCH_TIMEOUT_ENV, 20.0, minimum=1.0)
    names: list[str] = []
    try:
        with urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = [
            asset["name"]
            for asset in payload.get("assets", [])
            if str(asset.get("name", "")).endswith(".whl")
        ]
    except HTTPError as error:
        _log(f"Release asset lookup failed for {tag}: HTTP {error.code}")
    except (URLError, TimeoutError, json.JSONDecodeError, KeyError) as error:
        _log(f"Release asset lookup failed for {tag}: {error}")

    _RELEASE_ASSETS_CACHE[tag] = names
    return names


def _platform_tags_for_arch(arch: str) -> list[str]:
    if arch == "x86_64":
        return ["manylinux_2_28_x86_64", "linux_x86_64"]
    return ["linux_aarch64", "manylinux_2_34_aarch64"]


def _score_wheel_filename(
    filename: str, local_tags: list[str], platform_tags: list[str]
) -> int:
    if not filename.endswith(".whl"):
        return -1
    parts = filename.removesuffix(".whl").split("-")
    if len(parts) < 5:
        return -1
    local_tag = parts[1]
    py_tag = parts[2]
    platform_tag = parts[-1]

    if py_tag != _python_tag():
        return -1
    if platform_tag not in platform_tags:
        return -1

    score = 0
    if local_tag in local_tags:
        score += 1000 - local_tags.index(local_tag) * 10
    elif local_tag == "cpu":
        score += 1
    else:
        return -1

    if platform_tag.startswith("manylinux"):
        score += 5
    return score


def _pick_wheel_from_release_assets(tag: str, filenames: list[str]) -> str | None:
    assets = set(_release_asset_names(tag))
    if not assets:
        return None
    for name in filenames:
        if name in assets:
            return name
    return None


def _can_build_from_source() -> bool:
    if _env_true(_ALLOW_SOURCE_BUILD_ENV):
        return True
    if shutil.which("nvcc") or shutil.which("CUDACXX"):
        return True
    # CPU-only torch can compile extensions without nvcc.
    return _torch_cuda_tag() is None


def _no_wheel_install_help(tag: str, filenames: list[str]) -> str:
    torch_mm = _torch_major_minor() or "?"
    cuda_tag = _torch_cuda_tag() or "none"
    py_tag = _python_tag()
    assets = _release_asset_names(tag)
    hints: list[str] = []
    if assets:
        for needle in (py_tag, f"torch{torch_mm}"):
            matches = [a for a in assets if needle in a][:4]
            if matches:
                hints.append(
                    f"  release wheels containing '{needle}': {', '.join(matches)}"
                )
    example = (
        filenames[0]
        if filenames
        else (
            f"tmol-{_read_project_version()}+cu132torch{torch_mm}-{py_tag}-{py_tag}-linux_x86_64.whl"
        )
    )
    base = _release_download_base()
    return (
        "No matching prebuilt tmol wheel for this environment "
        f"(python={py_tag}, torch={torch_mm}, cuda={cuda_tag}).\n"
        "Install a published wheel directly instead of compiling from source:\n"
        f"  pip install {base}/{tag}/{example}\n"
        "Or pin the variant:\n"
        f"  TMOL_WHEEL_LOCAL_TAG=cu132torch{torch_mm} pip install tmol=={_read_project_version()}\n"
        + ("\nAvailable on this release:\n" + "\n".join(hints) if hints else "")
    )


_WHEEL_LOCAL_VERSION_RE = re.compile(r"^tmol-(?P<base>[^-+]+)\+(?P<local>[^-]+)-")


def _version_from_wheel_filename(filename: str) -> str | None:
    match = _WHEEL_LOCAL_VERSION_RE.match(filename)
    if not match:
        return None
    return f"{match.group('base')}+{match.group('local')}"


def _align_wheel_metadata_from_filename(wheel_path: Path) -> None:
    """Rename dist-info to match +local tag in filename and patch METADATA/RECORD."""
    import base64
    import hashlib
    import zipfile

    target_version = _version_from_wheel_filename(wheel_path.name)
    if not target_version:
        return

    with zipfile.ZipFile(wheel_path, "r") as zin:
        entries = {name: zin.read(name) for name in zin.namelist()}

    metadata_paths = [n for n in entries if n.endswith(".dist-info/METADATA")]
    if not metadata_paths:
        _log("wheel metadata align skipped: no METADATA in wheel")
        return

    old_dist = metadata_paths[0].split("/")[0]
    new_dist = f"tmol-{target_version}.dist-info"
    if old_dist == new_dist:
        meta = entries[metadata_paths[0]].decode("utf-8")
        if any(ln == f"Version: {target_version}" for ln in meta.splitlines()):
            return

    new_entries: dict[str, bytes] = {}
    for name, data in entries.items():
        if name.endswith("/RECORD"):
            continue
        new_name = name.replace(f"{old_dist}/", f"{new_dist}/", 1)
        if name.endswith("/METADATA"):
            lines: list[str] = []
            for line in data.decode("utf-8").splitlines():
                if line.startswith("Version:"):
                    line = f"Version: {target_version}"
                lines.append(line)
            data = ("\n".join(lines) + "\n").encode("utf-8")
        new_entries[new_name] = data

    record_lines: list[str] = []
    for name in sorted(new_entries):
        payload = new_entries[name]
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        record_lines.append(f"{name},sha256={digest},{len(payload)}")
    record_lines.append(f"{new_dist}/RECORD,,")
    new_entries[f"{new_dist}/RECORD"] = "\n".join(record_lines).encode("utf-8")

    patched = wheel_path.with_suffix(".patched.whl")
    with zipfile.ZipFile(patched, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in new_entries.items():
            zout.writestr(name, data)
    patched.replace(wheel_path)


def _try_fetch_prebuilt_wheel(
    wheel_dir: Path, filenames: list[str], tag: str
) -> str | None:
    base = _release_download_base()
    for filename in filenames:
        url = f"{base}/{tag}/{filename}"
        out_path = wheel_dir / filename
        _log(f"Trying prebuilt wheel: {url}")
        if _download_to_path(url, out_path):
            _align_wheel_metadata_from_filename(out_path)
            _log(f"Downloaded prebuilt wheel: {filename}")
            return filename

    asset_match = _pick_wheel_from_release_assets(tag, filenames)
    if asset_match:
        url = f"{base}/{tag}/{asset_match}"
        out_path = wheel_dir / asset_match
        _log(f"Trying release asset match: {url}")
        if _download_to_path(url, out_path):
            _align_wheel_metadata_from_filename(out_path)
            _log(f"Downloaded prebuilt wheel: {asset_match}")
            return asset_match
    return None


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build or fetch a wheel for the current environment."""
    _log(f"Build context: {_build_context_summary()}")
    isolated_build = _is_isolated_build_environment()
    local_tag_override = os.environ.get(_LOCAL_TAG_ENV, "").strip()

    if _is_repo_checkout() and not _env_true(_ENABLE_LOCAL_FETCH_ENV):
        _log("Source checkout detected; skipping wheel fetch and building locally.")
        return _skbuild_backend.build_wheel(
            wheel_directory,
            config_settings=config_settings,
            metadata_directory=metadata_directory,
        )

    if _env_true(_FORCE_BUILD_ENV) or _env_true(_DISABLE_FETCH_ENV):
        _log("Prebuilt wheel fetch disabled; building locally.")
        return _skbuild_backend.build_wheel(
            wheel_directory,
            config_settings=config_settings,
            metadata_directory=metadata_directory,
        )

    wheel_dir = Path(wheel_directory)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    filenames = _candidate_wheel_filenames()
    if filenames:
        _log("Wheel candidates: " + ", ".join(filenames))
        if isolated_build and not local_tag_override:
            _log(
                "Detected pip isolated build environment. "
                "Auto-detected torch/cuda may differ from your runtime env; "
                "use TMOL_WHEEL_LOCAL_TAG to pin an exact wheel variant."
            )
        tag = _release_tag()
        fetched = _try_fetch_prebuilt_wheel(wheel_dir, filenames, tag)
        if fetched:
            return fetched

        if not _can_build_from_source():
            raise RuntimeError(_no_wheel_install_help(tag, filenames))
        _log("No matching prebuilt wheel found; falling back to local build.")
    else:
        _log(
            "No compatible wheel naming candidates for this platform; building locally."
        )
        if not _can_build_from_source():
            raise RuntimeError(
                "tmol cannot compile CUDA extensions on this platform (no nvcc). "
                "Use a prebuilt wheel from GitHub Releases or set TMOL_WHEEL_LOCAL_TAG."
            )

    return _skbuild_backend.build_wheel(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _skbuild_backend.build_editable(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _skbuild_backend.build_sdist(
        sdist_directory,
        config_settings=config_settings,
    )


def get_requires_for_build_wheel(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _skbuild_backend.get_requires_for_build_wheel(
        config_settings=config_settings
    )


def get_requires_for_build_editable(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _skbuild_backend.get_requires_for_build_editable(
        config_settings=config_settings
    )


def get_requires_for_build_sdist(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _skbuild_backend.get_requires_for_build_sdist(
        config_settings=config_settings
    )


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _skbuild_backend.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings=config_settings,
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _skbuild_backend.prepare_metadata_for_build_editable(
        metadata_directory,
        config_settings=config_settings,
    )
