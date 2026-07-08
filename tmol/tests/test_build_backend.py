import io
from pathlib import Path
from urllib.error import URLError

import tmol_build_backend as backend


def test_build_wheel_uses_downloaded_wheel_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "_is_repo_checkout", lambda: False)
    monkeypatch.setattr(backend, "_is_isolated_build_environment", lambda: False)
    monkeypatch.setattr(
        backend, "_candidate_wheel_filenames", lambda: ["candidate.whl"]
    )
    monkeypatch.setattr(
        backend, "_release_download_base", lambda: "https://example.invalid"
    )
    monkeypatch.setattr(backend, "_release_tag", lambda: "v0.0.0")

    download_calls: list[tuple[str, Path]] = []

    def fake_download(url, out_path):
        download_calls.append((url, out_path))
        return True

    monkeypatch.setattr(backend, "_download_to_path", fake_download)
    monkeypatch.setattr(
        backend._skbuild_backend,
        "build_wheel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("build_wheel fallback should not run when download succeeds")
        ),
    )

    wheel_name = backend.build_wheel(str(tmp_path))

    assert wheel_name == "candidate.whl"
    assert download_calls == [
        ("https://example.invalid/v0.0.0/candidate.whl", tmp_path / "candidate.whl")
    ]


def test_build_wheel_falls_back_when_no_prebuilt_match(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "_is_repo_checkout", lambda: False)
    monkeypatch.setattr(backend, "_is_isolated_build_environment", lambda: False)
    monkeypatch.setattr(
        backend, "_candidate_wheel_filenames", lambda: ["candidate.whl"]
    )
    monkeypatch.setattr(
        backend, "_release_download_base", lambda: "https://example.invalid"
    )
    monkeypatch.setattr(backend, "_release_tag", lambda: "v0.0.0")
    monkeypatch.setattr(backend, "_download_to_path", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        backend._skbuild_backend,
        "build_wheel",
        lambda *args, **kwargs: "built-locally.whl",
    )

    wheel_name = backend.build_wheel(str(tmp_path))

    assert wheel_name == "built-locally.whl"


def test_build_wheel_skips_fetch_in_repo_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("TMOL_ENABLE_LOCAL_FETCH", raising=False)
    monkeypatch.setattr(backend, "_is_repo_checkout", lambda: True)
    monkeypatch.setattr(
        backend,
        "_candidate_wheel_filenames",
        lambda: (_ for _ in ()).throw(AssertionError("fetch path should be skipped")),
    )
    monkeypatch.setattr(
        backend._skbuild_backend,
        "build_wheel",
        lambda *args, **kwargs: "built-locally.whl",
    )

    wheel_name = backend.build_wheel(str(tmp_path))

    assert wheel_name == "built-locally.whl"


def test_build_wheel_force_build_env_skips_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("TMOL_FORCE_BUILD", "1")
    monkeypatch.setattr(backend, "_is_repo_checkout", lambda: False)
    monkeypatch.setattr(backend, "_is_isolated_build_environment", lambda: False)
    monkeypatch.setattr(
        backend,
        "_candidate_wheel_filenames",
        lambda: (_ for _ in ()).throw(AssertionError("fetch path should be skipped")),
    )
    monkeypatch.setattr(
        backend._skbuild_backend,
        "build_wheel",
        lambda *args, **kwargs: "built-locally.whl",
    )

    wheel_name = backend.build_wheel(str(tmp_path))

    assert wheel_name == "built-locally.whl"


def test_build_wheel_attempts_autodetect_in_isolated_build_by_default(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("TMOL_WHEEL_LOCAL_TAG", raising=False)
    monkeypatch.setattr(backend, "_is_repo_checkout", lambda: False)
    monkeypatch.setattr(backend, "_is_isolated_build_environment", lambda: True)
    monkeypatch.setattr(
        backend, "_candidate_wheel_filenames", lambda: ["candidate.whl"]
    )
    monkeypatch.setattr(
        backend, "_release_download_base", lambda: "https://example.invalid"
    )
    monkeypatch.setattr(backend, "_release_tag", lambda: "v0.0.0")
    monkeypatch.setattr(backend, "_download_to_path", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        backend._skbuild_backend,
        "build_wheel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("build_wheel fallback should not run when download succeeds")
        ),
    )

    wheel_name = backend.build_wheel(str(tmp_path))

    assert wheel_name == "candidate.whl"


def test_download_retries_then_succeeds(monkeypatch, tmp_path):
    out_path = tmp_path / "wheel.whl"
    monkeypatch.setenv("TMOL_WHEEL_FETCH_RETRIES", "2")
    monkeypatch.setenv("TMOL_WHEEL_FETCH_BACKOFF_S", "0")
    monkeypatch.setenv("TMOL_WHEEL_FETCH_TIMEOUT_S", "1")

    attempts = {"count": 0}

    class _Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def fake_urlopen(_request, timeout):
        assert timeout == 1.0
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise URLError("temporary network error")
        return _Response(b"wheel-bytes")

    monkeypatch.setattr(backend, "urlopen", fake_urlopen)

    assert (
        backend._download_to_path("https://example.invalid/wheel.whl", out_path) is True
    )
    assert attempts["count"] == 3
    assert out_path.read_bytes() == b"wheel-bytes"
