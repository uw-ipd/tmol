"""Profile cache lookup cost and retained memory after database churn."""

import gc
import json
from pathlib import Path
import statistics
import subprocess
import time
import tracemalloc

import attr
from tmol.database import ParameterDatabase
from tmol.ligand import _polymer_profile as current
from tmol.utility.weak_identity_cache import WeakIdentityLRU


def main():
    baseline = "09258fbad"
    path = "tmol/ligand/_polymer_profile.py"
    old = {}
    exec(
        compile(
            subprocess.check_output(["git", "show", baseline + ":" + path], text=True),
            path,
            "exec",
        ),
        old,
    )
    database = ParameterDatabase.get_default().chemical
    current._POLYMER_PROFILE_CACHE = WeakIdentityLRU()
    functions = {
        "before": (old["alpha_profile"], old["na_profile"]),
        "after": (current.alpha_profile, current.na_profile),
    }
    expected = (
        old["alpha_profile"](database),
        old["na_profile"](database, "dna"),
        old["na_profile"](database, "rna"),
    )
    rows = {}
    for name, (alpha, na) in functions.items():
        assert tuple(
            attr.asdict(p)
            for p in (alpha(database), na(database, "dna"), na(database, "rna"))
        ) == tuple(attr.asdict(p) for p in expected)
        timings = []
        for _ in range(7):
            start = time.perf_counter()
            for _ in range(10000):
                alpha(database)
                na(database, "dna")
                na(database, "rna")
            timings.append((time.perf_counter() - start) / 30000)
        old["_ALPHA_PROFILE_CACHE"].clear()
        old["_NA_PROFILE_CACHE"].clear()
        current._POLYMER_PROFILE_CACHE = WeakIdentityLRU()
        gc.collect()
        tracemalloc.start()
        owners = [attr.evolve(database) for _ in range(100)]
        for owner in owners:
            alpha(owner)
            na(owner, "dna")
            na(owner, "rna")
        live_bytes, peak_bytes = tracemalloc.get_traced_memory()
        del owner, owners
        gc.collect()
        retained_bytes, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        entries = (
            len(old["_ALPHA_PROFILE_CACHE"]) + len(old["_NA_PROFILE_CACHE"])
            if name == "before"
            else len(current._POLYMER_PROFILE_CACHE)
        )
        rows[name] = dict(
            warm_seconds_per_lookup=timings,
            warm_median=statistics.median(timings),
            live_bytes=live_bytes,
            peak_bytes=peak_bytes,
            retained_bytes=retained_bytes,
            retained_entries=entries,
        )
    result = dict(
        baseline=baseline,
        databases=100,
        profiles_per_database=3,
        profiles_equal=True,
        measurements="tracemalloc Python allocations; seven warm 30000-lookup samples",
        rows=rows,
    )
    output = Path(__file__).parent / "results/polymer-cache.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
