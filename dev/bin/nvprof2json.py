#!/usr/bin/env python3

import sqlite3
import argparse
import json
import subprocess
import os
import sys
import copy


def main():
    parser = argparse.ArgumentParser(
        description="Convert nvprof output to Google Event Trace compatible JSON."
    )
    parser.add_argument("filename")
    args = parser.parse_args()

    conn = sqlite3.connect(args.filename)
    conn.row_factory = sqlite3.Row

    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        strings[r["id"]] = demangle(r["value"])

    traceEvents = []

    """
    _id_: 11625
    cbid: 17
    start: 1496933427584362152
    end: 1496933427584362435
    processId: 1317533
    threadId: 1142654784
    correlationId: 13119
    returnValue: 0
    """
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        # eprintRow(row)
        if row["cbid"] in cbid_table:
            cbid = cbid_table[row["cbid"]]
        else:
            cbid = str(row["cbid"])
            eprint("Unrecognized cbid {}".format(cbid))
        event = {
            "name": cbid,
            "ph": "X",  # Complete Event (Begin + End event)
            "cat": "cuda",
            "ts": munge_time(row["start"]),
            "dur": munge_time(row["end"] - row["start"]),
            "tid": "Thread {}: Runtime API".format(row["threadId"]),
            "pid": "[{}] Process".format(row["processId"]),
            "args": {
                # TODO: More
            },
        }
        traceEvents.append(event)

    # TODO DRIVER

    """
    _id_: 1
    flags: 2
    timestamp: 1496844806028263989
    id: 1
    objectKind: 2
    objectId: b'\xe5\xc0\x16\x00@\xe7\x10J\x00\x00\x00\x00'
    name: 3
    domain: 0
    """
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MARKER WHERE name != 0"):
        # eprintRow(row)
        # TODO: SO INEFFICIENT
        end_row = conn.execute(
            "SELECT * FROM CUPTI_ACTIVITY_KIND_MARKER WHERE id = ? AND name = 0",
            (row["id"],),
        ).fetchone()
        # TODO: support discrete events
        event = {
            "name": strings[row["name"]],
            "ph": "X",  # Complete Event (Begin + End event)
            "cat": "cuda",
            "ts": munge_time(row["timestamp"]),
            "dur": munge_time(end_row["timestamp"] - row["timestamp"]),
            # Weirdly, these don't seem to be associated with a
            # CPU/GPU.  I guess there's no CUDA Context available
            # when you run these, so it makes sense.  But nvvp
            # associates these with a GPU strangely enough
            "tid": "Markers and Ranges",
            "pid": "Markers and Ranges",
            # TODO: NO COLORS FOR YOU (probably have to parse
            # objectId)
            "args": {
                # TODO: More
            },
        }
        traceEvents.append(event)

    """
    _id_: 1
    copyKind: 1
    srcKind: 1
    dstKind: 3
    flags: 0
    bytes: 7436640
    start: 1496933426915778221
    end: 1496933426916558424
    deviceId: 0
    contextId: 1
    streamId: 7
    correlationId: 809
    runtimeCorrelationId: 0
    """
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        # copyKind:
        #   1 - Memcpy HtoD
        #   2 - Memcpy DtoH
        #   8 - Memcpy DtoD
        # flags: ???
        #   0 - Sync
        #   1 - Async
        # srcKind/dstKind
        #   1 - Pageable
        #   2 - Page-locked ???
        #   3 - Device
        # eprintRow(row)
        if row["copyKind"] == 1:
            copyKind = "HtoD"
        elif row["copyKind"] == 2:
            copyKind = "DtoH"
        elif row["copyKind"] == 8:
            copyKind = "DtoD"
        else:
            copyKind = str(row["copyKind"])
        if row["flags"] == 0:
            flags = "sync"
        elif row["flags"] == 1:
            flags = "async"
        else:
            flags = str(row["flags"])
        event = {
            "name": "Memcpy {} [{}]".format(copyKind, flags),
            "ph": "X",  # Complete Event (Begin + End event)
            "cat": "cuda",
            "ts": munge_time(row["start"]),
            "dur": munge_time(row["end"] - row["start"]),
            "tid": "MemCpy ({})".format(copyKind),
            # TODO: lookup GPU name.  This is tored in
            # CUPTI_ACTIVITY_KIND_DEVICE
            "pid": "[{}:{}] Overview".format(row["deviceId"], row["contextId"]),
            "args": {
                "Size": sizeof_fmt(row["bytes"]),
                # TODO: More
            },
        }
        traceEvents.append(event)

    # name: index into StringTable
    # What is thed difference between end and completed?
    """
    _id_: 1
    cacheConfig: b'\x00'
    sharedMemoryConfig: 1
    registersPerThread: 32
    partitionedGlobalCacheRequested: 2
    partitionedGlobalCacheExecuted: 2
    start: 1496844806032514222
    end: 1496844806032531694
    completed: 1496844806032531694
    deviceId: 0
    contextId: 1
    streamId: 7
    gridX: 57
    gridY: 1
    gridZ: 1
    blockX: 128
    blockY: 1
    blockZ: 1
    staticSharedMemory: 0
    dynamicSharedMemory: 0
    localMemoryPerThread: 0
    localMemoryTotal: 78643200
    correlationId: 487
    gridId: 669
    name: 5
    """
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
        # eprint(strings[row["name"]])
        # eprintRow(row)
        event = {
            "name": strings[row["name"]],
            "ph": "X",  # Complete Event (Begin + End event)
            "cat": "cuda",
            "ts": munge_time(row["start"]),
            "dur": munge_time(row["end"] - row["start"]),
            "tid": "Compute",
            # TODO: lookup GPU name
            "pid": "[{}:{}] Overview".format(row["deviceId"], row["contextId"]),
            "args": {
                "Grid size": "[ {}, {}, {} ]".format(
                    row["gridX"], row["gridY"], row["gridZ"]
                ),
                "Block size": "[ {}, {}, {} ]".format(
                    row["blockX"], row["blockY"], row["blockZ"]
                ),
                # TODO: More
            },
        }
        alt_event = copy.deepcopy(event)
        alt_event["tid"] = alt_event["name"]
        alt_event["pid"] = "[{}:{}] Compute".format(row["deviceId"], row["contextId"])
        traceEvents.append(event)
        traceEvents.append(alt_event)

    json.dump(traceEvents, sys.stdout)
    print()


def munge_time(t):
    """Take a time from nvprof and convert it into a chrome://tracing time."""
    # For strict correctness, divide by 1000, but this reduces accuracy.
    return t  # / 1000.


def demangle(name):
    """Demangle a C++ identifier using c++filt"""
    # TODO: create the process only once.
    # Fortunately, this doesn't seem to be a bottleneck ATM.
    try:
        with open(os.devnull, "w") as devnull:
            return (
                subprocess.check_output(["c++filt", "-n", name], stderr=devnull)
                .rstrip()
                .decode("ascii")
            )
    except subprocess.CalledProcessError:
        return name


cbid_table = {
    16: "cudaSetDevice",
    10: "cudaGetLastError",
    13: "Launch",
    9: "cudaSetupArgument",
    8: "cudaConfigureArgument",
    17: "cudaGetDevice",
    20: "cudaMalloc",
    22: "cudaFree",
    4: "cudaGetDeviceProperties",
    # TODO: add more
    # 31
    # 41
    # 51
    # 55
    # 58
}


def sizeof_fmt(num, suffix="B"):
    """Format size with metric units (like nvvp)"""
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1000.0
    return "%.1f%s%s" % (num, "Y", suffix)


def eprintRow(row):
    """Print a sqlite3.Row to stderr."""
    for k in row.keys():
        eprint("{}: {}".format(k, row[k]))
    eprint("----")


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def inspect_db(conn):
    """Quickly dump data of all tables in database, with field names."""
    tables = """
CUPTI_ACTIVITY_KIND_BRANCH
CUPTI_ACTIVITY_KIND_CDP_KERNEL
CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
CUPTI_ACTIVITY_KIND_CONTEXT
CUPTI_ACTIVITY_KIND_CUDA_EVENT
CUPTI_ACTIVITY_KIND_DEVICE
CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE
CUPTI_ACTIVITY_KIND_DRIVER
CUPTI_ACTIVITY_KIND_ENVIRONMENT
CUPTI_ACTIVITY_KIND_EVENT
CUPTI_ACTIVITY_KIND_EVENT_INSTANCE
CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION
CUPTI_ACTIVITY_KIND_FUNCTION
CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS
CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT
CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE
CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC
CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE
CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION
CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION
CUPTI_ACTIVITY_KIND_KERNEL
CUPTI_ACTIVITY_KIND_MARKER
CUPTI_ACTIVITY_KIND_MARKER_DATA
CUPTI_ACTIVITY_KIND_MEMCPY
CUPTI_ACTIVITY_KIND_MEMCPY2
CUPTI_ACTIVITY_KIND_MEMSET
CUPTI_ACTIVITY_KIND_METRIC
CUPTI_ACTIVITY_KIND_METRIC_INSTANCE
CUPTI_ACTIVITY_KIND_MODULE
CUPTI_ACTIVITY_KIND_NAME
CUPTI_ACTIVITY_KIND_NVLINK
CUPTI_ACTIVITY_KIND_OPENACC_DATA
CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH
CUPTI_ACTIVITY_KIND_OPENACC_OTHER
CUPTI_ACTIVITY_KIND_OVERHEAD
CUPTI_ACTIVITY_KIND_PC_SAMPLING
CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO
CUPTI_ACTIVITY_KIND_PREEMPTION
CUPTI_ACTIVITY_KIND_RUNTIME
CUPTI_ACTIVITY_KIND_SHARED_ACCESS
CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR
CUPTI_ACTIVITY_KIND_STREAM
CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
""".strip().split(
        "\n"
    )

    for t in tables:
        eprint(t)
        for r in conn.execute("SELECT * FROM {} LIMIT 4".format(t)):
            eprintRow(r)
        eprint("----")
        eprint("----")
        eprint("----")
        eprint("----")


if __name__ == "__main__":
    main()
