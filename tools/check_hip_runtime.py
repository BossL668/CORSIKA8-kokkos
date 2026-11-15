#!/usr/bin/env python3
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
# Distributed under the 3-clause BSD license; see LICENSE.
"""Check HIP device visibility before building CORSIKA; no compiler is needed."""

import ctypes
import sys


def main():
    try:
        runtime = ctypes.CDLL("libamdhip64.so")
    except OSError as error:
        print(f"Cannot load the HIP runtime: {error}", file=sys.stderr)
        print("Source the selected ROCm/DTK env.sh and check its shared-library "
              "dependencies.", file=sys.stderr)
        return 1

    try:
        get_count = runtime.hipGetDeviceCount
        get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_count.restype = ctypes.c_int
        get_error = runtime.hipGetErrorString
        get_error.argtypes = [ctypes.c_int]
        get_error.restype = ctypes.c_char_p
        count = ctypes.c_int(0)
        status = get_count(ctypes.byref(count))
        if status != 0 or count.value == 0:
            message = get_error(status)
            reason = message.decode(errors="replace") if message else "unknown error"
            print(f"HIP runtime unavailable: {reason} (status={status}, "
                  f"visible devices={count.value}).", file=sys.stderr)
            print("Check the vendor kernel driver, device-file permissions and "
                  "device visibility settings. Installing the SDK or adding "
                  "a group cannot replace a missing kernel driver.", file=sys.stderr)
            return 1
    except AttributeError as error:
        print(f"The selected library does not provide the HIP runtime API: {error}",
              file=sys.stderr)
        return 1

    print(f"HIP runtime query passed: {count.value} visible device(s).")
    print("This verifies device discovery only. After building, run c8_hip_devices "
          "and a small CORSIKA event to verify device execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
