#!/usr/bin/env python3
"""Plot actual CUPTI kernel timestamps, never inferred device utilization."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--log", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--threads", type=int, default=4,
                    help="Actual OpenMP thread count of the recorded test")
kind = parser.add_mutually_exclusive_group()
kind.add_argument("--photon-front", action="store_true",
                    help="Plot the real photon producer kernel sequence instead of the arithmetic probe")
kind.add_argument("--lepton-front", action="store_true",
                  help="Plot real PROPOSAL lepton kernels with endpoint-local profile/radio checks")
args = parser.parse_args()
records = []
for line in args.log.read_text().splitlines():
    start = line.find('{"timeline":')
    if start >= 0:
        records.append(json.loads(line[start:]))
if len(records) != 1:
    raise SystemExit("Need exactly one complete CUPTI timeline")
record = records[0]
if record["dropped_records"] or record["overlap_ns"] <= 0:
    raise SystemExit("No complete actual-overlap evidence")
h0, h1 = record["host_start_ns"], record["host_end_ns"]
host_windows = record.get("host_windows", [[h0, h1]])
kernels = [k for k in record["cuda_kernels"] if k["end_ns"] > h0 and k["start_ns"] < h1]
if not kernels or (not (args.photon_front or args.lepton_front) and len(kernels) != 1):
    raise SystemExit("Expected one CUDA probe kernel overlapping host work")
if args.photon_front or args.lepton_front:
    kernels = record["cuda_kernels"]
origin = min(h0, min(k["start_ns"] for k in kernels))
ns_to_ms = lambda t: (t - origin) * 1e-6
args.output.mkdir(parents=True, exist_ok=False)
fig, ax = plt.subplots(figsize=(8.5, 2.8))
ax.broken_barh([(ns_to_ms(k["start_ns"]), (k["end_ns"] - k["start_ns"]) * 1e-6) for k in kernels],
               (1, .5), facecolors="#377eb8", label="CUPTI device timestamps")
ax.broken_barh([(ns_to_ms(a), (b-a)*1e-6) for a,b in host_windows],
               (0, .5), facecolors="#e69f00")
for kernel in kernels:
    for h0i, h1i in host_windows:
        a, b = max(h0i, kernel["start_ns"]), min(h1i, kernel["end_ns"])
        if b > a:
            ax.axvspan(ns_to_ms(a), ns_to_ms(b), color="grey", alpha=.15)
gpu_label = ("CUDA lepton kernels" if args.lepton_front else
             "CUDA photon kernels" if args.photon_front else "CUDA kernel")
ax.set_yticks([.25, 1.25], [f"OpenMP ({args.threads} threads)", gpu_label])
ax.set_ylim(-.2, 1.75)
ax.set_xlim(-.1, ns_to_ms(max(h1, max(k["end_ns"] for k in kernels))) + .2)
ax.set_xlabel("Time on the common CUPTI clock [ms, offset removed]")
ax.set_title(f"Actual overlap: {record['overlap_ns'] * 1e-6:.3f} ms")
ax.grid(axis="x", alpha=.2)
caption = ("Real PROPOSAL lepton front; endpoint-local profile/radio verified. Not a shower speed-up."
           if args.lepton_front else
           "Real photon producer kernels with an analytic rate fixture; not a full-shower benchmark."
           if args.photon_front else "Arithmetic primitive only; not an EM shower or speed-up measurement.")
fig.text(.5, .01, caption,
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .05, 1, 1))
fig.savefig(args.output / "actual_kernel_overlap.png", dpi=170)
fig.savefig(args.output / "actual_kernel_overlap.svg")
(args.output / "timeline.json").write_text(json.dumps(record, indent=2) + "\n")
print(args.output)
