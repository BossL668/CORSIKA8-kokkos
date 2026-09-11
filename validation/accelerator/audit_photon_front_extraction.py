#!/usr/bin/env python3
"""Check that only the photon producer block was extracted, without changing tokens.

Compare with a separately frozen pre-refactor function, not the same new helper
called twice. This structural check complements real-device record comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def tokens(text):
    return re.sub(r"\s+", "", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    old = args.baseline.read_text()
    new = args.source.read_text()
    marker = "  template <class ExecutionSpace>\n  gpu::em::ResidentPhotonCascadeResult "
    old_function = old[old.index(marker):].replace("runResidentPhotonCascadeBefore(",
                                                  "runResidentPhotonCascade(")
    new_function = new[new.index(marker):]
    begin = old_function.index("      auto const selections_raw = rawDeviceView(selections);")
    end = old_function.index("      // Scan totals, compacted interaction count", begin)
    old_block = old_function[begin:end]
    expected_helper = old_block.replace(
        "      auto const next = queue.next().rawDeviceView();\n", "").replace(
            "active_workspace.", "workspace.")
    extracted = new[new.index("  void enqueueResidentPhotonFront("):new.index(marker)]
    extracted = extracted[extracted.index("    auto const selections_raw"):].rstrip()
    # Final brace belongs to the helper, not the moved kernel block.
    extracted = extracted[:-1]
    call = """      enqueueResidentPhotonFront(
          physics_context, current, queue.next().rawDeviceView(),
          active_workspace, current_count, capacity, next_history_id,
          project_steps, capture_first_interaction, execution,
          openmp_chunk_size);

"""
    expected_function = old_function[:begin] + call + old_function[end:]
    result = {
        "baseline_sha256": hashlib.sha256(old.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(new.encode()).hexdigest(),
        "producer_kernel_tokens_unchanged": tokens(expected_helper) == tokens(extracted),
        "synchronous_control_flow_unchanged_except_helper_call":
            tokens(expected_function) == tokens(new_function),
        "scope": "photon producer extraction; not full shower or physical model acceptance",
    }
    result["pass"] = (result["producer_kernel_tokens_unchanged"] and
                      result["synchronous_control_flow_unchanged_except_helper_call"])
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
