#!/usr/bin/env python3
"""Check that resumable fronts reuse unchanged production kernel bodies."""
import argparse
from pathlib import Path
import hashlib
import json
import re

def tokens(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    old=a.baseline.read_text()
    new=a.source.read_text()
    def between(s,start,end):
        i=s.index(start)
        return s[i:s.index(end,i)]
    original_physics=between(old,"      // Keep rate inversion separate",
                              "      // The scans and final-state writers above")
    actual_physics=between(new,"    // Keep rate inversion separate",
        "  /** Same post-materialization profile")
    # Strip only the helper's closing brace, not any kernel contents.
    actual_physics=actual_physics[:actual_physics.rfind("  }")]
    old_accum=between(old,"      if (profile_accumulator != nullptr) {",
                      "      auto const projected_count =")
    after_accum=new[new.index("  void enqueueResidentLeptonAccumulation("):]
    actual_accum=between(after_accum,"    if (profile_accumulator != nullptr) {",
                         "  template <class ExecutionSpace>\n  gpu::em::ResidentLeptonCascadeResult")
    actual_accum=actual_accum[:actual_accum.rfind("  }")]
    expected=old[old.index("  template <class ExecutionSpace>"):]
    expected=expected.replace("runResidentLeptonCascadeBefore(", "runResidentLeptonCascade(")
    expected=expected.replace(original_physics,
        """      enqueueResidentLeptonFront(
          physics_context, electron_mass_GeV, air_moliere_fast_path, current,
          active_workspace, current_count, capacity, photon_capacity,
          next_history_id, secondary_history_id_limit_exclusive,
          project_steps, capture_first_interaction, execution, openmp_chunk_size);

""")
    expected=expected.replace(old_accum,
        """      enqueueResidentLeptonAccumulation(
          active_workspace, current_count, front_control, profile_projection,
          profile_accumulator, radio_accumulator, execution, openmp_chunk_size);
""")
    actual=new[new.index("  template <class ExecutionSpace>\n  gpu::em::ResidentLeptonCascadeResult runResidentLeptonCascade("):]
    report={"baseline_sha256":hashlib.sha256(old.encode()).hexdigest(),
            "source_sha256":hashlib.sha256(new.encode()).hexdigest(),
            "physics_kernel_tokens_unchanged":tokens(original_physics)==tokens(actual_physics),
            "accumulation_kernel_tokens_unchanged":tokens(old_accum)==tokens(actual_accum),
            "synchronous_wrapper_except_helper_calls_unchanged":tokens(expected)==tokens(actual)}
    report["pass"]=all(v for k,v in report.items() if k.endswith("unchanged"))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as f: json.dump(report,f,indent=2);f.write("\n")
    print(json.dumps(report))
    return 0 if report["pass"] else 1
if __name__=="__main__": raise SystemExit(main())

