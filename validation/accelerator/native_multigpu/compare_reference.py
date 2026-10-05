"""Optional two-GPU oracle. Python is used by this TEST only, never by the native run."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq

FILES = (
    "profile/profile.parquet", "production_profile/profile.parquet",
    "energyloss/dEdX.parquet", "CoREAS/observers.parquet",
    "ZHS/observers.parquet", "particles/particles.parquet",
    "interactions/interactions.parquet",
)

def compare(left, right):
    report = {}
    for name in FILES:
        a, b = pq.read_table(left/name), pq.read_table(right/name)
        assert a.column_names == b.column_names and a.num_rows == b.num_rows, name
        maximum = 0.0
        for col in a.column_names:
            x, y = a[col].to_numpy(), b[col].to_numpy()
            assert np.array_equal(x, y), (name, col, np.max(np.abs(x-y)))
            if x.size:
                maximum = max(maximum, float(np.max(np.abs(x.astype(float)-y.astype(float)))))
        report[name] = {"rows": a.num_rows, "maximum_absolute_difference": maximum}
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="c8-native-shower-"))
    root.mkdir(exist_ok=True)
    binary, previous = args.binary.resolve(), args.previous.resolve()
    source = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "reference", source/"applications/detail/air_shower_multigpu/run_multigpu.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    physics = ["-p", "22", "-E", "10", "--emthin", "0", "--ring", "1",
               "--antenna-file", "/dev/null", "-v", "warn"]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    def run(name, executable, extra):
        with (root/(name+".log")).open("x") as log:
            subprocess.run([str(executable)]+physics+extra+["-f",str(root/name)],
                           check=True, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600)
    print("Regression artifacts:", root, flush=True)
    run("native", binary, ["--device","0,1","--gpu-memory-fraction","0.20","-N","2","-s","26092441"])
    first = root/"native/seed_26092441"
    cfg = json.loads((first/"CONFIG.json").read_text())
    cfg["output"] = str(root/"python_reference")
    cfg["timeout_s"] = 600
    (root/"python_reference.json").write_text(json.dumps(cfg))
    reference.run(root/"python_reference.json")
    result = {"native_vs_python": compare(first/"merged", root/"python_reference/merged")}
    assert json.loads((first/"PARTITION.json").read_text()) == json.loads(
        (root/"python_reference/PARTITION.json").read_text())
    for i in range(2):
        assert reference.digest(first/("roots_%d.txt" % i)) == reference.digest(
            root/"python_reference"/("roots_%d.txt" % i))
    for execution in ("cuda", "openmp"):
        extra = ["--em-backend","kokkos","--radio-backend","kokkos","--kokkos-execution",execution,
                 "--kokkos-num-threads","1","--gpu-memory-fraction","0.20","-s","26092443"]
        run("old_"+execution, previous, extra)
        run("new_"+execution, binary, extra)
        result[execution+"_unchanged"] = compare(root/("old_"+execution), root/("new_"+execution))
    result["native_events"] = 2
    result["coordinator"] = "C++"
    (root/"REGRESSION.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
