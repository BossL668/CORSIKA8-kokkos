#!/usr/bin/env python3
"""PSR-only preparation: freeze provenance and reuse the exact three old natural events."""
import copy, hashlib, json, pathlib, socket, yaml
ROOT = pathlib.Path("/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912")
OLD = ROOT.parent / "beta5_tau_mass_energy_ledger_20260910_v2"
BUNDLE = ROOT.parent / "beta5_nutau100PeV_openmp120_20260910_v1/bundle"
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1048576), b""): h.update(block)
    return h.hexdigest()
def main():
    if not ROOT.is_dir() or not pathlib.Path("/data/yhlu").is_dir():
        raise RuntimeError("Run on PSR only")
    scene=yaml.safe_load((BUNDLE/"scene.yaml").read_text())
    mesh=pathlib.Path(scene["geometry"]["mesh_path"])
    assert digest(mesh) == "5060f55efc6a3dae8ab94c6ef2a27356521b56797ed059a1c150c3a47c2d7c1a"
    selected=["E01","E20","N10"]
    scene["radio"]={**scene.get("radio",{}), "enabled":True,
        "observers":[o for name in selected for o in scene["radio"]["observers"] if o["name"]==name],
        "start_ns":-1000., "sample_rate_GHz":.256, "samples":131072,
        "moment_order":12, "subdivision_frequency_GHz":.256,
        "fraunhofer_limit":.025, "maximum_subdivision_depth":20,
        "mesh_maximum_segment_m":.1, "memory_MiB":1024,
        "air_index_model":"native", "index_table_step_m":10.,
        "optical_integration_samples":64, "rock_attenuation_length_m":100.}
    assert len(scene["radio"]["observers"]) == 3
    scene["provenance"]["radio_disabled"]=False
    scene["provenance"]["radio_note"]="Three original stations; 100 m FIELD attenuation length is an explicit model assumption, not a measured site value."
    scene["radio"].pop("algorithm",None)
    (ROOT/"scene_radio.yaml").write_text(yaml.safe_dump(scene,sort_keys=False))
    off=copy.deepcopy(scene);off["radio"]["enabled"]=False
    (ROOT/"scene_off.yaml").write_text(yaml.safe_dump(off,sort_keys=False))
    rows=[]
    for seed in [158,946,3605]:
        old=json.loads((OLD/("command_"+str(seed)+".json")).read_text())
        s=yaml.safe_load((OLD/"runs"/("seed"+str(seed))/"terrain_run.yaml").read_text())
        decay=s["tau"]["decays"][0]
        pdgs=[abs(x["pdg"]) for x in decay["daughters"]]
        category="muonic_control" if 13 in pdgs else "electronic_second_cascade" if 11 in pdgs else "hadronic_second_cascade"
        rows.append(dict(seed=seed,old_command=old,old_classification=category,old_tau=decay,
            old_complete=s["complete"],old_diagnostics=s["diagnostics"]))
    (ROOT/"old_cases.json").write_text(json.dumps(rows,indent=2))
    files={}
    for folder in ["applications","corsika","src","tests","cmake","modules","validation"]:
        for p in (ROOT/"source"/folder).rglob("*"):
            if p.is_file() and p.suffix in [".hpp",".cpp",".h",".c",".cmake",".py",".sh",".txt"]:
                files[str(p.relative_to(ROOT/"source"))]=digest(p)
    (ROOT/"source_before.json").write_text(json.dumps(files,indent=2))
    (ROOT/"provenance.json").write_text(json.dumps(dict(host=socket.gethostname(),
        mesh_sha256=digest(mesh),observers=scene["radio"]["observers"],
        sample_rate_Hz=256e6,window_ns=[-1000.,511000.],seeds=[158,946,3605],
        natural_interactions=True,natural_tau_decay=True,
        inherited_cuts=dict(em_GeV=.1,hadron_GeV=10.,mu_tau_GeV=.3),
        thinning=.01,max_weight=50000,transport_window_ns=100000,
        radio_scope="paired CoREAS/ZHS, straight GO legs, direct or single transmission",
        scope_limit="Diagnostic high-energy reruns; coarse inherited cuts/thinning are not radio convergence validation"),indent=2))
    print(json.dumps(dict(prepared=True,old_cases=[(r["seed"],r["old_classification"]) for r in rows])))
if __name__=="__main__": main()

