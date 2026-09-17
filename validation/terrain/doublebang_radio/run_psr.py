#!/usr/bin/env python3
"""PSR-only, owned subprocesses, explicit completion gates; never overwrite a run."""
import argparse, copy, csv, hashlib, json, math, os, pathlib, signal, subprocess, time
import numpy as np
import yaml
ROOT=pathlib.Path("/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912")
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def setarg(cmd,k,v):
    cmd[cmd.index(k)+1]=str(v)
def gpu():
    v=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip().split(",")
    return [float(x) for x in v]
def run(tag,cmd,backend,timeout,reuse_complete=False):
    folder=ROOT/"runs"/tag
    output=folder/"output";setarg(cmd,"--output",output)
    binary=pathlib.Path(cmd[3]);binary_hash=digest(binary)
    scene_hash=digest(pathlib.Path(cmd[cmd.index("--scene")+1]))
    if reuse_complete and folder.exists():
        previous=json.loads((folder/"status.json").read_text())
        if not (previous["returncode"]==0 and previous["complete"] and previous["binary_sha256"]==binary_hash and previous.get("scene_sha256")==scene_hash and previous["command"]==cmd):
            raise RuntimeError("Existing control does not match current inputs: "+tag)
        print("Reusing completed current control "+tag,flush=True)
        return folder,yaml.safe_load((output/"terrain_run.yaml").read_text())
    folder.mkdir(parents=True,exist_ok=False)
    work=folder/"work";work.mkdir()
    (folder/"command.json").write_text(json.dumps(cmd,indent=2))
    env=dict(os.environ,OMP_NUM_THREADS="256",OMP_PROC_BIND="spread",OMP_PLACES="threads",OPENBLAS_NUM_THREADS="1")
    if backend=="cuda" and (tag.endswith("_on") or tag=="cuda_seed946"):
        plugin=pathlib.Path("/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-interface-resident-20260911/cuda-residency-profile/libInterfaceResidencyTrace.so")
        env["KOKKOS_TOOLS_LIBS"]=str(plugin)
        env["C8_INTERFACE_TRACE_FILE"]=str(folder/"cupti.jsonl")
        (folder/"profiler.json").write_text(json.dumps(dict(path=str(plugin),sha256=digest(plugin)),indent=2))
    start=time.monotonic();reason=None;peak=0
    with (folder/"run.log").open("x") as log, (folder/"resources.jsonl").open("x") as resource:
        child=subprocess.Popen(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (folder/"pid").write_text(str(child.pid))
        try:
            while child.poll() is None:
                mem=dict(line.split(":",1) for line in pathlib.Path("/proc/meminfo").read_text().splitlines())
                try:
                    status=pathlib.Path("/proc/%d/status"%child.pid).read_text()
                    rss=next((int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:")),0)
                except FileNotFoundError: rss=0
                peak=max(peak,rss)
                sample=dict(elapsed_s=time.monotonic()-start,rss_kib=rss,available_kib=int(mem["MemAvailable"].split()[0]))
                if backend=="cuda": sample["gpu"]=gpu()
                resource.write(json.dumps(sample)+"\n");resource.flush()
                if sample["elapsed_s"]>timeout: reason="owned run timeout"
                if rss>32*1024*1024: reason="owned run RSS exceeds 32 GiB"
                if sample["available_kib"]<16*1024*1024: reason="available host RAM below 16 GiB"
                if backend=="cuda" and sample["gpu"][0]>3800: reason="GPU memory exceeds 3800 MiB"
                if reason: break
                time.sleep(2.)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(10)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
    summary=yaml.safe_load((output/"terrain_run.yaml").read_text()) if (output/"terrain_run.yaml").exists() else {}
    result=dict(tag=tag,backend=backend,command=cmd,returncode=child.returncode,stop_reason=reason,
        wall_s=time.monotonic()-start,peak_rss_kib=peak,binary_sha256=binary_hash,
        scene_sha256=scene_hash,
        binary_unchanged=digest(binary)==binary_hash,complete=summary.get("complete",False),error=summary.get("error"))
    (folder/"status.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if child.returncode or reason or not result["complete"] or not result["binary_unchanged"]:
        raise RuntimeError("Incomplete run: "+tag)
    return folder,summary
def checks(folder,s,radio):
    a=s["accelerator"];d=s["diagnostics"];r=s.get("radio_result",{})
    c=dict(complete=s.get("complete") is True,pending_empty=a.get("pending_particles")==0,
        medium_valid=d["material_mismatches"]==0,all_tracks=not d["csv_truncated"],
        all_deposits=not d["deposition_csv_truncated"],
        energy_ledger=abs(s["energy_ledger"]["unexplained_over_initial"])<1.e-8,
        natural_tau=not s["tau"]["force_decay_called"])
    vertices=s["neutrino"].get("interactions",[])
    c["neutrino_vertex_conservation"]=all(
        v["vertex_audit"]["conservation"]["corsika_max_component_relative_residual"]<1.e-8 and
        v["vertex_audit"]["initial_charge_e"]==v["vertex_audit"]["final_charge_e"] for v in vertices)
    c["tau_decay_conservation"]=all(
        abs(t["relative_energy_residual"])<2.e-6 and t["relative_momentum_residual"]<2.e-6 and
        abs(sum(x["energy_GeV"] for x in t["daughters"])/t["energy_GeV"]-1)<2.e-6
        for t in s["tau"].get("decays",[]))
    if radio:
        c.update(radio_complete=r.get("complete") is True,both_algorithms=r.get("algorithms")==["CoREAS","ZHS"],
            radio_errors=r.get("errors")==0 and r.get("out_of_window")==0,
            single_finalization=r.get("downloads")==1,three_grids=r.get("moment_arrays")==3)
        c["paired_source_precision"]=r["CoREAS"].get("roundoff_limited_tracks",0)==r["ZHS"].get("roundoff_limited_tracks",0)
        charged=0
        # Charge inventory from transported CSV, including charged nuclei.
        charged_pdgs={11,13,15,211,321,2212,3222,3112,3312,3334}
        count=0;invalid=0;acausal=0;tau_tracks=[]
        with (folder/"output/terrain/tracks.csv").open() as stream:
            for t in csv.DictReader(stream):
                count+=1;pid=abs(int(t["pdg"]));w=float(t["weight"]);t0=float(t["t0_s"]);t1=float(t["t1_s"])
                vals=[float(t[k]) for k in ["x0_m","y0_m","z0_m","x1_m","y1_m","z1_m","E0_GeV","E1_GeV"]]
                invalid+=not(all(math.isfinite(v) for v in vals+[w,t0,t1]) and w>0 and t1>=t0 and vals[-1]>=0)
                distance=math.sqrt(sum((vals[k+3]-vals[k])**2 for k in range(3)))
                # Chord cannot exceed c*dt; allow 1 micrometre global-coordinate roundoff.
                acausal+=distance>299792458.*(t1-t0)*(1.+1.e-8)+1.e-6
                if pid==15: tau_tracks.append(t)
                moving=any(float(t[k+"0_m"])!=float(t[k+"1_m"]) for k in ["x","y","z"])
                nucleus_charge=pid>=1000000000 and (pid//10000)%1000>0
                if (pid in charged_pdgs or nucleus_charge) and moving and t1>t0 and w>0: charged+=1
        c["source_inventory"]=all(r[k]["device_tracks"]+r[k]["cpu_tracks"]==charged for k in ["CoREAS","ZHS"])
        c["csv_count"]=count==d["steps"];c["finite_transport"]=invalid==0
        c["causal_transport"]=acausal==0
        (folder/"tau_tracks.json").write_text(json.dumps(tau_tracks,indent=2))
        fft=[]
        for alg in ["CoREAS","ZHS"]:
            p=folder/"output/radio"/alg/"spectrum.csv"
            arr=np.genfromtxt(str(p),delimiter=",",names=True)
            fft.append(np.stack([arr[k+"_real"]+1j*arr[k+"_imag"] for k in ["Ex","Ey","Ez"]]))
        rel=float(np.linalg.norm(fft[0]-fft[1])/max(np.linalg.norm(fft[1]),1.e-100))
        c["paired_equivalence"]=math.isfinite(rel) and rel<1.e-5
        c["finite_radio"]=all(bool(np.isfinite(f).all()) for f in fft)
        scheck=dict(checks=c,charged_tracks=charged,coreas_zhs_relative_l2=rel,radio=r)
    else: scheck=dict(checks=c)
    (folder/"checks.json").write_text(json.dumps(scheck,indent=2))
    if not all(c.values()): raise RuntimeError("Failed checks: "+str(c))
    return scheck
def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=["controls","production"],required=True)
    p.add_argument("--backend",choices=["openmp","cuda"],required=True)
    p.add_argument("--prefix",default="")
    p.add_argument("--transport-refined",action="store_true",
        help="Controls: record old-transport differences without requiring identity after the magnetic integration refinement")
    p.add_argument("--seeds",type=int,nargs="+",default=[158,946,3605]);a=p.parse_args()
    if not (ROOT/("TESTS_"+a.backend.upper()+"_PASSED")).exists(): raise RuntimeError("Current backend tests must pass first")
    old=json.loads((ROOT/"old_cases.json").read_text())
    def command(seed,scene):
        row=next(x for x in old if x["seed"]==seed)
        cmd=list(row["old_command"]);cmd[2]="0-255";cmd[3]=str(ROOT/("build-"+a.backend)/"applications/c8_terrain_cascade")
        setarg(cmd,"--scene",scene);setarg(cmd,"--threads",256);setarg(cmd,"--device-memory-MiB",1536)
        # Original batch/physics retained; scalar scheduling may differ from old nonresident implementation.
        cmd+=["--em-scheduler","resident","--resident-capacity","262144"]
        return cmd
    if a.phase=="controls":
        scene=yaml.safe_load((ROOT/"scene_radio.yaml").read_text());scene["radio"].update(samples=32768,memory_MiB=256)
        scene["radio"]["observers"]=[dict(name="diagnostic_overhead",position_enu_m=[0.,0.,1000.]),scene["radio"]["observers"][0]]
        files={}
        for enabled in [False,True]:
            scene["radio"]["enabled"]=enabled
            f=ROOT/("scene_control_"+str(enabled)+".yaml");f.write_text(yaml.safe_dump(scene,sort_keys=False));files[enabled]=f
        controls=[]
        for name,primary,energy,seed in [("electron","electron",1.,1701),("tau","tau_minus",1000.,1702)]:
            pair=[]
            for enabled in [False,True]:
                cmd=command(158,files[enabled]);setarg(cmd,"--primary",primary);setarg(cmd,"--energy-GeV",energy);setarg(cmd,"--seed",seed)
                setarg(cmd,"--transport-window-ns",1100);setarg(cmd,"--batch",64);setarg(cmd,"--max-weight",50)
                i=cmd.index("--position-m");cmd[i+1:i+4]=["0","0","-.01"]
                i=cmd.index("--direction");cmd[i+1:i+4]=["0","0","1"]
                folder,s=run(a.prefix+a.backend+"_"+name+("_on" if enabled else "_off"),cmd,a.backend,3600,reuse_complete=True)
                report=checks(folder,s,enabled)
                hashes={f.name:digest(f) for f in (folder/"output/terrain").glob("*.csv")}
                pair.append(dict(folder=str(folder),checks=report,hashes=hashes,tau=s["tau"],neutrino=s["neutrino"]))
            same=pair[0]["hashes"]==pair[1]["hashes"] and pair[0]["tau"]==pair[1]["tau"] and pair[0]["neutrino"]==pair[1]["neutrino"]
            controls.append(dict(case=name,radio_does_not_change_transport=same,pair=pair))
            if a.prefix:
                reference=ROOT/"runs"/(a.backend+"_"+name+"_on")
                previous=json.loads((reference/"checks.json").read_text())
                old_hashes={f.name:digest(f) for f in (reference/"output/terrain").glob("*.csv")}
                errors={}
                for algorithm in ["CoREAS","ZHS"]:
                    for filename in ["moments.bin"]+(["regularized_moments.bin"] if algorithm=="CoREAS" else []):
                        actual=np.fromfile(pathlib.Path(pair[1]["folder"])/"output/radio"/algorithm/filename,dtype=np.float64)
                        expected=np.fromfile(reference/"output/radio"/algorithm/filename,dtype=np.float64)
                        errors[algorithm+"/"+filename]=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1.e-100))
                unchanged=old_hashes==pair[1]["hashes"] and all(e<1.e-10 for e in errors.values())
                comparison=dict(unchanged=unchanged,moment_relative_l2=errors,
                    transport_csv_identical=old_hashes==pair[1]["hashes"],
                    transport_refinement_expected=a.transport_refined)
                key="pre_refinement_baseline_comparison" if a.transport_refined else "exhaustive_baseline_comparison"
                controls[-1][key]=comparison
                if not unchanged and not a.transport_refined: raise RuntimeError("BVH changed accepted full-shower transport/radio")
            (ROOT/("controls_"+a.backend+".json")).write_text(json.dumps(controls,indent=2))
            if not same: raise RuntimeError("Radio changed transport/RNG in "+name)
        (ROOT/("CONTROLS_"+a.backend.upper()+"_PASSED")).touch()
    else:
        if not (ROOT/("CONTROLS_"+a.backend.upper()+"_PASSED")).exists(): raise RuntimeError("Controls must pass first")
        for seed in a.seeds:
            folder,s=run(a.prefix+a.backend+"_seed"+str(seed),command(seed,ROOT/"scene_radio.yaml"),a.backend,24*3600)
            checks(folder,s,True)
if __name__=="__main__": main()
