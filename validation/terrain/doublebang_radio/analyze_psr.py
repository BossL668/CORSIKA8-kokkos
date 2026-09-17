#!/usr/bin/env python3
"""Summarize completed PSR runs and plot their actual paired radio fields."""
import argparse, csv, json, pathlib, sys
from collections import Counter
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=pathlib.Path("/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912")
def profile(folder):
    s=yaml.safe_load((folder/"output/terrain_run.yaml").read_text());r=s["radio_result"]
    grids=["interface_radio_moments","interface_radio_coreas_regularized_moments","interface_radio_zhs_moments"]
    tracks=["interface_radio_transport_tracks","interface_radio_device_pending","interface_radio_cpu_pending"]
    patterns=dict(transport="InterfaceStepKernel",CoREAS="CoreasEndpointKernel",ZHS="AccumulateKernel")
    kernel_counts={k:0 for k in patterns};kernel_ns={k:0 for k in patterns}
    final=[];last=0;downloads=[];cpu_upload_bytes=0
    no_track_roundtrip=True;no_grid_reupload=True;queue_resident=True;queue_append=False
    # High-energy traces can be large; retain only the final grid transfers.
    with (folder/"cupti.jsonl").open() as stream:
        for line in stream:
            e=json.loads(line)
            if e["event"]=="finalize":final.append(e)
            if e["event"]=="gpu_kernel":
                for k,pattern in patterns.items():
                    if pattern in e["name"]:
                        kernel_counts[k]+=1;kernel_ns[k]+=e["end"]-e["start"]
                        if k!="transport":last=max(last,e["end"])
            if e["event"]!="copy_begin" or not e["bytes"] or e["src_label"]=="(none)":continue
            if e["src_label"] in grids and e["dst_space"]=="Host":downloads.append(e)
            if e["dst_label"]=="interface_radio_cpu_tracks" and e["src_space"]=="Host":cpu_upload_bytes+=e["bytes"]
            no_track_roundtrip &= not (e["src_space"]!=e["dst_space"] and (e["src_label"] in tracks or e["dst_label"] in tracks))
            no_grid_reupload &= not (e["dst_label"] in grids and e["src_space"]=="Host")
            queue_resident &= not (e["src_label"] in ["interface_resident_particles","interface_resident_successors"] and e["dst_space"]=="Host")
            queue_append |= e["src_space"]==e["dst_space"]=="Cuda" and e["dst_label"]=="interface_resident_particles"
    c=dict(trace_complete=len(final)==1 and not final[0]["failed"] and final[0]["dropped_records"]==0,
        actual_transport_kernel=kernel_counts["transport"]>0,
        both_radio_kernels=all(kernel_counts[k]==r["wavefronts"]>0 for k in ["CoREAS","ZHS"]),
        three_final_grids=len(downloads)==3 and {e["src_label"] for e in downloads}==set(grids) and all(e["time"]>last for e in downloads),
        no_radio_track_roundtrip=no_track_roundtrip,
        no_grid_reupload=no_grid_reupload,
        transport_queue_stays_on_device=queue_resident,
        queue_append_on_device=queue_append)
    row=dict(checks=c,gpu_kernel_counts=kernel_counts,
        gpu_kernel_ms={k:v/1.e6 for k,v in kernel_ns.items()},
        radio_grid_download_bytes=sum(e["bytes"] for e in downloads),
        cpu_source_upload_bytes=cpu_upload_bytes,
        resident=s["accelerator"],radio=r)
    (folder/"residency.json").write_text(json.dumps(row,indent=2))
    if not all(c.values()): raise RuntimeError("Residency check failed "+str(c))
    return row
def main():
    p=argparse.ArgumentParser();p.add_argument("--controls-only",action="store_true");a=p.parse_args()
    if not ROOT.is_dir(): raise RuntimeError("PSR only")
    profiles={}
    for folder in sorted((ROOT/"runs").glob("*cuda_*")):
        if (folder/"checks.json").exists() and (folder/"cupti.jsonl").exists(): profiles[folder.name]=profile(folder)
    (ROOT/"residency_summary.json").write_text(json.dumps(profiles,indent=2))
    if a.controls_only: return
    out=ROOT/"report";out.mkdir(exist_ok=True)
    figures=out/"figures";figures.mkdir(exist_ok=True)
    rows=[]
    for folder in sorted((ROOT/"runs").glob("*_seed*")):
        if not (folder/"checks.json").exists(): continue
        checked=json.loads((folder/"checks.json").read_text());assert all(checked["checks"].values())
        s=yaml.safe_load((folder/"output/terrain_run.yaml").read_text());config=json.loads((folder/"output/radio/CoREAS/config.json").read_text())
        n=config["samples"];rate=config["sample_rate_Hz"];freq=np.fft.rfftfreq(n,1/rate)
        band=(freq>=50e6)&(freq<=100e6)
        waves={}
        for alg in ["CoREAS","ZHS"]:
            data=np.genfromtxt(str(folder/"output/radio"/alg/"field.csv"),delimiter=",",names=True)
            values=np.stack([data[k+"_V_m"] for k in ["Ex","Ey","Ez"]],axis=-1).reshape(-1,n,3).transpose(0,2,1)
            waves[alg]=np.fft.irfft(np.fft.rfft(values,axis=-1)*band,n=n,axis=-1)
        t=(np.arange(n)/rate+config["start_time_s"])*1.e6
        fig,axes=plt.subplots(2,len(config["observers"]),figsize=(5.2*len(config["observers"]),6.4),squeeze=False,constrained_layout=True)
        peaks=[]
        for i,o in enumerate(config["observers"]):
            w=waves["ZHS"][i];peak_index=int(np.argmax(np.linalg.norm(w,axis=0)));axis=int(np.argmax(np.abs(w[:,peak_index])))
            magnitude=float(np.max(np.linalg.norm(w,axis=0)));peaks.append(dict(observer=o["name"],peak_V_m=magnitude,peak_time_us=float(t[peak_index])))
            for name,style in [("ZHS","-"),("CoREAS","--")]:
                axes[0,i].plot(t,waves[name][i,axis]*1e6,style,lw=1,label=name)
                sl=slice(max(0,peak_index-100),min(n,peak_index+101))
                axes[1,i].plot(t[sl],waves[name][i,axis,sl]*1e6,style,lw=1.5,label=name)
            axes[0,i].set(xlabel="Arrival time (us)",ylabel=["Ex","Ey","Ez"][axis]+" (uV/m)",title=o["name"]+": full reception window")
            axes[1,i].set(xlabel="Arrival time (us)",ylabel=["Ex","Ey","Ez"][axis]+" (uV/m)",title=o["name"]+": strongest pulse")
            for ax in axes[:,i]: ax.grid(alpha=.2);ax.legend()
        fig.suptitle(folder.name+" | 100 PeV nu_tau | 50-100 MHz ideal bandpass")
        fig.savefig(figures/(folder.name+"_waveforms.png"),dpi=170);fig.savefig(figures/(folder.name+"_waveforms.pdf"));plt.close(fig)
        decays=[]
        for decay in s["tau"]["decays"]:
            pdgs=[abs(x["pdg"]) for x in decay["daughters"]]
            category="muonic" if 13 in pdgs else "electronic" if 11 in pdgs else "hadronic"
            decays.append(dict(channel=category,medium=decay["medium"],energy_PeV=decay["energy_GeV"]/1e6,
                time_us=decay["time_ns"]/1000.,position_m=decay["position_enu_m"],
                relative_energy_residual=decay["relative_energy_residual"]))
        rows.append(dict(run=folder.name,seed=s["seed"],decays=decays,peaks=peaks,
            coreas_zhs_relative_l2=checked["coreas_zhs_relative_l2"],checks=checked["checks"],
            diagnostics=s["diagnostics"],accelerator=s["accelerator"],radio=s["radio_result"],energy_ledger=s["energy_ledger"]))
    (out/"results.json").write_text(json.dumps(rows,indent=2))
    md=["# 100 PeV 穿山 ντ：当前代码审计与射电复跑","",
        "图片中的两个算法使用同一批真实输运轨迹。上排是完整接收窗，下排放大最强脉冲；画出该脉冲最强的笛卡尔分量，采用 50–100 MHz 理想带通。不同站点可能选择不同分量，幅度是电场，不是天线电压。各站纵轴量程独立，比较强弱需读刻度及科学计数法倍率。",
        "","**不能把两条算法曲线重合解读为完整物理已验证，也不能把两个产生顶点直接称为可分辨的双脉冲。** 原 cut/thinning 较粗，本批首先用于原配置复跑和接线审计。",""]
    for row in rows:
        md.extend(["## "+row["run"],"", "![](figures/"+row["run"]+"_waveforms.png)","",
            "τ 末态："+", ".join(x["channel"]+" / "+x["medium"] for x in row["decays"])+"。CoREAS/ZHS 全频复数谱相对 L2 差异：%.3g。"%row["coreas_zhs_relative_l2"],
            "","看下排实线与虚线是否重合；看上排是否有时间上分离的信号。零信号或被遮挡站点也保留，不据此制造第二个峰。",""])
    (out/"RESULTS_CN.md").write_text("\n".join(md)+"\n")
    print(json.dumps(dict(completed_runs=len(rows),profiles=len(profiles))))
if __name__=="__main__":main()
