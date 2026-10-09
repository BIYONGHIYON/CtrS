"""Build compact curves and B0 comparisons from verified, measured artifacts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(root, baseline):
    root=Path(root); baseline=Path(baseline)
    results=json.loads((root/"results.json").read_text())
    old=json.loads((baseline/"results.json").read_text())
    verification=json.loads((root/"verification.json").read_text())
    if verification["status"]!="verified" or verification["baseline"]["status"]!="posthoc_artifact_conditions_verified":
        raise ValueError("Run verify_wavelet_archive.py first")
    b0=old["github_QB_baseline_k6_s42"]; previous=old["pasted_QB_band_gated_hf_k6_s42"]
    comparison={"comparison_scope":"posthoc matched nominal recipe; separate Kaggle sessions",
        "original_runtime_baseline_verified":False,"posthoc_verification":"verification.json",
        "optimizer_update_limitation":verification["realized_optimizer_updates"],
        "B0":b0,"previous_band_gated_hf":previous,"A0_A1":{},"deltas":{}}
    for name,result in results.items():
        comparison["A0_A1"][name]={"best_epoch":result["best_epoch"],"metrics":result["metrics"]}
        comparison["deltas"][name]={}
        for ref_name,ref in [("B0",b0),("previous_band_gated_hf",previous)]:
            comparison["deltas"][name][ref_name]={s:{k:result["metrics"][s]["mean"][k]-v
                for k,v in ref["metrics"][s]["mean"].items() if k in result["metrics"][s]["mean"] and k!="inference_ms"}
                for s in ("RR","FR")}
    (root/"posthoc_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n",encoding="utf-8")
    for name in results:
        run=root/"runs"/name; history=json.loads((run/"history.json").read_text())
        with (run/"curves.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
        gates=[json.loads((run/f"gate_epoch_{e:03}.json").read_text())["values"] for e in range(1,101)]
        gate=np.asarray(gates)
        with (run/"gate_compact.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f); writer.writerow(["epoch","stage","gate_mean","gate_std_mean","fraction_low","fraction_high","raw_residual_energy","injected_energy","MS_high_energy"])
            for e in range(100):
                for s in range(2): writer.writerow([e+1,s+1,*gate[e,s].mean((1,2)).tolist()])
    names=list(results); groups=[("B0",b0),*( (n.split('_')[0],results[n]) for n in names)]
    metrics=[("RR","PSNR"),("RR","SAM"),("RR","MSE_peak1"),("RR","ERGAS"),("RR","Q2n"),("FR","QNR")]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,(s,k) in zip(axes.flat,metrics):
        values=[r["metrics"][s]["mean"][k] for label,r in groups]
        ax.bar([label for label,r in groups],values,color=["#666666","#2788b5","#df852b"])
        ax.set_title(f"{s} {k}"+(" (provisional)" if s=="FR" else "")); ax.grid(axis="y",alpha=.2)
        for i,val in enumerate(values): ax.text(i,val,f"{val:.6g}",ha="center",va="bottom",fontsize=9)
    fig.suptitle("B0 / A0 / A1 | matched recipe, separate sessions; architecture/input/capacity differ",fontsize=11)
    fig.tight_layout(); fig.savefig(root/"B0_test_comparison_s42.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,(s,k) in zip(axes.flat,metrics):
        values=[comparison["deltas"][n]["B0"][s][k] for n in names]
        ax.bar([n.split('_')[0] for n in names],values,color=["#2788b5","#df852b"])
        ax.axhline(0,color="black",linewidth=.8); ax.set_title(f"{s} {k}: model minus B0"+(" (provisional)" if s=="FR" else "")); ax.grid(axis="y",alpha=.2)
        for i,val in enumerate(values): ax.text(i,val,f"{val:+.6g}",ha="center",va="bottom" if val>=0 else "top",fontsize=9)
    fig.tight_layout(); fig.savefig(root/"B0_test_delta_s42.png",dpi=150); plt.close(fig)
    a1=np.asarray([json.loads((root/"runs/A1_QB_s42"/f"gate_epoch_{e:03}.json").read_text())["values"] for e in range(1,101)])
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for stage in range(2):
        axes[0].plot(range(1,101),a1[:,stage,0].mean((1,2)),label=f"stage {stage+1}")
        axes[1].plot(range(1,101),a1[:,stage,2].mean((1,2)),label=f"stage {stage+1}: <0.05")
        axes[1].plot(range(1,101),a1[:,stage,3].mean((1,2)),ls="--",label=f"stage {stage+1}: >0.95")
        axes[2].plot(range(1,101),a1[:,stage,5].mean((1,2)),label=f"stage {stage+1}")
    for ax,title in zip(axes,["A1 mean gate","A1 saturation fractions","A1 injected residual energy"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Fixed first validation batch; diagnostics only")
    fig.tight_layout(); fig.savefig(root/"gate_diagnostics_s42.png",dpi=150); plt.close(fig)
    print("Compact curves, posthoc comparison, baseline and gate plots saved")


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("root",type=Path); p.add_argument("--baseline",type=Path,required=True)
    args=p.parse_args(); main(args.root,args.baseline)
