"""Create the mAP evidence archive and diagnostic figure on the GPU notebook runtime."""

import argparse
from pathlib import Path
import platform
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import SKETCHY_SPLITS
from .map_transfer_audit import verify_decision
from .shared_concept_likelihood import FAMILY,PRIMARY


def make_figure(folder,audit):
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.2),layout="constrained")
    splits=(*SKETCHY_SPLITS,"tuberlin")
    panels=[read(folder/"seed-ablation"/f"{s}.json")["scores"] for s in splits]
    x=[0,1,2]
    colors=["#43617d","#c16b30","#0d776e"]
    for name,term,color in zip(("rgb","bare","domain"),("Cosine ảnh–ảnh","Từ nguyên văn","Từ có mô tả miền"),colors,strict=True):
        offsets={"rgb":-.22,"bare":0,"domain":.22}
        pos=[j+offsets[name] for j in x]
        before=[v[name+"-initial"]["mAP"]*100 for v in panels]
        after=[v[name+"-smoothed"]["mAP"]*100 for v in panels]
        axes[0].vlines(pos,before,after,color=color,linewidth=2.6)
        axes[0].scatter(pos,before,facecolors="white",edgecolors=color,s=33,zorder=3)
        axes[0].scatter(pos,after,color=color,s=39,zorder=3,label=term)
    axes[0].set(title="Ảnh hưởng của cùng một đồ thị RGB",ylabel="mAP (%)",xticks=x,xticklabels=("Ext-1","Ext-2","TU-Berlin"))
    axes[0].legend(frameon=False,fontsize=9,loc="lower right")
    axes[0].set_xlabel("Vòng rỗng trước lan truyền, vòng đặc sau lan truyền")
    labels=("1–10","11–100","101–200","201–1.000",">1.000")
    for s,name,color in zip(splits,("Ext-1","Ext-2","TU-Berlin"),colors,strict=True):
        bins=audit["splits"][s]["diagnosis"]["rank_bins"]
        delta=[100*(a["contribution_to_official_AP"]-b["contribution_to_official_AP"])
               for a,b in zip(bins["primary"]["bins"],bins["uniform"]["bins"],strict=True)]
        axes[1].plot(labels,delta,marker="o",label=name,color=color,lw=2)
    axes[1].axhline(0,color="#686868",lw=.85)
    axes[1].set(title="Mức tăng AP theo vị trí ảnh đúng",ylabel="Chênh lệch đóng góp vào AP (điểm %)",
                xlabel="Dải hạng. Ext-2 chỉ tính đến hạng 200")
    axes[1].legend(frameon=False,fontsize=9)
    for axis in axes:
        axis.grid(axis="y",color="#d9dcdf",lw=.6)
        axis.spines[["top","right"]].set_visible(False)
    fig.savefig(folder/"map-mechanism.png",dpi=175)
    plt.close(fig)


def pack(w):
    root=w/"results"
    folder=root/"map-study"
    decision,locks=verify_decision(root)
    audit=read(folder/"transfer-audit.json")
    if (decision!=audit["joint"] or read(folder/"seed-ablation/complete.json")["joint_sha256"]!=decision["sha256"]):
        raise ValueError("Final controls are missing or use a different lock")
    env={"export_revision":revision(),"platform":platform.platform(),"torch":torch.__version__,
         "gpu":torch.cuda.get_device_name(),"gpu_memory_gib":torch.cuda.get_device_properties(0).total_memory/(1024**3),
         "checkpoint":CHECKPOINT,"KD_source_sha256":checksum(w/"inquiry_kd_reference.py"),
         "protocol":"two Sketchy splits, all 30 TU classes under the locked settings","primary":decision["primary"]}
    atomic_json(folder/"environment.json",env)
    for name,lock in locks.items():
        summaries={s:{"primary":v["metrics"][name],"controls":v["controls"][name],
                      "replay":read(root/name/("heldout" if s=="tuberlin" else "sketchy")/f"{s}-{lock['winner_id']}.json")["kd_check"]}
                   for s,v in audit["splits"].items()}
        atomic_json(root/name/"summary.json",{"family":name,"joint":decision["sha256"],"freeze":lock["sha256"],
                    "revision":lock["revision"],"selected_parameters":lock["winner"],"splits":summaries})
    make_figure(folder,audit)
    files=[p for subdir in [folder,*[root/k for k in locks]] for p in subdir.rglob("*") if p.is_file()]
    # Include the exact reference scores used for paired baseline comparisons.
    for s in (*SKETCHY_SPLITS,"tuberlin"):
        sub="heldout" if s=="tuberlin" else "sketchy"
        for family,method in (("conditional-semantic-index","native"),(FAMILY,PRIMARY),(FAMILY,"uniform")):
            path=root/family/sub
            files.extend((path/f"{s}-{method}.json",path/"per-query"/f"{s}-{method}.pt"))
        rev=root/"reverse-prompt-diffusion"/sub
        ref=audit["splits"][s]["metrics"]["old_diffusion"]["sha256"]
        chosen=next(v for v in rev.glob(s+"-*.json") if checksum(v)==ref)
        files.extend((chosen,rev/"per-query"/(chosen.stem+".pt")))
    target=w/"map-priority-evidence.zip"
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(set(files)):
            archive.write(p,p.relative_to(w))
    stamp={"size":target.stat().st_size,"sha":checksum(target),"count":len(set(files)),"joint":decision["sha256"]}
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None or len(archive.namelist())!=stamp["count"]:
            raise ValueError("Corrupt export")
    atomic_json(w/"map-priority-archive-info.json",stamp)
    print(stamp,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    pack(p.parse_args().workspace)
