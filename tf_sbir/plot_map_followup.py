"""Plot paired class-bootstrap effects and exact half-reference runs from Colab."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .concept_benchmark import read


def plot(root):
    report=read(root/"map-followup/comparison.json")
    checks=read(root/"map-portability/portability.json")
    if report["lock_sha256"]!=checks["joint_sha256"]:
        raise ValueError("Plots require the same locked method")
    primary=report["primary"]
    splits=("sketchy_1","sketchy_2","tuberlin")
    titles=("Sketchy Ext-1","Sketchy Ext-2","TU-Berlin")
    colors=("#335877","#af5b2c","#077768")
    fig,(effect,sensitivity)=plt.subplots(1,2,figsize=(13,4.6),layout="constrained")
    for i,(split,label,color) in enumerate(zip(splits,titles,colors,strict=True)):
        item=report["splits"][split]
        controls=item["controls"][primary]
        measures=[controls["no_neighborhood"]["gain_of_primary"]["mAP"],
                  controls["uniform_prior"]["gain_of_primary"]["mAP"],
                  item["paired"][primary]["versus_incumbent"]["mAP"]]
        values=np.array([m["delta_pp"] for m in measures])
        low,high=np.array([m["ci95_class"] for m in measures]).T
        effect.errorbar(values,np.array([2.,1.,0.])+(i-1)*.19,xerr=np.stack((values-low,high-values)),
                        marker="o",linestyle="none",color=color,capsize=3,label=label,ms=5)
        variant=checks["splits"][split]["experiments"]
        baseline=100*item["scores"][primary]["mAP"]
        y=2-i
        sensitivity.scatter(baseline,y,marker="D",s=44,color="#232b34",zorder=4)
        for key,off,tint in (("photos",.16,"#087769"),("words",-.16,"#af5b2c")):
            points=[100*variant[key+"_"+half]["mAP"] for half in ("a","b")]
            sensitivity.plot(points,[y+off,y+off],color=tint,lw=2.6,marker="o",ms=5)
    effect.axvline(0,color="#666",lw=1.)
    effect.set(yticks=[2,1,0],yticklabels=["So với chỉ hiệu chỉnh từ","So với prior đều","So với đồ thị RGB cũ"],
               xlabel="Chênh lệch mAP (điểm phần trăm)",title="Thành phần nào giúp xếp hạng?")
    effect.legend(loc="upper center",bbox_to_anchor=(.5,-.16),ncol=3,frameon=False,fontsize=9)
    sensitivity.set(yticks=[2,1,0],yticklabels=titles,xlabel="mAP (%)",title="Độ nhạy với tập tham chiếu")
    handles=[Line2D([],[],marker="D",linestyle="none",color="#232b34",label="Cấu hình đầy đủ"),
             Line2D([],[],marker="o",color="#087769",label="Hai nửa gallery"),
             Line2D([],[],marker="o",color="#af5b2c",label="Hai nửa từ điển")]
    sensitivity.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,-.16),ncol=3,frameon=False,fontsize=9)
    for axis in (effect,sensitivity):
        axis.grid(axis="x",color="#d7dbe0",lw=.7)
        axis.set_axisbelow(True)
        axis.spines[["right","top","left"]].set_visible(False)
        axis.tick_params(axis="y",length=0,pad=7)
        axis.margins(y=.19)
    fig.savefig(root/"map-followup/map-transfer-mechanism.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("/content/sbir/results"))
    plot(p.parse_args().root)
