"""Plot paired class-bootstrap effects and sensitivity of the frozen pairwise method."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .concept_benchmark import read


def plot(workspace):
    folder=workspace/"results/strict-public-audit"
    result=read(folder/"report.json")
    fig,(effects,sensitivity)=plt.subplots(1,2,figsize=(12.3,4.15),layout="constrained")
    colors=("#1b6f6a","#a95a32","#345b90")
    markers={"full":("o","Đủ từ và prompt miền"),"bare_prompts":("D","Từ nguyên văn"),
             "half_0":("<","Nửa từ điển A"),"half_1":(">","Nửa từ điển B")}
    for i,(split,label) in enumerate((("sketchy_1","Sketchy Ext-1"),("sketchy_2","Sketchy Ext-2"),("tuberlin","TU-Berlin"))):
        info=result["splits"][split]
        y=2-i
        d=info["full"][result["primary"]]["method_minus_uniform"]["mAP"]
        lo,hi=d["ci95_class"]
        effects.errorbar(d["delta_pp"],y,xerr=[[d["delta_pp"]-lo],[hi-d["delta_pp"]]],color=colors[i],
                         fmt="o",capsize=4,markersize=7,linewidth=2)
        effects.annotate(f'{d["delta_pp"]:+.2f}'.replace(".",","),(d["delta_pp"],y),
                         xytext=(0,10),textcoords="offset points",ha="center",color=colors[i])
        for key,(marker,text_label) in markers.items():
            value=100*info["dictionary_sample"]["conditions"][key]["mAP"]
            sensitivity.scatter(value,y+(0.09 if key=="half_0" else -0.09 if key=="half_1" else 0),
                                marker=marker,s=59,color=colors[i] if key=="full" else "white",
                                edgecolors=colors[i],linewidths=1.5,zorder=2)
    for ax in (effects,sensitivity):
        ax.set_yticks([2,1,0],["Sketchy Ext-1","Sketchy Ext-2","TU-Berlin"])
        ax.set_ylim(-.45,2.55)
        ax.spines[["top","left","right"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
        ax.grid(axis="x",alpha=.22)
        ax.set_axisbelow(True)
    effects.axvline(0,color="0.42",linewidth=1)
    effects.set_title("Dịch qua từ gần so với khớp nguyên từ")
    effects.set_xlabel("Chênh lệch mAP (điểm phần trăm), khoảng 95% theo lớp")
    sensitivity.set_title("Độ nhạy với nguồn từ và prompt")
    sensitivity.set_xlabel("mAP trên mẫu 24 sketch/lớp và toàn bộ gallery (%)")
    for key,(marker,label) in markers.items():
        sensitivity.scatter([],[],marker=marker,s=45,color="#3e4752" if key=="full" else "white",
                            edgecolors="#3e4752",label=label)
    sensitivity.legend(loc="upper center",bbox_to_anchor=(.5,-.23),ncol=2,frameon=False)
    fig.savefig(folder/"pairwise-transfer.png",dpi=160,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    plot(p.parse_args().workspace)
