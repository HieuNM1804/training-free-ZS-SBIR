"""Plot the six new retrieval methods and their TU transfer uncertainty."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from .concept_benchmark import read


NAMES={"retrieval-semantic-whitening":"Whitening qua từ công khai",
       "retrieval-reciprocal-encoding":"Láng giềng thuận nghịch",
       "retrieval-soft-assignment":"Gán mềm vào nhiều từ",
       "retrieval-democratic-regions":"Cân bằng vùng lặp",
       "retrieval-composed-modality":"Phối hợp ảnh và mô tả",
       "retrieval-vocabulary-tree":"Cây từ vựng"}


def draw(w):
    dest=w/"results/image-retrieval-round";report=read(dest/"report.json")
    splits=report["splits"]
    names=sorted(NAMES,key=lambda n:-sum(splits[s]["scores"][n]["mAP"] for s in ("sketchy_1","sketchy_2")))
    fig,(a,b)=plt.subplots(1,2,figsize=(12,5),layout="constrained",gridspec_kw={"width_ratios":[1.5,1]})
    colors=("#24527a","#be641f","#12796d");markers=("o","s","D")
    ypos=np.arange(len(names))[::-1]
    for j,(s,label) in enumerate(zip(("sketchy_1","sketchy_2","tuberlin"),("Sketchy Ext-1","Sketchy Ext-2","TU-Berlin"))):
        x=[100*splits[s]["scores"][n]["mAP"] for n in names]
        a.scatter(x,ypos+(j-1)*.17,s=48,color=colors[j],marker=markers[j],label=label,zorder=3)
    for y,n in zip(ypos,names):
        item=splits["tuberlin"]["scores"][n]["versus_reference_of_same_scope"]["mAP"]
        d=item["delta_pp"];lo,hi=item["ci95_class"]
        b.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt="o",color=colors[2],capsize=3,linewidth=1.8)
    a.set_yticks(ypos,[NAMES[n] for n in names]);b.set_yticks(ypos,[])
    a.set_title("Điểm của từng phương pháp");a.set_xlabel("mAP (%)")
    b.set_title("Chênh lệch trên TU-Berlin");b.set_xlabel("Điểm mAP so với mốc cùng điều kiện gallery\nKhoảng tin cậy 95% theo lớp")
    b.axvline(0,color=".45",linewidth=1)
    for ax in (a,b):
        ax.spines[["top","left","right"]].set_visible(False)
        ax.tick_params(axis="y",length=0);ax.grid(axis="x",alpha=.22);ax.set_axisbelow(True);ax.set_ylim(-.6,len(names)-.4)
    a.legend(loc="upper center",bbox_to_anchor=(.5,-.15),ncol=3,frameon=False)
    fig.savefig(dest/"retrieval-comparison.png",dpi=160,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    draw(p.parse_args().workspace)
