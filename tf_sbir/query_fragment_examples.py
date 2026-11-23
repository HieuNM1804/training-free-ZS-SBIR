"""Render deterministic, median-change retrieval examples from the saved post-ranking audit."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .concept_benchmark import read
from .crossdomain_inputs import runtime
from .query_fragment_reference_inputs import FAMILY


def render(workspace,dest):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"figure.facecolor":"white"})
    fig,axes=plt.subplots(6,3,figsize=(7.1,12.2))
    fig.subplots_adjust(left=.19,right=.985,bottom=.022,top=.952,hspace=.30,wspace=.09)
    for j,title in enumerate(("Sketch truy vấn","Top-1 với màu trung bình","Top-1 với sáu mảnh")):
        axes[0,j].set_title(title,fontsize=11,weight="bold",pad=7)
    row=0
    for split,label in (("sketchy_1","Sketchy Ext-1"),("sketchy_2","Sketchy Ext-2"),("tuberlin","TU-Berlin")):
        folder=Path(runtime(workspace,"tuberlin" if split=="tuberlin" else "sketchy",batch=1,workers=0).root)
        samples=read(workspace/"results"/FAMILY/("rank-audit-"+split+".json"))["pairs"]["mean_colour/walk"]["examples"]
        for name,outcome in (("repair","Sửa đúng"),("loss","Chuyển sai")):
            record=samples[name]
            for col,key in enumerate(("query","first_before","first_after")):
                ax=axes[row,col];path=Path(record[key])
                with Image.open(folder/path) as source:ax.imshow(source.convert("RGB"))
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                ax.set_xlabel(path.parent.name.replace("_"," "),labelpad=3)
            box=axes[row,0].get_position()
            fig.text(.016,(box.y0+box.y1)/2+.01,label,fontsize=10,weight="bold",va="center")
            fig.text(.016,(box.y0+box.y1)/2-.012,outcome,fontsize=9,va="center")
            row+=1
    dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=145,facecolor="white");plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args();render(a.workspace,a.output)
