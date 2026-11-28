"""Certify contradictory score-offset inequalities from actual erroneous retrieval decisions."""

import argparse
from collections import defaultdict, Counter

import networkx as nx
import torch
from torch.nn.functional import normalize

from .gallery_name_rank_decomposition import semantic_inputs
from .native_block_probe import deterministic
from .photo_name_evaluation import queries
from .photo_name_interaction import OUT as NAME_INDEX, PARTS, ordered_shards, record
from .description_choice_blocks import inputs
from .visual_category_relation import OUT
from .wordnet_renaming_pilot import SPLITS


def discrepancies(q,p,labels,classes,gi,source):
    y=classes.cuda()
    matches=[]
    for start in range(0,len(q),192):
        sl=slice(start,start+192)
        score=q[sl]@p.T
        same=y[None,:]==labels[sl,None].cuda()
        relevant=same & (gi.cuda()[None,:]!=source[sl,None].cuda())
        pos=score.masked_fill(~relevant,-torch.inf).argmax(1)
        neg=score.masked_fill(same,-torch.inf).argmax(1)
        row=torch.arange(len(pos),device=score.device)
        matches.append((pos.cpu(),neg.cpu(),(score[row,neg]-score[row,pos]).cpu()))
    return tuple(torch.cat([t[k] for t in matches]) for k in range(3))


def graph_certificates(src,dst,margin):
    n=len(src)
    groups=defaultdict(list)
    for j,(a,b) in enumerate(zip(src.tolist(),dst.tolist(),strict=True)):groups[a,b].append(j)
    graph=nx.DiGraph()
    for edge,rows in sorted(groups.items()):graph.add_edge(*edge,weight=len(rows))
    components=list(nx.strongly_connected_components(graph))
    membership={i:j for j,c in enumerate(components) for i in c}
    cyclic=sum(len(rows) for (a,b),rows in groups.items() if membership[a]==membership[b])

    # Any topological ordering of SCCs satisfies all the inter-component inequalities.
    condensed=nx.condensation(graph,components)
    order=[]
    for j in nx.lexicographical_topological_sort(condensed,key=lambda i:min(components[i])):
        component=graph.subgraph(components[j])
        order.extend(sorted(component, key=lambda i:(component.in_degree(i,weight="weight")-component.out_degree(i,weight="weight"),i)))
    rank={v:i for i,v in enumerate(order)}
    lo=torch.tensor([rank[x] for x in src.tolist()])
    hi=torch.tensor([rank[x] for x in dst.tolist()])
    stride=1.+float(margin.max())
    failures=((hi-lo).double()*stride<=margin.double()).nonzero().flatten()
    if not torch.equal(failures,(hi<=lo).nonzero().flatten()):raise ValueError("Invalid feasible potential")

    remaining=graph.copy()
    edge_rows={k:list(reversed(v)) for k,v in groups.items()}
    certificates=[]
    def consume(cycle):
        count=min(len(edge_rows[e]) for e in cycle)
        for _ in range(count):certificates.append([edge_rows[e].pop() for e in cycle])
        for e in cycle:
            if not edge_rows[e]:remaining.remove_edge(*e)
    for a,b in sorted(groups):
        if a==b:consume([(a,a)])
    for a,b in sorted(groups):
        if a<b and remaining.has_edge(a,b) and remaining.has_edge(b,a):
            consume([(a,b),(b,a)])
    for vertices in components:
        if len(vertices)<3:continue
        sub=remaining.subgraph(vertices)
        while True:
            try:cycle=nx.find_cycle(sub)
            except nx.NetworkXNoCycle:break
            consume(cycle)
    # Check the original row indices, not just the counts produced by the algorithm.
    seen=set()
    for ids in certificates:
        if any(i in seen for i in ids) or len(set(ids))!=len(ids):raise ValueError("Reused query edge")
        seen.update(ids)
        if not torch.equal(dst[ids],src[ids].roll(-1)) or (margin[ids]<=0).any():raise ValueError("Invalid strict cycle")
    if len(certificates)>len(failures) or len(failures)>cyclic or not nx.is_directed_acyclic_graph(remaining):
        raise ValueError("Unverified cycle packing or ordering")
    summary={"strict_errors":n,"vertices":len(order),"distinct_edges":len(groups),"cyclic_edges":cyclic,
             "largest_component":max(map(len,components)),"minimum_failures_lower":len(certificates),
             "minimum_failures_upper":len(failures),"lower_percent":100*len(certificates)/n,
             "upper_percent":100*len(failures)/n,"cycle_sizes":dict(Counter(map(len,certificates))),
             "self_loop_queries":int((src==dst).sum()),"stride":stride}
    return summary,dict(cycles=certificates,order=order,failed_edge_ids=failures)


@torch.inference_mode()
def run(split,revision,name_revision,native_revision):
    qi,labels,gi,classes,source,names,semantic_q,semantic_p,_=semantic_inputs(split,name_revision,native_revision)
    _,idx,_=inputs()
    raw=ordered_shards([NAME_INDEX/f"names-{i}.pt" for i in range(PARTS)],name_revision,torch.arange(len(idx["photo"]["paths"])))
    photo=normalize(raw["native"][gi].cuda(),dim=1)
    visual=normalize(queries(qi,native_revision),dim=1)
    reports,artifacts={},{}
    for reader,q,p in (("native",visual,photo),("semantic",semantic_q,semantic_p)):
        pos,neg,margin=discrepancies(q,p,labels,classes,gi,source)
        keep=margin>0
        if not keep.any() or not torch.equal(classes[pos],labels) or (classes[neg]==labels).any() or (gi[pos]==source).any():
            raise ValueError("Improper candidate set")
        reports[reader]={"strict_errors":int(keep.sum()),"ties":int((margin==0).sum()),"offset_families":{}}
        artifacts[reader]={"positive":pos,"negative":neg,"margin":margin,"edge_rows":keep.nonzero().flatten(),"families":{}}
        for kind,g in (("photo",gi),("noun",names.cpu()),("true_class",classes)):
            summary,proof=graph_certificates(g[neg[keep]],g[pos[keep]],margin[keep])
            reports[reader]["offset_families"][kind]=summary
            artifacts[reader]["families"][kind]=proof
    record(OUT/f"cycles-{split}.pt",{"query_ids":qi,"gallery_ids":gi,"query_classes":labels,"gallery_classes":classes,
           "sources":source,"gallery_nouns":names.cpu(),"readers":artifacts},revision,{"split":split,
           "queries":len(qi),"gallery":len(gi),"reports":reports,"names_revision":name_revision,
           "native_revision":native_revision})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--name-revision",required=True)
    p.add_argument("--native-revision",required=True)
    a=p.parse_args()
    deterministic()
    run(a.split,a.revision,a.name_revision,a.native_revision)
