"""Attribute the signed AP change to groups of previous decisions on the two exploratory splits."""

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from .concept_benchmark import atomic_json


def summarize(row, group, delta, gain):
    count = int(group.sum())
    if not count:
        return {"queries": 0}
    selected = delta[group]
    change = selected.mean(0) * 100
    return {"queries": count, "mean_delta_pp": change.tolist(),
            "contribution_to_all_query_mean_pp": (change*count/len(group)).tolist(),
            "native_expected_gain": float(gain[group].mean()),
            "AP_up": int((selected[:, 0] > 1e-6).sum()), "AP_down": int((selected[:, 0] < -1e-6).sum()),
            "mean_base_AP": float(row[group, 0].mean()),
            "median_delta_AP_pp": float(selected[:, 0].median()*100),
            "delta_AP_q10_q90_pp": (torch.quantile(selected[:, 0], torch.tensor([.1,.9], dtype=row.dtype))*100).tolist()}


def run(folder, dest):
    primary = json.loads((folder/"freeze.json").read_text())["primary"]
    report = {"description": "Exploratory post-hoc attribution of previously fixed Sketchy actions", "primary": primary, "splits": {}}
    with ZipFile(folder/"per-query.zip") as archive:
        for split in ("sketchy_1", "sketchy_2"):
            d = torch.load(io.BytesIO(archive.read(split+".pt")), map_location="cpu", weights_only=True)
            base = d["metrics"]["regional"].double()
            delta = d["metrics"][primary].double() - base
            before, after = d["base_correct"], d["primary_correct"]
            adopt = d["alpha/"+primary] > 0
            gain = d["predicted_gain/"+primary].double()
            if delta[~adopt].abs().max() != 0:
                raise ValueError("An unmodified action changed the original metrics")
            masks = {"unmodified": ~adopt, "fixed_first": ~before & after,
                     "broke_first": before & ~after, "kept_right_but_modified": adopt & before & after,
                     "kept_wrong_but_modified": adopt & ~before & ~after}
            if not torch.stack(list(masks.values())).sum(0).eq(1).all():
                raise ValueError("Groups do not form a partition")
            rows = {k: summarize(base, m, delta, gain) for k,m in masks.items()}
            # A rate- or topic-dependent stopping rule is evaluated post hoc only.
            mask = adopt
            a = gain[mask]
            b = delta[mask, 0]
            r = float(torch.corrcoef(torch.stack((a, b)))[0, 1])
            check = sum(torch.tensor(v["contribution_to_all_query_mean_pp"],dtype=torch.double) for v in rows.values() if v["queries"])
            if not torch.allclose(check, delta.mean(0)*100):
                raise ValueError("The partition failed to recover the official aggregate")
            cut = gain[mask].quantile(torch.tensor([.25,.5,.75],dtype=torch.double))
            bins = torch.bucketize(gain, cut)
            ranker = [summarize(base, mask & (bins==i), delta, gain) for i in range(4)]
            # At Ext-2 P@200 is the number of positives in the AP denominator divided by 200.
            detail = None
            if split == "sketchy_2":
                old_count = (base[:, 1]*200).round().long()
                new_count = ((base[:, 1]+delta[:, 1])*200).round().long()
                detail = {}
                for name,m in (("more",new_count>old_count),("same",new_count==old_count),("fewer",new_count<old_count)):
                    chosen = adopt & m
                    detail[name] = summarize(base, chosen, delta, gain)
                    detail[name]["mean_change_count"] = float((new_count-old_count)[chosen].double().mean()) if chosen.any() else None
                detail["positive_from_no_hits"] = int(((old_count==0)&(new_count>0)).sum())
                detail["lost_all_hits"] = int(((old_count>0)&(new_count==0)).sum())
            report["splits"][split] = {"groups": rows, "predicted_gain_AP_pearson_on_adopted": r,
                                       "predicted_gain_quartiles": ranker, "cutoff_count_change": detail}
    atomic_json(dest, report)
    for name,data in report["splits"].items():
        print(name, {k:(v["queries"],v.get("contribution_to_all_query_mean_pp")) for k,v in data["groups"].items()},
              "forecast corr",data["predicted_gain_AP_pearson_on_adopted"],flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-folder", type=Path, default=Path("docs/results/latent-intent-ranking"))
    parser.add_argument("--output", type=Path, default=Path("/content/sbir/results/latent-intent-ranking/decision-diagnosis.json"))
    a=parser.parse_args()
    run(a.result_folder, a.output)
