"""Recalculate the fixed native-motion action capacity without fitting a model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    sys.path.insert(0,str(Path(__file__).resolve().parent))

from experiments import run_okutama_paired_detail_innovation as ancestry

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/".runs/research_20260920/native_motion_innovation_v1/capacity_before_results.json"


def calculate()->dict:
    d=ancestry.data();p=d["anchor"];labels=d["labels"];prediction=p.argmax(1)
    eligible=p[:,0]<np.minimum(p[:,1],p[:,2]);pair_label=labels>0
    within_bound=np.abs(np.log(p[:,1]/p[:,2]))<.5
    reachable=eligible&(prediction!=labels)&pair_label&within_bound
    susceptible=eligible&(prediction==labels)&pair_label&within_bound
    oracle=prediction.copy();oracle[reachable]=labels[reachable]
    return {"status":"FIXED_ELIGIBILITY_CAPACITY_DIAGNOSTIC_BEFORE_MOTION_RESULTS",
        "anchor":"retained_ARFTR","eligibility":"ARFTR_top2_are_standing_and_walking; pair_logit_delta_bound_0.5",
        "population":len(labels),"eligible_rows":int(eligible.sum()),
        "eligible_ARFTR_errors_total":int((eligible&(prediction!=labels)).sum()),
        "eligible_target_pair_errors":int((eligible&(prediction!=labels)&pair_label).sum()),
        "eligible_sitting_label_errors_invariantly_unreachable":int((eligible&(prediction!=labels)&~pair_label).sum()),
        "reachable_errors_under_bound":int(reachable.sum()),
        "reachable_errors_by_outer_fold":[int(reachable[d["folds"]==fold].sum()) for fold in range(5)],
        "harm_susceptible_correct_rows":int(susceptible.sum()),
        "no_harm_oracle_macro_f1":float(f1_score(labels,oracle,average="macro")),
        "no_harm_oracle_uplift_percentage_points":float(100*(f1_score(labels,oracle,average="macro")-f1_score(labels,prediction,average="macro"))),
        "interpretation":"diagnostic upper bound only; not deployable performance and not used to choose any threshold"}


if __name__=="__main__":
    value=calculate();expected=json.loads(OUTPUT.read_text(encoding="utf-8"))
    if value!=expected:raise RuntimeError(f"Capacity artifact differs from recalculation: {value}")
    print(json.dumps(value,indent=2))
