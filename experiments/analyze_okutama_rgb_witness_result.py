"""Fixed post-result diagnostics for the completed RGB-witness screen; no fits."""
from __future__ import annotations
import hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src",ROOT/"experiments"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from experiments import run_okutama_rgb_witness as locked
from hac.rgb_witness import ARMS

RUN=ROOT/".runs/research_20260920/rgb_witness_v1"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
    payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8")!=payload:raise RuntimeError(f"Immutable diagnostic differs:{path}")
    else:
        with path.open("x",encoding="utf-8") as f:f.write(payload)

def main():
    d=locked.data();y=d["labels"];anchor=d["anchor"];old=anchor.argmax(1);ac=old==y
    result={"status":"RGB_WITNESS_FIXED_POST_RESULT_DIAGNOSTIC_COMPLETE","model_fits":0,"threshold_changes":0,
        "baseline_sha256":sha(locked.ARFTR),"summary_v2_sha256":sha(RUN/"summary_v2.json"),"arms":{},
        "unavailable_rows":int((~d["rgb_available"].all(1)).sum())}
    raws={}
    for arm in ARMS:
        p=np.full_like(anchor,np.nan);delta=np.full(len(y),np.nan);eligible=np.zeros(len(y),bool);routed=np.full_like(anchor,np.nan)
        support=[]
        for outer in range(5):
            a=locked.load_predictions(RUN/arm/f"fold-{outer}"/"predictions.npz");rows=a["held_rows"]
            p[rows]=a["candidate_probabilities"];routed[rows]=a["routed_probabilities"];delta[rows]=a["delta"];eligible[rows]=a["eligible"]
            support.append(locked.read_json(RUN/arm/f"fold-{outer}"/"receipt.json")["policy_receipt"])
        new=p.argmax(1);correct=new==y;change=new!=old;raws[arm]=p
        transitions={"rescues":int((correct&~ac).sum()),"harms":int((~correct&ac).sum()),
            "wrong_to_other_wrong":int((~correct&~ac&change).sum()),"class_changes":int(change.sum()),
            "net":int(correct.sum()-ac.sum()),"per_fold_net":[int((correct[d["folds"]==f]&~ac[d["folds"]==f]).sum()-(~correct[d["folds"]==f]&ac[d["folds"]==f]).sum()) for f in range(5)]}
        result["arms"][arm]={"eligible_rows":int(eligible.sum()),"delta_abs_median":float(np.median(np.abs(delta[eligible]))),
            "delta_abs_p95":float(np.quantile(np.abs(delta[eligible]),.95)),"raw_transitions":transitions,
            "routed_exact_anchor":bool(np.array_equal(routed,anchor)),"policy_support_by_fold":support}
    c=raws["conditional"];direct=raws["direct"]
    result["conditional_vs_direct"]={"different_classes":int((c.argmax(1)!=direct.argmax(1)).sum()),
        "conditional_correct_direct_wrong":int(((c.argmax(1)==y)&(direct.argmax(1)!=y)).sum()),
        "conditional_wrong_direct_correct":int(((c.argmax(1)!=y)&(direct.argmax(1)==y)).sum())}
    reasons=Counter()
    for path in sorted((RUN/"cache").glob("shard-*.json")):
        for row in locked.read_json(path)["row_audits"]:
            if not row["decode"]["decode_valid"]:reasons[row["decode"]["reason"]]+=1
    result["unavailable_reasons"]=dict(reasons)
    write_new(RUN/"postmortem.json",result)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":main()
