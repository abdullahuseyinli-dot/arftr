"""Independent no-fit replay of RGB-witness producers, policies and ancestry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_EXPERIMENTS_DIRECTORY=str(Path(__file__).resolve().parent)
if _EXPERIMENTS_DIRECTORY not in sys.path:
    sys.path.insert(0,_EXPERIMENTS_DIRECTORY)
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from experiments import run_okutama_rgb_witness as runner
from hac.actor_memory_base import canonical_hash, file_sha256
from hac.rgb_witness import ARMS, RGBWitness, bounded_pair_candidate
from hac.rgb_witness_policy import fit_policy, route

ROOT=Path(__file__).resolve().parents[1]


def replay_producer(d: dict,directory: Path,device: str) -> dict:
    receipt=runner.read_json(directory/"receipt.json")
    saved=runner.load_predictions(directory/"predictions.npz")
    transform_path=ROOT/receipt["transform_path"]
    if file_sha256(transform_path)!=receipt["transform_sha256"] or file_sha256(directory/"checkpoint.pt")!=receipt["checkpoint_sha256"] or file_sha256(directory/"predictions.npz")!=receipt["predictions_sha256"]:
        raise RuntimeError("Producer output hash changed")
    with np.load(transform_path,allow_pickle=False) as transform_arrays:
        fit_rows=transform_arrays["fit_rows"].astype(np.int64)
        if canonical_hash(d["sample_ids"][fit_rows].tolist())!=receipt["fit_ids_sha256"]:
            raise RuntimeError("Transform/training population mismatch")
    transform=runner._load_transform(transform_path)
    held=saved["held_rows"].astype(np.int64)
    x,g=runner.transformed(d,held,transform)
    checkpoint=torch.load(directory/"checkpoint.pt",map_location=device,weights_only=True)
    model=RGBWitness(receipt["arm"]).to(device);model.load_state_dict(checkpoint["state_dict"]);model.eval()
    values=[]
    with torch.no_grad():
        for start in range(0,len(held),256):
            stop=min(len(held),start+256);rows=held[start:stop]
            out=model(torch.from_numpy(x[start:stop]).to(device),torch.from_numpy(g[start:stop]).to(device),
                torch.from_numpy(d["rgb_available"][rows]).to(device),
                torch.from_numpy(saved["anchor_probabilities"][start:stop].astype(np.float32)).to(device))
            values.append(out["delta"].float().cpu().numpy())
    delta=np.concatenate(values).astype(np.float64)
    candidate,eligible=bounded_pair_candidate(saved["anchor_probabilities"],delta,d["rgb_available"][held])
    max_delta=float(np.max(np.abs(delta-saved["delta"])))
    max_candidate=float(np.max(np.abs(candidate-saved["candidate_probabilities"])))
    if max_delta>1e-7 or max_candidate>1e-7 or not np.array_equal(eligible,saved["eligible"]):
        raise RuntimeError(f"Producer replay mismatch: {directory}")
    if canonical_hash(d["sample_ids"][held].tolist())!=receipt["held_ids_sha256"]:
        raise RuntimeError("Producer held population changed")
    ancestral=receipt["ancestry"]
    if "train_scenarios" in ancestral and set(ancestral["train_scenarios"])&set(ancestral["held_scenarios"]):
        raise RuntimeError("Ancestor train/held scenario overlap")
    return {"directory":str(directory.relative_to(ROOT)).replace("\\","/"),"rows":len(held),
        "max_delta_difference":max_delta,"max_candidate_difference":max_candidate}


def audit(run: Path,device: str) -> dict:
    runner.validate_lock(run);d=runner.data();replays=[];policy_replays=[]
    with np.load(runner.BENCHMARK/"shard-0000-0016.npz",allow_pickle=False) as repeated, np.load(runner.CACHE/"shard-0000-0032.npz",allow_pickle=False) as full:
        extraction_repeat={key:bool(np.array_equal(repeated[key],full[key][:16])) for key in ("sample_ids","features","available","geometry","source_image_sha256")}
    if not all(extraction_repeat.values()):raise RuntimeError("Independent16-row extraction repetition mismatch")
    index={sample:i for i,sample in enumerate(d["sample_ids"].astype(str))}
    old_audits=runner.read_json(runner.OLD_PILOT)["decode_audits"]
    old_matches=[d["rgb_source_image_sha256"][index[row["sample_id"]]]==row["image_sha256"] for row in old_audits]
    if len(old_matches)!=128 or not all(old_matches):raise RuntimeError("Current native decode differs from prior128-center extraction")
    outer_labels_indexed=0
    for arm in ARMS:
        for outer in range(5):
            folder=run/arm/f"fold-{outer}";outer_train=np.flatnonzero(d["folds"]!=outer)
            candidates=np.full((len(outer_train),3),np.nan);anchors=np.full((len(outer_train),3),np.nan)
            features=np.full((len(outer_train),19),np.nan);locations={r:i for i,r in enumerate(outer_train.tolist())}
            for inner in range(5):
                directory=folder/f"inner-{inner}";replays.append(replay_producer(d,directory,device))
                saved=runner.load_predictions(directory/"predictions.npz");dest=np.asarray([locations[x] for x in saved["held_rows"]])
                candidates[dest]=saved["candidate_probabilities"];anchors[dest]=saved["anchor_probabilities"];features[dest]=saved["policy_features"]
                receipt=runner.read_json(directory/"receipt.json")
                if receipt["outer_held_labels_read"]!=0:raise RuntimeError("Producer receipt read held labels")
            policy,replayed_receipt=fit_policy(features,anchors,candidates,d["labels"][outer_train],d["scenarios"][outer_train])
            saved_policy,saved_receipt=runner._load_policy(folder/"policy.npz")
            if replayed_receipt["status"]!=saved_receipt["status"]:
                raise RuntimeError("Policy support/fit status replay mismatch")
            if (policy is None)!=(saved_policy is None):raise RuntimeError("Policy availability replay mismatch")
            final_dir=folder/"final";replays.append(replay_producer(d,final_dir,device))
            final=runner.load_predictions(final_dir/"predictions.npz")
            if policy is not None:
                for left,right in ((policy.mean,saved_policy.mean),(policy.scale,saved_policy.scale),
                    (policy.utility_coef,saved_policy.utility_coef),(policy.utility_intercept,saved_policy.utility_intercept),
                    (policy.nll_coef,saved_policy.nll_coef),(np.asarray(policy.nll_intercept),np.asarray(saved_policy.nll_intercept))):
                    if not np.allclose(left,right,atol=1e-10,rtol=0):raise RuntimeError("Policy parameter replay mismatch")
            routed,choices,_=route(final["anchor_probabilities"],final["candidate_probabilities"],final["policy_features"],policy)
            if not np.array_equal(routed[choices==0],final["anchor_probabilities"][choices==0]):
                raise RuntimeError("Policy failed byte-exact retain")
            if not np.array_equal(final["candidate_probabilities"][:,2],final["anchor_probabilities"][:,2]):
                raise RuntimeError("Candidate changed canonical walking mass")
            released=runner.load_predictions(folder/"predictions.npz")
            maxdiff=float(np.max(np.abs(routed-released["routed_probabilities"])))
            if maxdiff>1e-10 or not np.array_equal(choices,released["choices"]):raise RuntimeError("Route replay mismatch")
            policy_replays.append({"arm":arm,"outer":outer,"status":replayed_receipt["status"],"max_route_difference":maxdiff})
    result={"status":"RGB_WITNESS_INDEPENDENT_REPLAY_PASS","producer_replays":len(replays),"policy_replays":len(policy_replays),
        "outer_held_labels_read":outer_labels_indexed,"max_producer_delta_difference":max(x["max_delta_difference"] for x in replays),
        "max_producer_candidate_difference":max(x["max_candidate_difference"] for x in replays),
        "max_route_difference":max(x["max_route_difference"] for x in policy_replays),
        "independent_16_row_extraction_repeat":extraction_repeat,
        "prior_native_decode_image_hash_matches":sum(old_matches),
        "execution_lock_sha256":file_sha256(run/"execution_lock.json"),"producer_details":replays,"policy_details":policy_replays}
    runner.write_json(run/"independent_audit.json",result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,default=runner.DEFAULT_RUN);p.add_argument("--device",default="cpu")
    a=p.parse_args();print(json.dumps(audit(a.run.resolve(),a.device),indent=2))


if __name__=="__main__":main()
