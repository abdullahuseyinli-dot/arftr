"""Independent checkpoint/cache replay for the native-motion screen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from experiments import run_okutama_native_motion_innovation as experiment
from hac.native_motion_data import phase_inputs
from hac.native_motion_innovation import ARMS,NativeMotionInnovation,bounded_motion_candidate
from hac.rgb_witness_policy import policy_features,route

ROOT=Path(__file__).resolve().parents[1]


def transform(path:Path,value:np.ndarray)->np.ndarray:
    with np.load(path,allow_pickle=False) as saved:return ((value-saved["mean"])/saved["scale"]).astype(np.float32)


def replay_fit(d:dict,directory:Path,device:str)->dict:
    receipt=experiment.read_json(directory/"receipt.json");saved=experiment.load_predictions(directory/"predictions.npz")
    checkpoint=torch.load(directory/"checkpoint.pt",map_location=device,weights_only=False)
    model=NativeMotionInnovation().to(device);model.load_state_dict(checkpoint["state_dict"]);model.eval()
    rows=saved["held_rows"].astype(np.int64);values=[]
    with torch.no_grad():
        for start in range(0,len(rows),256):
            batch=rows[start:start+256]
            x=torch.from_numpy(phase_inputs(d["motion_crops"][batch],receipt["arm"])).to(device)
            g=torch.from_numpy(transform(ROOT/receipt["transform_path"],d["motion_geometry"][batch])).to(device)
            values.append(model(x,g).float().cpu().numpy())
    delta=np.concatenate(values).astype(np.float64)
    candidate,eligible=bounded_motion_candidate(saved["anchor_probabilities"],delta,d["motion_available"][rows])
    features=policy_features(saved["anchor_probabilities"],candidate,delta,d["motion_body_geometry"][rows],d["motion_available"][rows])
    maxima={"delta":float(np.max(np.abs(delta-saved["delta"]))),
            "candidate":float(np.max(np.abs(candidate-saved["candidate_probabilities"]))),
            "policy_features":float(np.max(np.abs(features-saved["policy_features"])))}
    if not np.array_equal(eligible,saved["eligible"]) or any(value!=0 for value in maxima.values()):
        raise RuntimeError(f"Native-motion fit replay differs: {directory}: {maxima}")
    return {"path":str(directory.relative_to(ROOT)).replace("\\","/"),"held_rows":len(rows),"max_abs":maxima}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,default=experiment.DEFAULT_RUN)
    parser.add_argument("--device",default="cpu");args=parser.parse_args();run=args.run.resolve()
    experiment.validate_lock(run)
    if args.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    d=experiment.data();replays=[];route_replays=[]
    for arm in ARMS:
        for outer in range(5):
            folder=run/arm/f"fold-{outer}"
            for inner in range(5):replays.append(replay_fit(d,folder/f"inner-{inner}",args.device))
            replays.append(replay_fit(d,folder/"final",args.device))
            saved=experiment.load_predictions(folder/"final"/"predictions.npz")
            expected=experiment.load_predictions(folder/"predictions.npz");policy,_=experiment.load_policy(folder/"policy.npz")
            routed,choices,_=route(saved["anchor_probabilities"],saved["candidate_probabilities"],saved["policy_features"],policy)
            maximum=float(np.max(np.abs(routed-expected["routed_probabilities"])))
            if maximum!=0 or not np.array_equal(choices,expected["choices"]):raise RuntimeError("Policy route replay differs")
            route_replays.append({"arm":arm,"outer":outer,"max_abs":maximum,"interventions":int(choices.sum())})
    with np.load(experiment.BENCHMARK/"shard-0000-0016.npz",allow_pickle=False) as left,np.load(experiment.BENCHMARK_REPEAT/"shard-0000-0016.npz",allow_pickle=False) as right:
        repeat={name:bool(np.array_equal(left[name],right[name])) for name in left.files}
    if not all(repeat.values()):raise RuntimeError("Repeated label-blind extraction differs")
    result={"status":"NATIVE_MOTION_INDEPENDENT_REPLAY_PASS","producer_replays":len(replays),
        "policy_replays":len(route_replays),"producer_details":replays,"policy_details":route_replays,
        "benchmark_repeat_exact":repeat,"outer_held_labels_read_during_replay":0,
        "locked_dependencies_verified":True,"retain_action_present":True}
    experiment.write_json(run/"independent_audit.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
