"""Strict nested screen for the camera-compensated native-motion innovation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from experiments import run_okutama_paired_detail_innovation as ancestry
from hac.actor_memory_base import canonical_hash, file_sha256
from hac.native_motion_data import load_native_motion_cache, motion_geometry, phase_inputs
from hac.native_motion_innovation import ARMS, NativeMotionInnovation, bounded_motion_candidate, motion_loss
from hac.rgb_witness_policy import WitnessPolicy, fit_policy, policy_features, route

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"experiments/okutama_native_motion_innovation_protocol.json"
ARFTR=ROOT/".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz"
CACHE=ROOT/".runs/research_20260920/native_motion_innovation_v1/cache"
BENCHMARK=ROOT/".runs/research_20260920/native_motion_benchmark16"
BENCHMARK_REPEAT=ROOT/".runs/research_20260920/native_motion_benchmark16_repeat"
CAPACITY=ROOT/".runs/research_20260920/native_motion_innovation_v1/capacity_before_results.json"
DEFAULT_RUN=ROOT/".runs/research_20260920/native_motion_innovation_v1"
LOCK_STATUS="NATIVE_MOTION_SCREEN_LOCKED_BEFORE_TASK_FITS"


def read_json(path:Path)->dict:return json.loads(path.read_text(encoding="utf-8"))


def write_json(path:Path,value:dict)->None:
    payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8")!=payload:raise RuntimeError(f"Immutable artifact differs: {path}")
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(payload,encoding="utf-8")


def write_npz(path:Path,**arrays)->None:
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays) or any(not np.array_equal(saved[k],v,equal_nan=True) for k,v in arrays.items()):
                raise RuntimeError(f"Immutable NPZ differs: {path}")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:np.savez_compressed(stream,**arrays)


def record(path:Path)->dict:
    return {"path":str(path.resolve().relative_to(ROOT.resolve())).replace("\\","/"),"bytes":path.stat().st_size,"sha256":file_sha256(path)}


def data()->dict:
    d=ancestry.data();cache=load_native_motion_cache(CACHE,d["sample_ids"])
    d["motion_crops"]=cache.crops;d["motion_frame_available"]=cache.frame_available
    d["motion_available"]=cache.phase_available;d["motion_body_geometry"]=cache.geometry
    d["motion_camera"]=cache.camera;d["motion_geometry"]=motion_geometry(cache.geometry,cache.camera)
    d["motion_source_image_sha256"]=cache.source_image_sha256
    return d


def inner_rows(d:dict,outer:int,inner:int):return ancestry.inner_rows(d,outer,inner)


@dataclass(frozen=True)
class GeometryTransform:
    mean:np.ndarray;scale:np.ndarray
    def apply(self,value:np.ndarray)->np.ndarray:return ((np.asarray(value,np.float32)-self.mean)/self.scale).astype(np.float32)


def geometry_transform(run:Path,d:dict,fit_rows:np.ndarray,scope:str)->tuple[GeometryTransform,Path]:
    path=run/"transforms"/f"{scope}.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if not np.array_equal(saved["fit_rows"],fit_rows) or str(saved["fit_ids_sha256"].item())!=canonical_hash(d["sample_ids"][fit_rows].tolist()):
                raise RuntimeError("Geometry transform population mismatch")
            return GeometryTransform(saved["mean"],saved["scale"]),path
    values=np.asarray(d["motion_geometry"][fit_rows],np.float64);mean=values.mean(0);scale=values.std(0);scale[scale==0]=1
    transform=GeometryTransform(mean.astype(np.float32),scale.astype(np.float32))
    write_npz(path,mean=transform.mean,scale=transform.scale,fit_rows=np.asarray(fit_rows,np.int64),
              fit_ids_sha256=np.asarray(canonical_hash(d["sample_ids"][fit_rows].tolist())))
    return transform,path


def _save_checkpoint(path:Path,payload:dict)->None:
    if path.exists():raise RuntimeError(f"Refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:torch.save(payload,stream)


def load_predictions(path:Path)->dict:
    with np.load(path,allow_pickle=False) as saved:return {k:saved[k] for k in saved.files}


def fit_predict(d:dict,fit_rows:np.ndarray,held_rows:np.ndarray,fit_anchor:np.ndarray,held_anchor:np.ndarray,
                arm:str,directory:Path,lock_sha256:str,device:str,transform:GeometryTransform,
                transform_path:Path,ancestry_receipt:dict)->dict:
    receipt_path=directory/"receipt.json"
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        if (receipt.get("execution_lock_sha256")!=lock_sha256
                or file_sha256(directory/"checkpoint.pt")!=receipt.get("checkpoint_sha256")
                or file_sha256(directory/"predictions.npz")!=receipt.get("predictions_sha256")):
            raise RuntimeError("Completed native-motion fit changed")
        return receipt
    if directory.exists() and any(directory.iterdir()):raise RuntimeError(f"Partial fit directory: {directory}")
    directory.mkdir(parents=True,exist_ok=False);torch.manual_seed(42);np.random.seed(42)
    model=NativeMotionInnovation().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    rng=np.random.default_rng(42);order=np.empty(0,np.int64);cursor=0;losses=[];started=time.perf_counter();model.train()
    for update in range(1,257):
        if cursor+128>len(order):order=rng.permutation(len(fit_rows));cursor=0
        local=order[cursor:cursor+128];cursor+=len(local);rows=fit_rows[local]
        x=torch.from_numpy(phase_inputs(d["motion_crops"][rows],arm)).to(device)
        g=torch.from_numpy(transform.apply(d["motion_geometry"][rows])).to(device)
        available=torch.from_numpy(d["motion_available"][rows]).to(device)
        anchor=torch.from_numpy(np.asarray(fit_anchor[local],np.float32)).to(device)
        labels=torch.from_numpy(d["labels"][rows].astype(np.int64)).to(device)
        optimizer.zero_grad(set_to_none=True);delta=model(x,g);loss,pieces=motion_loss(delta,anchor,labels,available)
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite native-motion loss")
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));optimizer.step()
        if update==1 or update%16==0:losses.append({"update":update,"loss":float(loss.detach()),"gradient_norm":gradient,
            **{k:float(v.detach()) for k,v in pieces.items()}})
    if device.startswith("cuda"):torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    _save_checkpoint(directory/"checkpoint.pt",{"state_dict":model.state_dict(),"arm":arm,"seed":42})
    deltas=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(held_rows),256):
            rows=held_rows[start:start+256]
            x=torch.from_numpy(phase_inputs(d["motion_crops"][rows],arm)).to(device)
            g=torch.from_numpy(transform.apply(d["motion_geometry"][rows])).to(device)
            deltas.append(model(x,g).float().cpu().numpy())
    delta=np.concatenate(deltas).astype(np.float64)
    candidate,eligible=bounded_motion_candidate(held_anchor,delta,d["motion_available"][held_rows])
    features=policy_features(held_anchor,candidate,delta,d["motion_body_geometry"][held_rows],d["motion_available"][held_rows])
    write_npz(directory/"predictions.npz",sample_ids=d["sample_ids"][held_rows],held_rows=held_rows,
        anchor_probabilities=np.asarray(held_anchor,np.float64),candidate_probabilities=candidate,delta=delta,
        eligible=eligible,policy_features=features,available=d["motion_available"][held_rows],
        body_geometry=d["motion_body_geometry"][held_rows])
    receipt={"status":"NATIVE_MOTION_PRODUCER_FIT_COMPLETE_OUTER_METRICS_EMBARGOED","arm":arm,"seed":42,
        "execution_lock_sha256":lock_sha256,"fit_rows":len(fit_rows),"held_rows":len(held_rows),"updates":256,
        "parameter_count":model.trainable_parameters,"elapsed_seconds":elapsed,"outer_held_labels_read":0,
        "fit_ids_sha256":canonical_hash(d["sample_ids"][fit_rows].tolist()),"held_ids_sha256":canonical_hash(d["sample_ids"][held_rows].tolist()),
        "checkpoint_sha256":file_sha256(directory/"checkpoint.pt"),"predictions_sha256":file_sha256(directory/"predictions.npz"),
        "transform_path":str(transform_path.relative_to(ROOT)).replace("\\","/"),"transform_sha256":file_sha256(transform_path),
        "loss_curve":losses,"ancestry":ancestry_receipt}
    write_json(receipt_path,receipt);del model,optimizer
    if device.startswith("cuda"):torch.cuda.empty_cache()
    return receipt


def _save_policy(path:Path,policy:WitnessPolicy|None,receipt:dict)->None:
    arrays={"receipt_json":np.asarray(json.dumps(receipt,sort_keys=True))}
    if policy is not None:arrays.update(mean=policy.mean,scale=policy.scale,utility_coef=policy.utility_coef,
        utility_intercept=policy.utility_intercept,nll_coef=policy.nll_coef,nll_intercept=np.asarray(policy.nll_intercept))
    write_npz(path,**arrays)


def load_policy(path:Path)->tuple[WitnessPolicy|None,dict]:
    with np.load(path,allow_pickle=False) as saved:
        receipt=json.loads(str(saved["receipt_json"].item()))
        if "mean" not in saved.files:return None,receipt
        return WitnessPolicy(saved["mean"],saved["scale"],saved["utility_coef"],saved["utility_intercept"],saved["nll_coef"],float(saved["nll_intercept"])),receipt


def run_arm_fold(run:Path,d:dict,arm:str,outer:int,device:str)->dict:
    folder=run/arm/f"fold-{outer}";receipt_path=folder/"receipt.json"
    if receipt_path.exists():return read_json(receipt_path)
    lock_sha=file_sha256(run/"execution_lock.json");outer_train=np.flatnonzero(d["folds"]!=outer)
    oof_candidate=np.full((len(outer_train),3),np.nan);oof_anchor=np.full((len(outer_train),3),np.nan)
    oof_features=np.full((len(outer_train),19),np.nan);locations={row:i for i,row in enumerate(outer_train.tolist())};inner_receipts=[]
    for inner in range(5):
        fit,held=inner_rows(d,outer,inner);fit_anchor,held_anchor,anc=ancestry.inner_anchors(d,outer,inner)
        transform,tpath=geometry_transform(run,d,fit,f"outer-{outer}_inner-{inner}");directory=folder/f"inner-{inner}"
        fit_predict(d,fit,held,fit_anchor,held_anchor,arm,directory,lock_sha,device,transform,tpath,anc)
        saved=load_predictions(directory/"predictions.npz");dest=np.asarray([locations[x] for x in held])
        oof_candidate[dest]=saved["candidate_probabilities"];oof_anchor[dest]=saved["anchor_probabilities"];oof_features[dest]=saved["policy_features"]
        inner_receipts.append(record(directory/"receipt.json"))
    if not all(np.isfinite(x).all() for x in (oof_candidate,oof_anchor,oof_features)):raise RuntimeError("Incomplete inner motion OOF")
    policy,policy_receipt=fit_policy(oof_features,oof_anchor,oof_candidate,d["labels"][outer_train],d["scenarios"][outer_train])
    folder.mkdir(parents=True,exist_ok=True);_save_policy(folder/"policy.npz",policy,policy_receipt)
    train,held,train_anchor,anc=ancestry.outer_anchors(d,outer)
    if not np.array_equal(train,outer_train):raise RuntimeError("Outer ancestry population mismatch")
    transform,tpath=geometry_transform(run,d,train,f"outer-{outer}_final")
    fit_predict(d,train,held,train_anchor,d["anchor"][held],arm,folder/"final",lock_sha,device,transform,tpath,anc)
    saved=load_predictions(folder/"final"/"predictions.npz")
    routed,choices,route_receipt=route(saved["anchor_probabilities"],saved["candidate_probabilities"],saved["policy_features"],policy)
    write_npz(folder/"predictions.npz",sample_ids=saved["sample_ids"],held_rows=held,
        anchor_probabilities=saved["anchor_probabilities"],candidate_probabilities=saved["candidate_probabilities"],
        routed_probabilities=routed,choices=choices,delta=saved["delta"],eligible=saved["eligible"],policy_features=saved["policy_features"])
    receipt={"status":"NATIVE_MOTION_ARM_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED","arm":arm,"outer_fold":outer,
        "outer_held_labels_read":0,"inner_receipts":inner_receipts,"final_receipt":record(folder/"final"/"receipt.json"),
        "policy_receipt":policy_receipt,"route_receipt_without_labels":route_receipt,
        "policy_sha256":file_sha256(folder/"policy.npz"),"predictions_sha256":file_sha256(folder/"predictions.npz")}
    write_json(receipt_path,receipt);return receipt


def prepare(run:Path)->dict:
    protocol=read_json(PROTOCOL);summary=read_json(CACHE/"summary.json")
    if summary.get("status")!="NATIVE_MOTION_CACHE_COMPLETE":raise RuntimeError("Complete native-motion cache required")
    d=data();dependencies=[PROTOCOL,Path(__file__),ROOT/"experiments/audit_okutama_native_motion_innovation.py",ROOT/"src/hac/native_motion_data.py",ROOT/"src/hac/native_motion_innovation.py",
        ROOT/"src/hac/rgb_witness_policy.py",ROOT/"tests/test_native_motion_innovation.py",ROOT/"experiments/analyze_native_motion_capacity.py",ARFTR,CAPACITY,
        CACHE/"execution_lock.json",CACHE/"summary.json",BENCHMARK/"shard-0000-0016.npz",BENCHMARK_REPEAT/"shard-0000-0016.npz",
        ROOT/"experiments/run_okutama_paired_detail_innovation.py"]
    dependencies+=list(CACHE.glob("shard-*.npz"))+list(CACHE.glob("shard-*.json"));populations={};ancestor_receipts=[]
    for outer in range(5):
        for inner in range(5):
            fit,held=inner_rows(d,outer,inner);_,_,anc=ancestry.inner_anchors(d,outer,inner)
            p=ROOT/".runs/research_20260916/source_posture_failure_router_v1/inner_ancestors/populations"/anc["population_id"]/"receipt.json"
            ancestor_receipts.append(p);populations[f"o{outer}_i{inner}"]={"fit_rows":len(fit),"held_rows":len(held),
                "fit_ids_sha256":canonical_hash(d["sample_ids"][fit].tolist()),"held_ids_sha256":canonical_hash(d["sample_ids"][held].tolist()),
                "ancestor_population_id":anc["population_id"]}
    dependencies+=ancestor_receipts;parameter_count=NativeMotionInnovation().trainable_parameters
    lock={"status":LOCK_STATUS,"rows":len(d["labels"]),"arms":list(ARMS),"primary":"signed_motion","outer_folds":5,
        "producer_fits_planned":90,"policy_bundles_planned_max":15,"task_fits_at_lock":0,"outer_labels_read_at_lock":0,
        "parameter_count_each_arm":parameter_count,"sample_ids_sha256":canonical_hash(d["sample_ids"].tolist()),
        "available_rows":int(d["motion_available"].all(1).sum()),"populations":populations,
        "dependencies":[record(p) for p in sorted(set(dependencies),key=str)]}
    run.mkdir(parents=True,exist_ok=True);write_json(run/"execution_lock.json",lock);return lock


def validate_lock(run:Path)->dict:
    lock=read_json(run/"execution_lock.json")
    if lock.get("status")!=LOCK_STATUS or lock.get("task_fits_at_lock")!=0:raise RuntimeError("Invalid native-motion lock")
    for item in lock["dependencies"]:
        path=ROOT/item["path"]
        if not path.is_file() or path.stat().st_size!=item["bytes"] or file_sha256(path)!=item["sha256"]:raise RuntimeError(f"Locked dependency changed: {item['path']}")
    return lock


def preflight(run:Path,device:str)->dict:
    lock=validate_lock(run);d=data()
    if device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    for outer in range(5):
        train,held,train_anchor,_=ancestry.outer_anchors(d,outer)
        if not np.isfinite(train_anchor).all() or not np.isfinite(d["anchor"][held]).all():raise RuntimeError("Outer ancestry failed")
        for inner in range(5):
            fit,ih=inner_rows(d,outer,inner);fa,ha,_=ancestry.inner_anchors(d,outer,inner)
            if set(d["scenarios"][fit])&set(d["scenarios"][ih]) or not np.isfinite(fa).all() or not np.isfinite(ha).all():raise RuntimeError("Inner ancestry failed")
    torch.manual_seed(42);phases=torch.randn(32,2,6,64,64,device=device);geometry=torch.randn(32,12,device=device)
    anchor=torch.tensor([[.05,.50,.45]],device=device).repeat(32,1);labels=torch.arange(32,device=device)%3
    available=torch.ones(32,2,dtype=torch.bool,device=device);model=NativeMotionInnovation().to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    for _ in range(3):opt.zero_grad();delta=model(phases,geometry);loss,_=motion_loss(delta,anchor,labels,available);loss.backward();opt.step()
    active=sum(int(p.grad is not None and torch.count_nonzero(p.grad).item()>0) for p in model.parameters())
    if active!=sum(1 for _ in model.parameters()):raise RuntimeError("Inactive native-motion parameters")
    result={"status":"NATIVE_MOTION_PREFLIGHT_PASS_QUEUE_AUTHORIZED","rows":len(d["labels"]),
        "available_rows":int(d["motion_available"].all(1).sum()),"inner_populations_checked":25,"classifier_fits":0,
        "device":device,"cuda_available":torch.cuda.is_available(),"parameter_count":model.trainable_parameters,
        "active_parameter_tensors":active,"environment":{n:importlib.metadata.version(n) for n in ("numpy","scipy","scikit-learn","torch")}}
    write_json(run/"preflight.json",result);return result


def queue(run:Path,device:str,maximum_bundles:int|None=None)->dict:
    validate_lock(run)
    if read_json(run/"preflight.json").get("status")!="NATIVE_MOTION_PREFLIGHT_PASS_QUEUE_AUTHORIZED":raise RuntimeError("Preflight required")
    d=data();started=time.perf_counter();before=sum((run/a/f"fold-{o}"/"receipt.json").exists() for a in ARMS for o in range(5));new=0
    for arm in ARMS:
        for outer in range(5):
            if (run/arm/f"fold-{outer}"/"receipt.json").exists():continue
            result=run_arm_fold(run,d,arm,outer,device);new+=1
            print(json.dumps({"event":"native_motion_bundle_complete","arm":arm,"outer":outer,
                "policy":result["policy_receipt"]["status"],"elapsed":time.perf_counter()-started}),flush=True)
            if maximum_bundles is not None and new>=maximum_bundles:break
        if maximum_bundles is not None and new>=maximum_bundles:break
    result={"status":"NATIVE_MOTION_QUEUE_COMPLETE" if before+new==15 else "NATIVE_MOTION_QUEUE_PARTIAL",
        "bundles_complete":before+new,"bundles_total":15,"new_bundles":new,"elapsed_seconds":time.perf_counter()-started}
    write_json(run/f"queue_receipt_{before+new:02d}.json",result);return result


def metrics(y:np.ndarray,p:np.ndarray)->dict:
    pred=p.argmax(1);precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    return {"macro_f1":float(f1.mean()),"accuracy":float((pred==y).mean()),"errors":int((pred!=y).sum()),
        "nll":float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
        "brier_sum":float(np.square(p-np.eye(3)[y]).sum(1).mean()),"per_class_f1":f1.tolist(),
        "precision":precision.tolist(),"recall":recall.tolist(),"support":support.tolist(),
        "confusion":confusion_matrix(y,pred,labels=[0,1,2]).tolist()}


def bootstrap(d:dict,left:np.ndarray,right:np.ndarray,level:float=.95)->dict:
    groups=np.unique(d["scenarios"]);cms=[]
    for p in (left,right):cms.append(np.asarray([confusion_matrix(d["labels"][d["scenarios"]==s],p[d["scenarios"]==s].argmax(1),labels=[0,1,2]) for s in groups]))
    rng=np.random.default_rng(20260920);draws=rng.integers(0,len(groups),size=(100000,len(groups)))
    def f1(cm):
        diag=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-1)+cm.sum(-2)
        return np.divide(2*diag,den,out=np.zeros_like(diag,dtype=float),where=den!=0).mean(-1)
    delta=f1(cms[0][draws].sum(1))-f1(cms[1][draws].sum(1));alpha=(1-level)/2
    return {"groups":len(groups),"draws":100000,"seed":20260920,"level":level,
        "interval":np.quantile(delta,[alpha,1-alpha]).tolist(),"mean":float(delta.mean())}


def summarize(run:Path)->dict:
    validate_lock(run);audit=read_json(run/"independent_audit.json")
    if audit.get("status")!="NATIVE_MOTION_INDEPENDENT_REPLAY_PASS":raise RuntimeError("Independent replay required")
    d=data();raw={};routed={}
    for arm in ARMS:
        raw[arm]=np.full_like(d["anchor"],np.nan);routed[arm]=np.full_like(d["anchor"],np.nan)
        for outer in range(5):
            saved=load_predictions(run/arm/f"fold-{outer}"/"predictions.npz");rows=saved["held_rows"]
            raw[arm][rows]=saved["candidate_probabilities"];routed[arm][rows]=saved["routed_probabilities"]
    scores={"arftr":metrics(d["labels"],d["anchor"])};details={};anchor_correct=d["anchor"].argmax(1)==d["labels"]
    for arm in ARMS:
        scores[arm]={"raw":metrics(d["labels"],raw[arm]),"routed":metrics(d["labels"],routed[arm])};correct=routed[arm].argmax(1)==d["labels"]
        details[arm]={"rescues":int((correct&~anchor_correct).sum()),"harms":int((~correct&anchor_correct).sum()),
            "net":int(correct.sum()-anchor_correct.sum()),"per_fold_net":[int((correct[d["folds"]==f]&~anchor_correct[d["folds"]==f]).sum()-(~correct[d["folds"]==f]&anchor_correct[d["folds"]==f]).sum()) for f in range(5)],
            "bootstrap_vs_arftr":bootstrap(d,routed[arm],d["anchor"])}
    adjusted=1-(.05/2);comparisons={other:bootstrap(d,raw["signed_motion"],raw[other],adjusted) for other in ("appearance","phase_destroyed")}
    mechanism={"signed_minus_appearance_at_least_0_002":bool(scores["signed_motion"]["raw"]["macro_f1"]-scores["appearance"]["raw"]["macro_f1"]>=.002),
        "positive_adjusted_intervals":bool(all(v["interval"][0]>0 for v in comparisons.values()))}
    primary=scores["signed_motion"]["routed"];info=details["signed_motion"];base=scores["arftr"]
    promotion={"macro_f1":bool(primary["macro_f1"]>=.8588364807373623),"net":bool(info["net"]>=25),
        "positive_each_fold":bool(all(x>0 for x in info["per_fold_net"])),"scenario_interval":bool(info["bootstrap_vs_arftr"]["interval"][0]>0),
        "nll":bool(primary["nll"]<=.3773500139530721),"brier":bool(primary["brier_sum"]<=.21350592452064215),
        "class_f1":bool(min(np.asarray(primary["per_class_f1"])-np.asarray(base["per_class_f1"]))>=-.005),
        "independent_replay_and_ancestry":True}
    passed=all(mechanism.values());promoted=passed and all(promotion.values())
    result={"status":"NATIVE_MOTION_SCREEN_COMPLETE","scores":scores,"details":details,
        "mechanism_comparisons":comparisons,"mechanism_gates":mechanism,"promotion_gates":promotion,
        "mechanism_pass":passed,"promoted":promoted,"retained_arftr_changed":False,
        "next_action":"confirmation_seed_screen" if promoted else "close_fixed_motion_design"}
    write_json(run/"summary.json",result);return result


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--stage",choices=("prepare","preflight","queue","summarize"),required=True)
    parser.add_argument("--device",default="cpu");parser.add_argument("--maximum-bundles",type=int)
    args=parser.parse_args();run=args.run.resolve()
    result=prepare(run) if args.stage=="prepare" else preflight(run,args.device) if args.stage=="preflight" else queue(run,args.device,args.maximum_bundles) if args.stage=="queue" else summarize(run)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
