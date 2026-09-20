"""Label-blind extraction of two disjoint masked RGB witnesses per exact center."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.extract_okutama_source_posture import DINO_SUMMARY, load_observations
from hac.absolute_body_evidence import FrozenAbsoluteDinoCLS, preprocess_rgb
from hac.actor_memory_base import canonical_hash, file_sha256
from hac.body_witness_data import (GAP_FRACTION, MASK_IDS, PADDING_RGB, apply_raw_body_mask,
    crop_raw_body, decode_exact_center, load_verified_source_map, make_body_crop_geometry,
    resolve_center_source)
from hac.image_encoders import load_dinov2_encoder
from hac.rgb_witness_data import FEATURE_DIM, SLOT_NAMES, witness_geometry

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_rgb_witness_protocol.json"
OLD_EXTRACTION = ROOT / ".runs/research_20260913/body_witness_pilot_v1/extraction_receipt.json"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260920/rgb_witness_v1/cache"


def write_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Immutable artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite cache shard: {path}")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def lock_value(observations, shard_size: int) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model = json.loads(DINO_SUMMARY.read_text(encoding="utf-8"))["model"]
    if (model["snapshot"]["revision"] != protocol["feature_cache"]["encoder_revision"]
            or protocol["feature_cache"]["input_size"] != 378
            or tuple(MASK_IDS) != SLOT_NAMES or GAP_FRACTION != .05 or tuple(PADDING_RGB) != (124,116,104)):
        raise RuntimeError("RGB-witness encoder/mask protocol mismatch")
    authority = ROOT / protocol["design_authority"]
    dependencies = [PROTOCOL, authority, Path(__file__), ROOT/"src/hac/rgb_witness_data.py",
                    ROOT/"src/hac/body_witness_data.py", ROOT/"src/hac/absolute_body_evidence.py",
                    DINO_SUMMARY, OLD_EXTRACTION]
    for info in model["snapshot"]["files"].values():
        path = Path(info["path"])
        if file_sha256(path) != info["sha256"]:
            raise RuntimeError("Frozen DINO snapshot changed")
    return {"status":"RGB_WITNESS_LABEL_BLIND_EXTRACTION_LOCK", "rows":len(observations),
        "slot_names":list(SLOT_NAMES),"features_per_slot":FEATURE_DIM,"shard_size":shard_size,
        "extent":1.25,"gap_fraction":GAP_FRACTION,"padding_rgb":list(PADDING_RGB),
        "sample_ids_sha256":canonical_hash([o.sample_id for o in observations]),
        "encoder_revision":model["snapshot"]["revision"],"encoder_snapshot":model["snapshot"],
        "dependencies":[{"path":str(p.resolve().relative_to(ROOT.resolve())).replace("\\","/") if p.resolve().is_relative_to(ROOT.resolve()) else str(p.resolve()),
                         "sha256":file_sha256(p),"bytes":p.stat().st_size} for p in dependencies],
        "labels_read":0,"ARFTR_outputs_read":0,"model_fits":0}


def establish_lock(output: Path, observations, shard_size: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    value = lock_value(observations, shard_size)
    write_json(output/"execution_lock.json", value)
    return value


@torch.inference_mode()
def encode(encoder: FrozenAbsoluteDinoCLS, tensors: list[torch.Tensor], device: str) -> np.ndarray:
    values=[]
    for start in range(0,len(tensors),8):
        batch=torch.stack(tensors[start:start+8]).to(device,non_blocking=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
            values.append(encoder(batch).float().cpu().numpy())
    result=np.concatenate(values).astype(np.float16)
    if result.shape!=(len(tensors),FEATURE_DIM) or not np.isfinite(result).all():
        raise RuntimeError("Malformed RGB-witness encoding")
    return result


def extract_shard(output: Path, observations, start: int, count: int, encoder, device: str,
                  source_map: dict, video_receipts: dict) -> dict:
    stop=min(len(observations),start+count)
    if not 0<=start<stop<=len(observations) or count>64:
        raise ValueError("Shard must contain1..64 consecutive rows")
    npz_path=output/f"shard-{start:04d}-{stop:04d}.npz"
    receipt_path=output/f"shard-{start:04d}-{stop:04d}.json"
    if npz_path.exists() != receipt_path.exists():
        raise RuntimeError("Partial cache shard exists")
    if npz_path.exists():
        receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("npz_sha256")!=file_sha256(npz_path):
            raise RuntimeError("Existing cache shard hash mismatch")
        return receipt
    n=stop-start
    features=np.zeros((n,2,FEATURE_DIM),np.float16)
    available=np.zeros((n,2),bool)
    geometry=np.zeros((n,6),np.float32)
    image_hash=np.full(n,"",dtype="U64")
    audits=[]; started=time.perf_counter()
    for local,observation in enumerate(observations[start:stop]):
        request=resolve_center_source(observation,source_map)
        rgb,audit=decode_exact_center(request,verified_video=video_receipts[observation.recording])
        row={"sample_id":observation.sample_id,"decode":audit,"views":{}}
        if rgb is not None:
            geom=make_body_crop_geometry(request.native_box,extent=1.25,source_size=request.video.source_size)
            raw=crop_raw_body(rgb,geom)
            masked=[apply_raw_body_mask(raw,geom,name) for name in SLOT_NAMES]
            encoded=encode(encoder,[preprocess_rgb(view) for view in masked],device)
            features[local]=encoded; available[local]=True
            geometry[local]=witness_geometry(geom.native_box,geom.crop_box,geom.source_size)
            image_hash[local]=audit["image_sha256"]
            row["views"]={name:{"masked_rgb_sha256":hashlib.sha256(np.ascontiguousarray(view).tobytes()).hexdigest(),
                                      "feature_sha256":hashlib.sha256(np.ascontiguousarray(encoded[i]).tobytes()).hexdigest()}
                                  for i,(name,view) in enumerate(zip(SLOT_NAMES,masked,strict=True))}
            row["geometry"]={"native_box":list(geom.native_box),"crop_box":list(geom.crop_box),
                             "fields":geometry[local].tolist()}
        audits.append(row)
    if device.startswith("cuda"): torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    ids=np.asarray([o.sample_id for o in observations[start:stop]])
    save_npz(npz_path,sample_ids=ids,slot_names=np.asarray(SLOT_NAMES),features=features,
             available=available,geometry=geometry,source_image_sha256=image_hash)
    receipt={"status":"RGB_WITNESS_SHARD_COMPLETE","start":start,"stop":stop,"rows":n,
        "npz_sha256":file_sha256(npz_path),"available_rows":int(available.all(1).sum()),
        "available_views":int(available.sum()),"elapsed_seconds":elapsed,"seconds_per_row":elapsed/n,
        "labels_read":0,"ARFTR_outputs_read":0,"model_fits":0,"row_audits":audits}
    write_json(receipt_path,receipt)
    return receipt


def summarize(output: Path, observations) -> dict:
    ranges=[]; seen=np.zeros(len(observations),np.int8); elapsed=views=rows=0
    for path in sorted(output.glob("shard-*.json")):
        r=json.loads(path.read_text(encoding="utf-8")); start,stop=r["start"],r["stop"]
        npz=output/f"shard-{start:04d}-{stop:04d}.npz"
        if r.get("npz_sha256")!=file_sha256(npz): raise RuntimeError("Cache shard hash changed")
        seen[start:stop]+=1; ranges.append([start,stop]); elapsed+=r["elapsed_seconds"]
        views+=r["available_views"]; rows+=r["available_rows"]
    result={"status":"RGB_WITNESS_CACHE_COMPLETE" if np.all(seen==1) else "RGB_WITNESS_CACHE_INCOMPLETE",
        "population":len(observations),"available_rows":rows,"available_views":views,
        "shards":len(ranges),"ranges":ranges,"summed_shard_seconds":elapsed,"labels_read":0,
        "ARFTR_outputs_read":0,"model_fits":0}
    write_json(output/"summary.json",result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--stage",choices=("lock","extract","summarize"),required=True)
    parser.add_argument("--start",type=int,default=0); parser.add_argument("--count",type=int,default=16)
    parser.add_argument("--all",action="store_true"); parser.add_argument("--shard-size",type=int,default=32)
    args=parser.parse_args(); output=args.output_dir.resolve(); observations=load_observations()
    establish_lock(output,observations,args.shard_size)
    if args.stage=="lock": result={"status":"RGB_WITNESS_EXTRACTION_LOCKED","output":str(output)}
    elif args.stage=="summarize": result=summarize(output,observations)
    else:
        source_map=load_verified_source_map(ROOT)
        video_receipts=json.loads(OLD_EXTRACTION.read_text(encoding="utf-8"))["video_receipts"]
        dino=json.loads(DINO_SUMMARY.read_text(encoding="utf-8"))["model"]
        device="cuda" if torch.cuda.is_available() else "cpu"
        loaded,_=load_dinov2_encoder(Path(dino["snapshot"]["path"]),device=device)
        encoder=FrozenAbsoluteDinoCLS(loaded.backbone).to(device).eval(); del loaded
        ranges=[(i,min(args.shard_size,len(observations)-i)) for i in range(0,len(observations),args.shard_size)] if args.all else [(args.start,args.count)]
        shards=[]
        for start,count in ranges:
            r=extract_shard(output,observations,start,count,encoder,device,source_map,video_receipts)
            shards.append({k:r[k] for k in ("start","stop","available_rows","elapsed_seconds")})
            print(json.dumps({"event":"rgb_witness_shard_complete",**shards[-1]}),flush=True)
        result={"status":"REQUESTED_RGB_WITNESS_SHARDS_COMPLETE","device":device,"shards":shards}
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
