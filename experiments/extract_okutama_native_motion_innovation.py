"""Extract the fixed exact-native camera-compensated three-frame cache."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from experiments.extract_okutama_source_posture import load_observations
from hac.absolute_body_evidence import make_crop_geometry
from hac.actor_memory_base import canonical_hash, file_sha256
from hac.body_witness_data import load_verified_source_map, resolve_center_source
from hac.native_motion_data import (
    CROP_SIZE, FRAME_NAMES, OFFSETS, PHASE_NAMES, aligned_actor_crop,
    decode_exact_triplet, estimate_camera_translation, offset_request,
)
from hac.rgb_witness_data import witness_geometry

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"experiments/okutama_native_motion_innovation_protocol.json"
OLD_EXTRACTION=ROOT/".runs/research_20260913/body_witness_pilot_v1/extraction_receipt.json"
DEFAULT_OUTPUT=ROOT/".runs/research_20260920/native_motion_innovation_v1/cache"


def write_json(path:Path,value:dict)->None:
    payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8")!=payload: raise RuntimeError(f"Immutable artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(payload,encoding="utf-8")


def save_npz(path:Path,**arrays)->None:
    if path.exists(): raise RuntimeError(f"Refusing to overwrite cache shard: {path}")
    with path.open("xb") as stream: np.savez_compressed(stream,**arrays)


def establish_lock(output:Path,observations,shard_size:int)->dict:
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source=protocol["source"]
    if (tuple(source["native_offsets"])!=OFFSETS or source["crop_size"]!=CROP_SIZE
            or source["shard_size"]!=shard_size or source["camera_grid"]!=[320,180]):
        raise RuntimeError("Native-motion extraction protocol changed")
    dependencies=[PROTOCOL,Path(__file__),ROOT/"experiments/run_native_motion_extraction_worker.py",ROOT/"src/hac/native_motion_data.py",
                  ROOT/"tests/test_native_motion_innovation.py",OLD_EXTRACTION]
    value={"status":"NATIVE_MOTION_LABEL_BLIND_EXTRACTION_LOCK","rows":len(observations),
        "frame_names":list(FRAME_NAMES),"phase_names":list(PHASE_NAMES),"offsets":list(OFFSETS),
        "crop_size":CROP_SIZE,"shard_size":shard_size,
        "sample_ids_sha256":canonical_hash([o.sample_id for o in observations]),
        "dependencies":[{"path":str(p.relative_to(ROOT)).replace("\\","/"),"sha256":file_sha256(p),"bytes":p.stat().st_size} for p in dependencies],
        "labels_read":0,"ARFTR_outputs_read":0,"model_fits":0}
    output.mkdir(parents=True,exist_ok=True);write_json(output/"execution_lock.json",value);return value


def extract_shard(output:Path,observations,start:int,count:int,source_map:dict,video_receipts:dict)->dict:
    stop=min(len(observations),start+count)
    if not 0<=start<stop<=len(observations) or count>64: raise ValueError("Shard must hold1..64 rows")
    npz=output/f"shard-{start:04d}-{stop:04d}.npz";receipt_path=npz.with_suffix(".json")
    if npz.exists()!=receipt_path.exists(): raise RuntimeError("Partial native-motion shard")
    if npz.exists():
        receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("npz_sha256")!=file_sha256(npz): raise RuntimeError("Existing shard hash mismatch")
        return receipt
    n=stop-start;crops=np.zeros((n,3,64,64,3),np.uint8);frame_available=np.zeros((n,3),bool)
    phase_available=np.zeros((n,2),bool);geometry=np.zeros((n,6),np.float32);camera=np.zeros((n,2,3),np.float32)
    hashes=np.full((n,3),"",dtype="U64");audits=[];reasons=Counter();started=time.perf_counter()
    for local,observation in enumerate(observations[start:stop]):
        center=resolve_center_source(observation,source_map);requests=[offset_request(center,o) for o in OFFSETS]
        verified=video_receipts[observation.recording]
        images,frame_audits=decode_exact_triplet(requests,verified_video=verified)
        for slot,(image,audit) in enumerate(zip(images,frame_audits,strict=True)):
            frame_available[local,slot]=image is not None
            hashes[local,slot]=audit.get("image_sha256") or ""
            if image is None: reasons[audit.get("reason") or "unknown"]+=1
        row={"sample_id":observation.sample_id,"frames":frame_audits,"camera":[]}
        if center.native_box is not None:
            geom=make_crop_geometry(center.native_box,source_size=center.video.source_size,
                                    source_id="N",region_id="whole",whole_scale=1.5)
            geometry[local]=witness_geometry(center.native_box,geom.integer_box,center.video.source_size)
        if all(image is not None for image in images) and center.native_box is not None:
            camera[local,0],valid0=estimate_camera_translation(images[0],images[1],center.native_box,source_size=center.video.source_size)
            camera[local,1],valid1=estimate_camera_translation(images[1],images[2],center.native_box,source_size=center.video.source_size)
            phase_available[local]=[valid0,valid1]
            # The past->center translation is inverted to sample past in center coordinates;
            # center->future is applied to sample future in center coordinates.
            crops[local,0]=aligned_actor_crop(images[0],center.native_box,source_size=center.video.source_size,
                                               source_sampling_shift=-camera[local,0,:2])
            crops[local,1]=aligned_actor_crop(images[1],center.native_box,source_size=center.video.source_size)
            crops[local,2]=aligned_actor_crop(images[2],center.native_box,source_size=center.video.source_size,
                                               source_sampling_shift=camera[local,1,:2])
            row["camera"]=[{"translation_native":camera[local,i].tolist(),"valid":bool(phase_available[local,i])} for i in range(2)]
            if not valid0: reasons["camera_phase0_sanity_failed"]+=1
            if not valid1: reasons["camera_phase1_sanity_failed"]+=1
        else:
            reasons["triplet_incomplete"]+=1
        audits.append(row)
    elapsed=time.perf_counter()-started
    save_npz(npz,sample_ids=np.asarray([o.sample_id for o in observations[start:stop]]),
        frame_names=np.asarray(FRAME_NAMES),phase_names=np.asarray(PHASE_NAMES),crops=crops,
        frame_available=frame_available,phase_available=phase_available,geometry=geometry,camera=camera,
        source_image_sha256=hashes)
    receipt={"status":"NATIVE_MOTION_SHARD_COMPLETE","start":start,"stop":stop,"rows":n,
        "npz_sha256":file_sha256(npz),"complete_triplets":int(frame_available.all(1).sum()),
        "complete_phases":int(phase_available.sum()),"complete_rows":int(phase_available.all(1).sum()),
        "reasons":dict(reasons),"elapsed_seconds":elapsed,"seconds_per_row":elapsed/n,
        "labels_read":0,"ARFTR_outputs_read":0,"model_fits":0,"row_audits":audits}
    write_json(receipt_path,receipt);return receipt


def summarize(output:Path,observations)->dict:
    seen=np.zeros(len(observations),np.int8);elapsed=triplets=phases=rows=0;ranges=[];reasons=Counter()
    for path in sorted(output.glob("shard-*.json")):
        r=json.loads(path.read_text(encoding="utf-8"));start,stop=r["start"],r["stop"]
        if r.get("npz_sha256")!=file_sha256(output/f"shard-{start:04d}-{stop:04d}.npz"): raise RuntimeError("Shard changed")
        seen[start:stop]+=1;ranges.append([start,stop]);elapsed+=r["elapsed_seconds"]
        triplets+=r["complete_triplets"];phases+=r["complete_phases"];rows+=r["complete_rows"];reasons.update(r["reasons"])
    result={"status":"NATIVE_MOTION_CACHE_COMPLETE" if np.all(seen==1) else "NATIVE_MOTION_CACHE_INCOMPLETE",
        "population":len(observations),"complete_triplets":triplets,"complete_phases":phases,"complete_rows":rows,
        "shards":len(ranges),"ranges":ranges,"reasons":dict(reasons),"summed_shard_seconds":elapsed,
        "labels_read":0,"ARFTR_outputs_read":0,"model_fits":0}
    write_json(output/"summary.json",result);return result


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--stage",choices=("lock","extract","summarize"),required=True);parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--count",type=int,default=16);parser.add_argument("--all",action="store_true");parser.add_argument("--shard-size",type=int,default=32)
    args=parser.parse_args();output=args.output_dir.resolve();observations=load_observations();establish_lock(output,observations,args.shard_size)
    if args.stage=="lock":result={"status":"NATIVE_MOTION_EXTRACTION_LOCKED","output":str(output)}
    elif args.stage=="summarize":result=summarize(output,observations)
    else:
        source_map=load_verified_source_map(ROOT);video_receipts=json.loads(OLD_EXTRACTION.read_text(encoding="utf-8"))["video_receipts"]
        ranges=[(i,min(args.shard_size,len(observations)-i)) for i in range(0,len(observations),args.shard_size)] if args.all else [(args.start,args.count)]
        shards=[]
        for start,count in ranges:
            r=extract_shard(output,observations,start,count,source_map,video_receipts);shards.append({k:r[k] for k in ("start","stop","complete_rows","elapsed_seconds")})
            print(json.dumps({"event":"native_motion_shard_complete",**shards[-1]}),flush=True)
        result={"status":"REQUESTED_NATIVE_MOTION_SHARDS_COMPLETE","shards":shards}
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
