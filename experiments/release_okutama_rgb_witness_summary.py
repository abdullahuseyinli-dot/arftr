"""Append-only native-JSON release for the locked RGB-witness result.

The original summarizer completed all calculations but failed before writing
because gate values were NumPy booleans.  This wrapper does not change or fit a
model: it captures the same locked calculation, converts scalar containers to
native JSON values, and writes a separately named release artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for value in (ROOT,ROOT/"src",ROOT/"experiments"):
    if str(value) not in sys.path:sys.path.insert(0,str(value))
from experiments import run_okutama_rgb_witness as locked


def native(value):
    if isinstance(value,np.ndarray):return native(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    return value


def sha256(path: Path) -> str:return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path,value: dict) -> None:
    payload=json.dumps(native(value),indent=2,sort_keys=True,allow_nan=False)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8")!=payload:raise RuntimeError(f"Immutable release differs: {path}")
    else:
        with path.open("x",encoding="utf-8") as f:f.write(payload)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,default=locked.DEFAULT_RUN);a=p.parse_args();run=a.run.resolve()
    original_write=locked.write_json
    captured={}
    locked.write_json=lambda path,value:captured.update(path=str(path),value=value)
    try:result=locked.summarize(run)
    finally:locked.write_json=original_write
    result=native(result)
    write_new(run/"summary_v2.json",result)
    receipt={"status":"RGB_WITNESS_SUMMARY_NATIVE_JSON_RELEASE_COMPLETE","scientific_recalculation_changed":False,
        "model_fits":0,"prediction_changes":0,"source_summarizer":str(Path(locked.__file__).resolve().relative_to(ROOT.resolve())).replace("\\","/"),
        "source_summarizer_sha256":sha256(Path(locked.__file__)),"execution_lock_sha256":sha256(run/"execution_lock.json"),
        "independent_audit_sha256":sha256(run/"independent_audit.json"),"summary_v2_sha256":sha256(run/"summary_v2.json"),
        "original_release_failure":"TypeError: Object of type bool is not JSON serializable (numpy.bool_ gate values)"}
    write_new(run/"summary_v2_receipt.json",receipt)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
