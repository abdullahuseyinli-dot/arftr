"""Run one disjoint resumable native-motion extraction lane."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--worker",type=int,required=True)
    parser.add_argument("--workers",type=int,default=4);parser.add_argument("--rows",type=int,default=4977)
    parser.add_argument("--shard-size",type=int,default=32);parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if not 0<=args.worker<args.workers: raise ValueError("Invalid disjoint worker index")
    script=ROOT/"experiments/extract_okutama_native_motion_innovation.py"
    for start in range(args.worker*args.shard_size,args.rows,args.workers*args.shard_size):
        count=min(args.shard_size,args.rows-start)
        subprocess.run([sys.executable,str(script),"--output-dir",str(args.output_dir),"--stage","extract",
                        "--start",str(start),"--count",str(count),"--shard-size",str(args.shard_size)],
                       cwd=ROOT,check=True)


if __name__=="__main__":main()
