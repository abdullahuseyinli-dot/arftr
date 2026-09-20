"""Synthetic-only resource audit for a future matched local-detail comparison.

This program never opens a dataset, labels, historical predictions, or model
checkpoints. It makes no optimizer step. Running it cannot start classification
training or modify either active memory/ordered experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from hac.fine_local_motion import (
    FineLocalMotion,
    expand_coarse_tokens,
    group_fine_cells,
    ungroup_fine_cells,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_fine_motion_protocol.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def synthetic_pair(batch: int, seed: int, device: str) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(batch, 8, 9, 768, generator=generator)
    repeated = expand_coarse_tokens(coarse)
    noise = torch.randn(batch, 8, 9, 16, 768, generator=generator) * 0.1
    noise = noise - noise.mean(dim=-2, keepdim=True)
    fine = ungroup_fine_cells(group_fine_cells(repeated) + noise)
    times = torch.arange(8).float()[None].expand(batch, -1) * (8 / 30)
    return coarse.to(device), repeated.to(device), fine.to(device), times.to(device)


def one_configuration(protocol, source_hashes, output, batch, rank, match, device):
    request = {
        "batch_size": batch,
        "rank": rank,
        "matching": match,
        "device": device,
        "source_sha256": source_hashes,
        "torch_version": str(torch.__version__),
        "seed": protocol["synthetic_resource_pilot"]["seed"],
    }
    path = output / f"rank{rank}_{match}_batch{batch}.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["request"] != request or not result["parameters_unchanged"]:
            raise RuntimeError("Retained synthetic pilot request changed; use a fresh output path")
        return result
    torch.manual_seed(request["seed"])
    coarse, repeated, fine, times = synthetic_pair(batch, request["seed"], device)
    config = protocol["model"]
    model = FineLocalMotion(
        rank=rank,
        width=config["width"],
        layers=config["layers"],
        heads=config["heads"],
        dropout=config["dropout"],
        match_mode=match,
        parameter_limit=config["max_parameters"],
    ).to(device)
    model.train()
    before = parameter_hash(model)
    audit = protocol["synthetic_resource_pilot"]
    conditions = {}
    for name, values in (("repeated_coarse", repeated), ("genuine_synthetic_detail", fine)):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        timings = []
        for iteration in range(audit["warmup_iterations"] + audit["measured_iterations"]):
            model.zero_grad(set_to_none=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model(values, times, coarse_reference=coarse)
            loss = result["logits"].square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Synthetic fine-head loss is nonfinite")
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("Synthetic fine-head gradient is nonfinite")
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if iteration >= audit["warmup_iterations"]:
                timings.append(elapsed)
        peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
        reserved = torch.cuda.max_memory_reserved() if device.startswith("cuda") else None
        conditions[name] = {
            "median_forward_backward_seconds": statistics.median(timings),
            "measured_seconds": timings,
            "peak_allocated_bytes": peak,
            "peak_reserved_bytes": reserved,
            "all_gradients_finite": True,
        }
        if peak is not None and peak > audit["maximum_allocated_bytes"]:
            raise RuntimeError(
                f"Synthetic fine pilot exceeded the2GiB allocation gate at batch{batch}/rank{rank}/{match}"
            )
    after = parameter_hash(model)
    if before != after:
        raise RuntimeError("Synthetic resource measurement unexpectedly changed model state")
    result = {
        "status": "SYNTHETIC_FINE_DETAIL_RESOURCE_COMPLETE",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "request": request,
        "parameter_count": model.trainable_parameters,
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "parameters_unchanged": True,
        "conditions": conditions,
        "optimizer_steps": 0,
        "dataset_reads": 0,
        "labels_read": 0,
        "historical_predictions_read": 0,
        "classification_scores_produced": False,
    }
    write_new_json(path, result)
    print(
        json.dumps(
            {
                "completed": path.name,
                "parameters": model.trainable_parameters,
                "conditions": conditions,
            }
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".runs/research_20260908/fine_motion_synthetic_pilot",
    )
    parser.add_argument("--batches", default="8,16,32")
    parser.add_argument("--ranks", default="32,64")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    batches = [int(value) for value in args.batches.split(",")]
    ranks = [int(value) for value in args.ranks.split(",")]
    if len(set(batches)) != len(batches) or not set(batches).issubset(
        protocol["synthetic_resource_pilot"]["batches"]
    ):
        raise ValueError("Only unique declared resource batch sizes are permitted")
    if len(set(ranks)) != len(ranks) or not set(ranks).issubset(
        protocol["model"]["ranks_for_resource_audit"]
    ):
        raise ValueError("Only unique declared resource ranks are permitted")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sources = {
        "protocol": file_hash(PROTOCOL),
        "module": file_hash(ROOT / "src/hac/fine_local_motion.py"),
        "pilot": file_hash(Path(__file__)),
        "tests": file_hash(ROOT / "tests/test_fine_local_motion.py"),
    }
    results = []
    for batch in batches:
        for rank in ranks:
            for match in protocol["synthetic_resource_pilot"]["matching_modes"]:
                results.append(
                    one_configuration(
                        protocol, sources, args.output_dir, batch, rank, match, args.device
                    )
                )
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
    summary = {
        "status": "FINE_DETAIL_SYNTHETIC_PILOT_COMPLETE_NO_CLASSIFIER_FITS",
        "source_sha256": sources,
        "configurations": results,
        "dataset_reads": 0,
        "labels_read": 0,
        "optimizer_steps": 0,
        "next_phase": "Review with memory/ordered outcomes before declaring a separate classifier trial",
    }
    summary_path = (
        args.output_dir
        / f"summary_{args.device.replace(':', '_')}_{'-'.join(map(str, batches))}_{'-'.join(map(str, ranks))}.json"
    )
    if summary_path.exists():
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise RuntimeError("Existing summary differs; preserve it and choose a new path")
    else:
        write_new_json(summary_path, summary)
    print(
        json.dumps(
            {"summary": str(summary_path), "configurations": len(results), "optimizer_steps": 0}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
