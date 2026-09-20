"""Synthetic-only v2 episode GPU pilot: no datasets or optimizer steps."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from hac.actor_memory_base import file_sha256
from hac.actor_memory_training import seed_training
from hac.center_episode_memory import unweighted_boundary_loss
from hac.stable_center_episode_memory import NUMERICAL_VERSION, StableCenterEpisodeMemory


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source_protocol = root / "experiments/okutama_center_episode_protocol.json"
    protocol = json.loads(source_protocol.read_text(encoding="utf-8"))
    settings = protocol["matched_capacity"]
    batch = protocol["batch_size"]
    if batch != 128 or settings["input_dim"] != 3072 or settings["parameters_per_arm"] != 849059:
        raise RuntimeError("Numerical-only pilot cannot change original model capacity or batch")
    output = root / ".runs/research_20260908/center_episode_v2/resource_pilot.json"
    if output.exists():
        raise RuntimeError("Existing synthetic resource pilot is immutable")
    torch.set_num_threads(2)
    results = {}
    for arm in protocol["arms"]:
        seed_training(42)
        model = (
            StableCenterEpisodeMemory(
                settings["input_dim"],
                arm,
                width=settings["width"],
                layers=settings["layers"],
                heads=settings["heads"],
                hazard_width=settings["hazard_width"],
                dropout=settings["dropout"],
            )
            .cuda()
            .train()
        )
        initial_digest = state_digest(model)
        features = torch.randn(batch, 5, settings["input_dim"], device="cuda")
        probabilities = torch.rand(batch, 5, 3, device="cuda", dtype=torch.float64)
        probabilities /= probabilities.sum(-1, keepdim=True)
        times = torch.arange(-2, 3, device="cuda", dtype=torch.float32).expand(batch, -1)
        valid = torch.ones(batch, 5, device="cuda", dtype=torch.bool)
        valid[::3, 1] = False
        valid[::7] = torch.tensor([False, False, True, False, False], device="cuda")
        labels = torch.arange(batch, device="cuda") % 3
        boundary = (torch.arange(batch * 5, device="cuda").reshape(batch, 5) % 11 == 0).float()
        step_times, finite_gradients = [], True
        torch.cuda.reset_peak_memory_stats()
        for step in range(12):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            result = model(features, probabilities, times, valid)
            loss = F.nll_loss(result["probabilities"].float().log(), labels)
            loss += protocol["boundary_loss_weight"] * unweighted_boundary_loss(
                result, boundary, valid
            )
            loss += protocol["gate_square_penalty"] * result["gate"].square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.hazard_parameters(), 1.0, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(model.utility_parameters(), 1.0, error_if_nonfinite=True)
            torch.cuda.synchronize()
            if step >= 2:
                step_times.append(time.perf_counter() - tick)
            finite_gradients &= all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        bound = float((result["attention"] - result["survival"]).max())
        final_bound = float(
            (result["gate"][:, None] * result["attention"] - result["survival"]).max()
        )
        if arm == "expected_episode" and max(bound, final_bound) > 1e-6:
            raise RuntimeError("Unchanged 1e-6 episode coefficient gate failed")
        if not torch.isfinite(loss) or not finite_gradients:
            raise RuntimeError("Stable pilot loss or gradients are not finite")
        if state_digest(model) != initial_digest:
            raise RuntimeError("Synthetic pilot changed model state")
        simplex_error = float((result["probabilities"].sum(-1) - 1).abs().max())
        if simplex_error > 1e-12:
            raise RuntimeError("Stable pilot probability normalization failed")
        results[arm] = {
            "parameters": model.trainable_parameters,
            "batch_size": batch,
            "warmup_steps": 2,
            "timed_forward_backward_steps": len(step_times),
            "mean_forward_backward_seconds": sum(step_times) / len(step_times),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "finite_loss": True,
            "finite_gradients": True,
            "attention_bound_excess": bound,
            "final_neighbor_coefficient_bound_excess": final_bound,
            "maximum_simplex_error": simplex_error,
            "parameter_digest_before": initial_digest,
            "parameter_digest_after": state_digest(model),
            "state_unchanged": True,
            "neural_parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in model.parameters()}
            ),
            "attention_dtype": str(result["attention"].dtype),
        }
        del model, result, loss, features, probabilities
        torch.cuda.empty_cache()
    receipt = {
        "synthetic_only": True,
        "dataset_rows_read": 0,
        "optimizer_steps": 0,
        "classifier_fits": 0,
        "numerical_version": NUMERICAL_VERSION,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "source_v1_protocol_sha256": file_sha256(source_protocol),
        "pilot_source_sha256": file_sha256(Path(__file__)),
        "model_source_sha256": file_sha256(root / "src/hac/stable_center_episode_memory.py"),
        "inherited_v1_model_source_sha256": file_sha256(root / "src/hac/center_episode_memory.py"),
        "tests_sha256": file_sha256(root / "tests/test_stable_center_episode_memory.py"),
        "timing_caveat": "Two warmups excluded; other authorized jobs may overlap. This is feasibility, not a performance comparison.",
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
