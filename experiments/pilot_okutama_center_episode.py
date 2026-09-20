"""Synthetic-only GPU feasibility check for the post-M4 episode experiment."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from hac.actor_memory_base import file_sha256
from hac.actor_memory_training import seed_training
from hac.center_episode_memory import CenterEpisodeMemory, unweighted_boundary_loss


def main():
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "experiments/okutama_center_episode_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    settings = protocol["matched_capacity"]
    batch = protocol["batch_size"]
    output = root / ".runs/research_20260908/center_episode_v1/resource_pilot.json"
    if output.exists():
        raise RuntimeError("Do not overwrite the resource pilot")
    torch.set_num_threads(2)
    results = {}
    for arm in protocol["arms"]:
        seed_training(42)
        model = CenterEpisodeMemory(
            settings["input_dim"],
            arm,
            width=settings["width"],
            layers=settings["layers"],
            heads=settings["heads"],
            hazard_width=settings["hazard_width"],
            dropout=settings["dropout"],
        ).cuda()
        features = torch.randn(batch, 5, settings["input_dim"], device="cuda")
        probabilities = torch.rand(batch, 5, 3, dtype=torch.float64, device="cuda")
        probabilities /= probabilities.sum(-1, keepdim=True)
        times = torch.arange(-2, 3, dtype=torch.float32, device="cuda").expand(batch, -1)
        valid = torch.ones(batch, 5, dtype=torch.bool, device="cuda")
        valid[::3, 1] = False
        labels = torch.arange(batch, device="cuda") % 3
        boundary = (torch.arange(batch * 5, device="cuda").reshape(batch, 5) % 11 == 0).float()
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            model.zero_grad(set_to_none=True)
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
        bound_excess = float((result["attention"] - result["survival"]).max())
        if arm == "expected_episode" and bound_excess > 1e-6:
            raise RuntimeError("Episode probability influence bound failed")
        if any(not torch.equal(value, initial[name]) for name, value in model.state_dict().items()):
            raise RuntimeError("Synthetic pilot changed model parameters")
        results[arm] = {
            "parameters": model.trainable_parameters,
            "batch_size": batch,
            "mean_forward_backward_seconds": (time.perf_counter() - start) / 10,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "finite_loss": bool(torch.isfinite(loss)),
            "attention_bound_excess": bound_excess,
        }
        del model, result, loss, features, initial
        torch.cuda.empty_cache()
    receipt = {
        "synthetic_only": True,
        "dataset_rows_read": 0,
        "optimizer_steps": 0,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "concurrent_work": "Ordered local-motion training and CPU native-video crop construction",
        "protocol_at_pilot_sha256": file_sha256(protocol_path),
        "pilot_source_sha256": file_sha256(Path(__file__)),
        "model_source_sha256": file_sha256(root / "src/hac/center_episode_memory.py"),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
