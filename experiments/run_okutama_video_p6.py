"""Replay the locked Orthogonal Cross-View Consensus (OCVC) fusion."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_video_p2a as p2
import run_okutama_video_probe as p1

from hac.video_consensus import ARM_COMPONENTS, ARMS, derive_consensus


@dataclass(frozen=True)
class Evidence:
    sample_ids: np.ndarray
    scenarios: np.ndarray
    labels: np.ndarray
    folds: np.ndarray
    long_valid: np.ndarray
    baseline: np.ndarray
    p3_best: np.ndarray
    p5_best: np.ndarray
    sources: dict[str, dict[str, np.ndarray]]


def _load_lock(root: Path, path: Path) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    locker = importlib.import_module("tools.lock_okutama_video_p6")
    return locker.validate_lock(root, path.resolve())


def _read_phase(root: Path, entry: dict[str, Any]) -> dict[str, np.ndarray]:
    path = p1._checked_path(root, entry["oof"]["path"])
    if p1.sha256_file(path) != entry["oof"]["sha256"] or path.stat().st_size != int(
        entry["oof"]["size_bytes"]
    ):
        raise RuntimeError("P6 OOF evidence receipt changed")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(entry["required_arrays"]) - set(archive.files))
        if missing:
            raise RuntimeError("P6 evidence arrays missing: " + ", ".join(missing))
        return {name: archive[name].copy() for name in entry["required_arrays"]}


def load_evidence(root: Path, lock: dict[str, Any]) -> Evidence:
    p3 = _read_phase(root, lock["evidence"]["p3"])
    p5 = _read_phase(root, lock["evidence"]["p5"])
    for name in ("sample_ids", "recording_ids", "labels", "folds", "long_valid"):
        if not np.array_equal(p3[name], p5[name]):
            raise RuntimeError(f"P3/P5 OOF identity changed: {name}")
    rows = int(lock["protocol"]["primary_rows"])
    if (
        len(p3["labels"]) != rows
        or len(np.unique(p3["sample_ids"])) != rows
        or len(np.unique(p3["recording_ids"])) != 11
        or set(np.unique(p3["folds"]).tolist()) != set(range(5))
        or set(np.unique(p3["labels"]).tolist()) != {0, 1, 2}
        or p3["long_valid"].dtype != np.bool_
    ):
        raise RuntimeError("P6 population contract changed")
    source_names = {phase: {name for item_phase, name in sum(ARM_COMPONENTS.values(), ()) if item_phase == phase} for phase in ("p3", "p5")}
    sources = {
        phase: {name: values[name] for name in source_names[phase]}
        for phase, values in (("p3", p3), ("p5", p5))
    }
    for values in (*sources["p3"].values(), *sources["p5"].values()):
        p1.validate_probability_array(values, rows)
    p1.validate_probability_array(p3["baseline_probabilities"], rows)
    p1.validate_probability_array(p3["dual_scale_factorized"], rows)
    p1.validate_probability_array(p5["spatial_contrast_factorized"], rows)
    return Evidence(
        sample_ids=p3["sample_ids"],
        scenarios=p3["recording_ids"],
        labels=p3["labels"],
        folds=p3["folds"],
        long_valid=p3["long_valid"],
        baseline=p3["baseline_probabilities"],
        p3_best=p3["dual_scale_factorized"],
        p5_best=p5["spatial_contrast_factorized"],
        sources=sources,
    )


def _error_correlations(
    labels: np.ndarray, probabilities: dict[str, np.ndarray]
) -> dict[str, float]:
    names = sorted(probabilities)
    output: dict[str, float] = {}
    for left_index, left in enumerate(names):
        left_error = (probabilities[left].argmax(axis=1) != labels).astype(np.float64)
        for right in names[left_index + 1 :]:
            right_error = (probabilities[right].argmax(axis=1) != labels).astype(np.float64)
            if left_error.std() == 0 or right_error.std() == 0:
                correlation = float(np.array_equal(left_error, right_error))
            else:
                correlation = float(np.corrcoef(left_error, right_error)[0, 1])
            output[f"{left}|{right}"] = correlation
    return output


def summarize(
    evidence: Evidence,
    probabilities: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    references = {
        **probabilities,
        "baseline": evidence.baseline,
        "p3_best": evidence.p3_best,
        "p5_best": evidence.p5_best,
    }
    models = {
        arm: {
            "metrics": p1.metrics(evidence.labels, values),
            "component_count": len(ARM_COMPONENTS[arm]),
            "model_fits": 0,
            "per_scenario": {
                str(group): p1.metrics(evidence.labels[mask], values[mask])
                for group in np.unique(evidence.scenarios)
                for mask in [evidence.scenarios == group]
            },
            "diagnostic_strata": {
                name: p1.metrics(evidence.labels[mask], values[mask])
                for name, mask in (
                    ("long_valid", evidence.long_valid),
                    ("long_invalid_short_anchor", ~evidence.long_valid),
                )
            },
        }
        for arm, values in probabilities.items()
    }
    statistics = protocol["statistics"]
    contrasts: dict[str, Any] = {}
    for item in statistics["comparisons"]:
        candidate, reference = item["candidate"], item["reference"]
        name = f"{candidate}_vs_{reference}"
        contrasts[name] = p1.paired_statistics(
            evidence.labels,
            probabilities[candidate],
            references[reference],
            evidence.scenarios,
            bootstrap_resamples=int(statistics["bootstrap_resamples"]),
            bootstrap_seed=int(statistics["bootstrap_seed"]),
        )
        contrasts[name]["diagnostic_strata"] = {
            stratum: p2.subgroup_statistics(
                evidence.labels,
                probabilities[candidate],
                references[reference],
                evidence.scenarios,
                mask,
                bootstrap_resamples=int(statistics["bootstrap_resamples"]),
                bootstrap_seed=int(statistics["bootstrap_seed"]),
            )
            for stratum, mask in (
                ("long_valid", evidence.long_valid),
                ("long_invalid_short_anchor", ~evidence.long_valid),
            )
        }
    family = sorted(contrasts)
    adjusted = p1.holm_adjust(
        {name: contrasts[name]["one_sided_exact_swap_pvalue"] for name in family}
    )
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_one_sided_pvalue"] = value
    components = {
        f"{phase}.{name}": values
        for phase, phase_values in evidence.sources.items()
        for name, values in phase_values.items()
    }
    primary = protocol["architecture"]["primary_arm"]
    return {
        "status": "OKUTAMA_VIDEO_P6_ADAPTIVE_REPLAY_COMPLETE",
        "method": protocol["method_name"],
        "rows": len(evidence.labels),
        "scenarios": len(np.unique(evidence.scenarios)),
        "baseline": p1.metrics(evidence.labels, evidence.baseline),
        "p3_best_reference": p1.metrics(evidence.labels, evidence.p3_best),
        "p5_best_reference": p1.metrics(evidence.labels, evidence.p5_best),
        "component_metrics": {
            name: p1.metrics(evidence.labels, values) for name, values in components.items()
        },
        "models": models,
        "contrasts": contrasts,
        "holm_family": family,
        "component_error_correlations": _error_correlations(evidence.labels, components),
        "primary_arm": primary,
        "primary_macro_f1": models[primary]["metrics"]["macro_f1"],
        "breakthrough_target": float(statistics["breakthrough_target"]),
        "target_crossed_by": [
            arm
            for arm, result in models.items()
            if result["metrics"]["macro_f1"] >= float(statistics["breakthrough_target"])
        ],
        "adaptation_disclosure": protocol["adaptation_disclosure"],
        "scope": "adaptive_same_development_scenarios_numerical_result_known_before_replay",
        "model_fits": 0,
        "labels_used_in_fusion": False,
        "raw_images_read": 0,
        "protected_rows_read": 0,
    }


def _validate_protocol(protocol: dict[str, Any]) -> None:
    locker = importlib.import_module("tools.lock_okutama_video_p6")
    architecture = protocol["architecture"]
    expected_arms = {
        arm: [f"{phase}.{name}" for phase, name in ARM_COMPONENTS[arm]] for arm in ARMS
    }
    if (
        architecture["arms"] != expected_arms
        or architecture["primary_arm"] != ARMS[0]
        or architecture["learned_parameters"] != 0
        or protocol["statistics"]["comparisons"] != locker.expected_comparisons()
        or int(protocol["statistics"]["bootstrap_resamples"]) != 10000
        or int(protocol["execution_budget"]["model_fits"]) != 0
    ):
        raise RuntimeError("P6 deterministic replay contract changed")


def _validate_output(
    output: Path, summary: dict[str, Any], request_sha: str, *, published: bool = True
) -> None:
    if (
        summary.get("status") != "OKUTAMA_VIDEO_P6_ADAPTIVE_REPLAY_COMPLETE"
        or summary.get("request_sha256") != request_sha
    ):
        raise RuntimeError("Retained P6 summary belongs to another request")
    expected = {"oof_probabilities.npz", "metrics.csv", "paired_statistics.json"}
    if set(summary.get("artifacts", {})) != expected:
        raise RuntimeError("Retained P6 aggregate inventory changed")
    for name, digest in summary["artifacts"].items():
        if p1.sha256_file(output / name) != digest:
            raise RuntimeError("Retained P6 artifact bytes changed")
    observed = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    markers = {"request.json", "summary.json"} if published else {"request.json"}
    if observed != expected | markers:
        raise RuntimeError("Unexpected files in completed P6 output")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root / ".runs" or not output.is_relative_to(root / ".runs"):
        raise RuntimeError("P6 output must use a dedicated directory below .runs")
    lock = _load_lock(root, args.protocol_lock)
    protocol = lock["protocol"]
    _validate_protocol(protocol)
    evidence = load_evidence(root, lock)
    request = {
        "status": "P6_REQUEST_BEFORE_DETERMINISTIC_FUSION",
        "lock_sha256": p1.sha256_file(args.protocol_lock.resolve()),
        "protocol": protocol,
        "sample_ids_sha256": p1.canonical_digest(evidence.sample_ids.tolist()),
        "source_sha256": p1.sha256_file(Path(__file__)),
        "output": str(output),
        "model_fits": 0,
    }
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("P6 output belongs to another locked request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty P6 output lacks its matching request")
    else:
        p1.atomic_json(request_path, request)
    request_sha = p1.canonical_digest(request)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        _validate_output(output, summary, request_sha)
        return summary
    if any(
        (output / name).exists()
        for name in ("oof_probabilities.npz", "metrics.csv", "paired_statistics.json")
    ):
        raise RuntimeError("Incomplete P6 aggregate evidence retained; refusing overwrite")
    probabilities = derive_consensus(evidence.sources)
    for values in probabilities.values():
        p1.validate_probability_array(values, len(evidence.labels))
    _load_lock(root, args.protocol_lock)
    summary = summarize(evidence, probabilities, protocol)
    p1.atomic_bytes(
        output / "oof_probabilities.npz",
        p1.npz_bytes(
            sample_ids=evidence.sample_ids,
            recording_ids=evidence.scenarios,
            labels=evidence.labels,
            folds=evidence.folds,
            long_valid=evidence.long_valid,
            baseline_probabilities=evidence.baseline,
            p3_best_probabilities=evidence.p3_best,
            p5_best_probabilities=evidence.p5_best,
            **probabilities,
        ),
    )
    table = io.StringIO(newline="")
    writer = csv.DictWriter(
        table,
        fieldnames=["model", "macro_f1", "accuracy", "nll", "brier", "model_fits"],
    )
    writer.writeheader()
    for arm, result in summary["models"].items():
        writer.writerow(
            {
                "model": arm,
                "model_fits": 0,
                **{
                    key: result["metrics"][key]
                    for key in ("macro_f1", "accuracy", "nll", "brier")
                },
            }
        )
    p1.atomic_bytes(output / "metrics.csv", table.getvalue().encode())
    p1.atomic_json(
        output / "paired_statistics.json",
        {"contrasts": summary["contrasts"], "holm_family": summary["holm_family"]},
    )
    inventory = [
        output / name
        for name in ("oof_probabilities.npz", "metrics.csv", "paired_statistics.json")
    ]
    summary.update(
        request_sha256=request_sha,
        artifacts={path.name: p1.sha256_file(path) for path in inventory},
    )
    _validate_output(output, summary, request_sha, published=False)
    p1.atomic_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = run(parser.parse_args())
    concise = {
        "status": result["status"],
        "rows": result.get("rows"),
        "models": {arm: values["metrics"] for arm, values in result.get("models", {}).items()},
        "target_crossed_by": result.get("target_crossed_by", []),
    }
    print(json.dumps(concise, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
