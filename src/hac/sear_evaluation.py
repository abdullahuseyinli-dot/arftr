"""Evaluation-only slot diagnostics and explicitly named reference comparisons.

No model, optimizer, fit, label-based routing, calibration, or fusion is defined
here. Slot diagnostics never receive action labels. Slot identities are not
aligned between independent seeds, so every slot-level result is per seed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    if not np.isfinite(values).all():
        raise ValueError("A reported diagnostic contains nonfinite values")
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def _boolean_mask(values, rows: int, name: str) -> np.ndarray:
    mask = np.asarray(values)
    if mask.shape != (rows,) or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be an explicitly boolean [{rows}] mask")
    return mask


def _entropy(shares: np.ndarray) -> np.ndarray:
    logarithm = np.zeros_like(shares)
    np.log(shares, out=logarithm, where=shares > 0)
    return -(shares * logarithm).sum(-1)


def _pair_summary(values: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> dict:
    selected = values[mask]
    weight = weights[mask]
    has_pair = mask.any(-1)
    count = mask.sum(-1)
    row_mean = np.divide(
        np.where(mask, values, 0).sum(-1), count, out=np.zeros(len(count)), where=count > 0
    )
    total_weight = float(weight.sum())
    return {
        "pairs": _distribution(selected),
        "rows_with_defined_pairs": int(has_pair.sum()),
        "per_row_pair_mean": _distribution(row_mean[has_pair]),
        "mass_product_weighted_mean": float(np.dot(selected, weight) / total_weight)
        if total_weight > 0
        else None,
    }


def slot_diagnostics(
    seed_masses: np.ndarray,
    *,
    seed_positions: np.ndarray | None = None,
    seed_descriptors: np.ndarray | None = None,
    local_valid: np.ndarray | None = None,
    seed_ids: Sequence[str | int] | None = None,
    active_share_threshold: float = 0.05,
    descriptor_cosine_threshold: float = 0.95,
    position_distance_threshold: float = 0.1,
    position_space: str = "caller-supplied coordinates; threshold uses those units",
    epsilon: float = 1e-12,
) -> dict:
    """Describe [seed,row,slot] evidence without action labels or seed alignment.

    Masses exclude the dustbin. Positive masses may have arbitrary total scale;
    concentration uses their within-row shares. Rows with zero total mass are
    counted separately, not assigned an artificial uniform allocation. Invalid
    local rows and zero-mass slot descriptors/positions may be NaN. Active means
    positive evidence with at least ``active_share_threshold`` of that row's mass.
    Pair similarities ignore zero-norm descriptors and explicitly count them.
    Thresholds are fixed descriptive conventions, not fitted decision rules.
    """
    masses = np.asarray(seed_masses, dtype=np.float64)
    if masses.ndim != 3 or min(masses.shape[0], masses.shape[2]) < 1:
        raise ValueError("seed_masses must have shape [seeds,rows,slots] with seeds,slots>=1")
    seeds, rows, slots = masses.shape
    valid = (
        np.ones(rows, bool)
        if local_valid is None
        else _boolean_mask(local_valid, rows, "local_valid")
    )
    if (
        not np.isfinite(epsilon)
        or epsilon <= 0
        or not np.isfinite(active_share_threshold)
        or not 0 < active_share_threshold <= 1
        or not np.isfinite(descriptor_cosine_threshold)
        or not -1 <= descriptor_cosine_threshold <= 1
        or not np.isfinite(position_distance_threshold)
        or position_distance_threshold < 0
    ):
        raise ValueError("Invalid diagnostic epsilon or threshold")
    if not np.isfinite(masses[:, valid]).all() or (masses[:, valid] < 0).any():
        raise ValueError("Available slot masses must be finite and nonnegative")
    names = [str(value) for value in (range(seeds) if seed_ids is None else seed_ids)]
    if len(names) != seeds or len(set(names)) != seeds:
        raise ValueError("seed_ids must uniquely identify every seed")
    positions = None if seed_positions is None else np.asarray(seed_positions, dtype=np.float64)
    descriptors = (
        None if seed_descriptors is None else np.asarray(seed_descriptors, dtype=np.float64)
    )
    if positions is not None and positions.shape != (*masses.shape, 2):
        raise ValueError("seed_positions must have shape [seeds,rows,slots,2]")
    if descriptors is not None and (
        descriptors.ndim != 4 or descriptors.shape[:3] != masses.shape or descriptors.shape[-1] < 1
    ):
        raise ValueError("seed_descriptors must have shape [seeds,rows,slots,dimension]")
    left, right = np.triu_indices(slots, k=1)
    reports = {}
    for seed, name in enumerate(names):
        mass = masses[seed, valid]
        total = mass.sum(-1)
        if not np.isfinite(total).all():
            raise ValueError("Available slot total mass overflowed")
        observed = total > epsilon
        shares = np.divide(mass, total[:, None], out=np.zeros_like(mass), where=observed[:, None])
        active = (shares >= active_share_threshold) & (mass > 0)
        entropy = _entropy(shares)
        concentration = np.square(shares).sum(-1)
        effective = np.divide(1, concentration, out=np.zeros_like(total), where=observed)
        aggregate = mass.sum(0)
        aggregate_total = float(aggregate.sum())
        if not np.isfinite(aggregate_total):
            raise ValueError("Batch slot mass overflowed")
        aggregate_shares = (
            aggregate / aggregate_total if aggregate_total > epsilon else np.zeros(slots)
        )
        batch_entropy = float(_entropy(aggregate_shares)) if aggregate_total > epsilon else None
        row_report = {
            "valid_local_rows": int(valid.sum()),
            "masked_local_rows": int((~valid).sum()),
            "positive_evidence_rows": int(observed.sum()),
            "zero_or_negligible_total_mass_rows": int((~observed).sum()),
            "total_matched_mass": _distribution(total),
            "active_slot_count": _distribution(active.sum(-1)),
            "positive_rows_with_zero_active_slots": int((observed & (active.sum(-1) == 0)).sum()),
            "positive_rows_with_one_active_slot": int((observed & (active.sum(-1) == 1)).sum()),
            "within_row": {
                "maximum_mass_share": _distribution(shares[observed].max(-1)),
                "herfindahl_concentration": _distribution(concentration[observed]),
                "inverse_concentration_effective_slots": _distribution(effective[observed]),
                "entropy_nats": _distribution(entropy[observed]),
                "entropy_effective_slots": _distribution(np.exp(entropy[observed])),
            },
            "batch_utilization": {
                "mean_raw_mass_by_slot": (aggregate / len(mass)).tolist()
                if len(mass)
                else [None] * slots,
                "aggregate_mass_share_by_slot": aggregate_shares.tolist(),
                "slots_with_zero_aggregate_mass": int((aggregate == 0).sum()),
                "entropy_nats": batch_entropy,
                "entropy_effective_slots": float(np.exp(batch_entropy))
                if batch_entropy is not None
                else None,
                "maximum_mass_share": float(aggregate_shares.max())
                if aggregate_total > epsilon
                else None,
                "batch_minus_mean_within_row_entropy": batch_entropy
                - float(entropy[observed].mean())
                if batch_entropy is not None and observed.any()
                else None,
            },
        }
        pair_active = active[:, left] & active[:, right]
        pair_weight = shares[:, left] * shares[:, right]
        descriptor_pair_valid = None
        cosines = None
        if descriptors is not None:
            descriptor = descriptors[seed, valid]
            if not np.isfinite(descriptor[mass > 0]).all():
                raise ValueError("Positive-mass slot descriptors must be finite")
            safe = np.where((mass > 0)[..., None], descriptor, 0)
            norms = np.linalg.norm(safe, axis=-1)
            if not np.isfinite(norms).all():
                raise ValueError("Descriptor norm overflowed")
            defined = norms > epsilon
            normalized = np.divide(
                safe, norms[..., None], out=np.zeros_like(safe), where=defined[..., None]
            )
            cosines = np.clip((normalized[:, left] * normalized[:, right]).sum(-1), -1, 1)
            descriptor_pair_valid = pair_active & defined[:, left] & defined[:, right]
            redundant = descriptor_pair_valid & (cosines >= descriptor_cosine_threshold)
            row_report["descriptor_distinction"] = {
                **_pair_summary(cosines, descriptor_pair_valid, pair_weight),
                "active_zero_norm_slots": int((active & ~defined).sum()),
                "near_duplicate_pairs": int(redundant.sum()),
                "rows_with_near_duplicate_active_pair": int(redundant.any(-1).sum()),
            }
        if positions is not None:
            position = positions[seed, valid]
            if not np.isfinite(position[mass > 0]).all():
                raise ValueError("Positive-mass slot positions must be finite")
            safe = np.where((mass > 0)[..., None], position, 0)
            distances = np.linalg.norm(safe[:, left] - safe[:, right], axis=-1)
            nearby = pair_active & (distances <= position_distance_threshold)
            row_report["position_distinction"] = {
                **_pair_summary(distances, pair_active, pair_weight),
                "nearby_pairs": int(nearby.sum()),
                "rows_with_nearby_active_pair": int(nearby.any(-1).sum()),
            }
            if descriptor_pair_valid is not None:
                redundant = (
                    nearby & descriptor_pair_valid & (cosines >= descriptor_cosine_threshold)
                )
                row_report["joint_descriptor_position_redundancy"] = {
                    "defined_active_pairs": int(descriptor_pair_valid.sum()),
                    "near_duplicate_pairs": int(redundant.sum()),
                    "rows_with_near_duplicate_active_pair": int(redundant.any(-1).sum()),
                }
        reports[name] = row_report
    return {
        "evaluation_only": True,
        "action_labels_used": 0,
        "rows": rows,
        "seeds": seeds,
        "slots": slots,
        "thresholds": {
            "active_share": active_share_threshold,
            "descriptor_cosine": descriptor_cosine_threshold,
            "position_distance": position_distance_threshold,
            "mass_and_norm_epsilon": epsilon,
        },
        "position_space": position_space,
        "per_seed": reports,
        "interpretation": "Within-image concentration and across-batch utilization are different. Slot indices are not matched across seeds. Similar descriptors or nearby positions are descriptive redundancy, not proof of anatomical or semantic collapse. No diagnostic threshold is an inference-routing rule.",
    }


def _validate_probabilities(probabilities, rows: int, classes: int | None = None) -> np.ndarray:
    values = np.asarray(probabilities)
    if (
        values.ndim != 2
        or values.shape[0] != rows
        or values.shape[1] < 2
        or (classes is not None and values.shape[1] != classes)
    ):
        raise ValueError("Probabilities must have aligned shape [rows,classes>=2]")
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Probabilities must be finite nonnegative simplex vectors")
    return values


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Fixed-class metrics; empty strata are undefined, not synthetic zero scores."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    p = _validate_probabilities(probabilities, len(labels))
    classes = p.shape[1]
    if ((labels < 0) | (labels >= classes)).any():
        raise ValueError("Target labels fall outside probability class order")
    labels = labels.astype(np.int64, copy=False)
    matrix = np.bincount(classes * labels + p.argmax(1), minlength=classes**2).reshape(
        classes, classes
    )
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(classes), where=denominator > 0)
    return {
        "rows": len(labels),
        "macro_f1": float(f1.mean()) if len(labels) else None,
        "accuracy": float(matrix.trace() / len(labels)) if len(labels) else None,
        "nll": float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean())
        if len(labels)
        else None,
        "brier": float(np.square(p - np.eye(classes)[labels]).sum(1).mean())
        if len(labels)
        else None,
        "per_class_f1": f1.tolist() if len(labels) else [None] * classes,
        "confusion": matrix.tolist(),
    }


def reference_strata_summary(
    labels: np.ndarray,
    candidate: np.ndarray,
    references: Mapping[str, np.ndarray],
    strata: Mapping[str, np.ndarray],
    original_shared: np.ndarray,
    *,
    candidate_name: str = "candidate",
) -> dict:
    """Evaluate existing aligned predictions; no probabilities are fitted or fused.

    The caller must establish source/sample-ID/class-order identity before calling.
    Strata may overlap and are never silently converted to a cohort partition.
    ``original_shared`` remains the same historical error set for every reference.
    """
    labels = np.asarray(labels)
    classification_metrics(labels, candidate)  # Validate all rows, not just selected strata.
    p = np.asarray(candidate)
    classes, rows = p.shape[1], len(labels)
    shared = _boolean_mask(original_shared, rows, "original_shared")
    masks = {name: _boolean_mask(mask, rows, str(name)) for name, mask in strata.items()}
    if not references or not masks or not isinstance(candidate_name, str) or not candidate_name:
        raise ValueError("Named candidate, references and strata are required")
    if any(not isinstance(name, str) or not name for name in (*references, *masks)):
        raise ValueError("Reference and stratum names must be nonempty strings")
    predicted = p.argmax(1)
    correct = predicted == labels
    summaries = {}
    for name, probabilities in references.items():
        reference = _validate_probabilities(probabilities, rows, classes)
        reference_predicted = reference.argmax(1)
        reference_correct = reference_predicted == labels
        by_stratum = {}
        for stratum, mask in masks.items():
            rescued = correct & ~reference_correct & mask
            harmed = ~correct & reference_correct & mask
            both_wrong = ~correct & ~reference_correct & mask
            by_stratum[stratum] = {
                "rows": int(mask.sum()),
                "candidate_metrics": classification_metrics(labels[mask], p[mask]),
                "reference_metrics": classification_metrics(labels[mask], reference[mask]),
                "candidate_errors": int((~correct & mask).sum()),
                "reference_errors": int((~reference_correct & mask).sum()),
                "rescues": int(rescued.sum()),
                "harms": int(harmed.sum()),
                "net_corrections": int(rescued.sum() - harmed.sum()),
                "both_correct": int((correct & reference_correct & mask).sum()),
                "both_wrong": int(both_wrong.sum()),
                "correctness_flips": int(rescued.sum() + harmed.sum()),
                "prediction_flips": int(((predicted != reference_predicted) & mask).sum()),
                "wrong_to_different_wrong": int(
                    (both_wrong & (predicted != reference_predicted)).sum()
                ),
                "original_shared_rows": int((shared & mask).sum()),
                "original_shared_repairs": int((shared & correct & mask).sum()),
                "reference_original_shared_repairs": int((shared & reference_correct & mask).sum()),
                "original_shared_rescues_vs_reference": int((shared & rescued).sum()),
                "original_shared_harms_vs_reference": int((shared & harmed).sum()),
            }
        summaries[name] = {"strata": by_stratum}
    return {
        "evaluation_only": True,
        "fitting_or_routing_allowed": False,
        "candidate": candidate_name,
        "rows": rows,
        "classes": classes,
        "references": summaries,
        "definitions": "Rescue/harm uses each named reference. Original-shared repairs count candidate-correct rows in the caller's unchanged historical mask, not newly rescued rows versus every reference. Macro-F1 always uses the fixed full class order, including inside single-class strata. IDs/class order must be verified by the caller; masks may overlap.",
    }
