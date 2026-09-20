"""Fixed two-criterion policy for one RGB-witness candidate per center."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

FEATURE_NAMES = tuple([f"log_anchor_{i}" for i in range(3)] + [f"log_candidate_{i}" for i in range(3)] +
    ["pair_delta", "anchor_gap", "candidate_gap", "anchor_entropy", "candidate_entropy"] +
    [f"geometry_{i}" for i in range(6)] + ["available_upper", "available_lower"])


def policy_features(anchor: np.ndarray, candidate: np.ndarray, delta: np.ndarray,
                    geometry: np.ndarray, available: np.ndarray) -> np.ndarray:
    p, c = np.asarray(anchor, np.float64), np.asarray(candidate, np.float64)
    logp, logc = np.log(np.clip(p, 1e-12, 1)), np.log(np.clip(c, 1e-12, 1))
    ps, cs = np.sort(p, axis=1), np.sort(c, axis=1)
    out = np.column_stack((logp, logc, np.asarray(delta, np.float64), ps[:, -1]-ps[:, -2],
        cs[:, -1]-cs[:, -2], -(p*logp).sum(1), -(c*logc).sum(1),
        np.asarray(geometry, np.float64), np.asarray(available, np.float64)))
    if out.shape != (len(p), len(FEATURE_NAMES)) or not np.isfinite(out).all():
        raise ValueError("Malformed RGB-witness policy features")
    return out


def _multinomial_objective(parameters: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    w, b = parameters[:-3].reshape(3, x.shape[1]), parameters[-3:]
    logits = x @ w.T + b
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    loss = -logp[np.arange(len(y)), y].mean() + .025*np.sum(w*w) + .00005*np.sum(b*b)
    err = np.exp(logp); err[np.arange(len(y)), y] -= 1; err /= len(y)
    grad = np.r_[(err.T @ x + .05*w).ravel(), err.sum(0) + .0001*b]
    return float(loss), grad


@dataclass(frozen=True)
class WitnessPolicy:
    mean: np.ndarray; scale: np.ndarray
    utility_coef: np.ndarray; utility_intercept: np.ndarray
    nll_coef: np.ndarray; nll_intercept: float

    def scores(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = (np.asarray(features, np.float64)-self.mean)/self.scale
        logits = x @ self.utility_coef.T + self.utility_intercept
        prob = np.exp(logits-logsumexp(logits, axis=1, keepdims=True))
        return prob[:, 0]-2*prob[:, 1], x @ self.nll_coef + self.nll_intercept


def fit_policy(features: np.ndarray, anchor: np.ndarray, candidate: np.ndarray, labels: np.ndarray,
               scenarios: np.ndarray) -> tuple[WitnessPolicy | None, dict]:
    x = np.asarray(features, np.float64); y = np.asarray(labels)
    old, new = np.asarray(anchor).argmax(1), np.asarray(candidate).argmax(1)
    crossing = new != old
    rescue = crossing & (old != y) & (new == y)
    harm = crossing & (old == y) & (new != y)
    both = crossing & ~rescue & ~harm
    receipt = {"event_rows": int(crossing.sum()), "rescues": int(rescue.sum()), "harms": int(harm.sum()),
        "both_wrong": int(both.sum()), "scenarios": len(set(np.asarray(scenarios)[crossing].tolist()))}
    if (receipt["event_rows"] < 50 or receipt["rescues"] < 20 or receipt["harms"] < 20 or receipt["scenarios"] < 4):
        return None, {**receipt, "status": "RGB_WITNESS_POLICY_SUPPORT_NO_FIT_RETAIN"}
    xe = x[crossing]
    mean, scale = xe.mean(0), xe.std(0); scale[scale == 0] = 1
    z = (xe-mean)/scale
    outcome = np.where(rescue[crossing], 0, np.where(harm[crossing], 1, 2)).astype(np.int64)
    initial = np.zeros(3*len(FEATURE_NAMES)+3)
    result = minimize(_multinomial_objective, initial, args=(z, outcome), jac=True, method="L-BFGS-B",
        options={"maxiter":1000,"gtol":1e-7,"ftol":0.0,"maxls":50,"maxcor":108})
    objective, gradient = _multinomial_objective(result.x, z, outcome)
    glinf = float(np.abs(gradient).max())
    if not (result.success and np.isfinite(objective) and glinf <= 1e-7):
        raise RuntimeError(f"RGB-witness multinomial failed: {result.message}; gradient={glinf}")
    w, b = result.x[:-3].reshape(3, -1), result.x[-3:]
    anchor_nll = -np.log(np.clip(np.asarray(anchor)[np.arange(len(y)), y], 1e-12, 1))
    candidate_nll = -np.log(np.clip(np.asarray(candidate)[np.arange(len(y)), y], 1e-12, 1))
    target = (anchor_nll-candidate_nll)[crossing]
    zmean, tmean = z.mean(0), target.mean(); centered = z-zmean
    coef = np.linalg.solve(centered.T@centered + 10*np.eye(z.shape[1]), centered.T@(target-tmean))
    intercept = float(tmean-zmean@coef)
    policy = WitnessPolicy(mean, scale, w, b, coef, intercept)
    return policy, {**receipt, "status":"RGB_WITNESS_POLICY_FIT_COMPLETE", "objective":objective,
        "gradient_linf":glinf,"iterations":int(result.nit),"nll_ridge_alpha":10.0}


def route(anchor: np.ndarray, candidate: np.ndarray, features: np.ndarray,
          policy: WitnessPolicy | None) -> tuple[np.ndarray, np.ndarray, dict]:
    original = np.asarray(anchor)
    output = original.copy(); choices = np.zeros(len(original), dtype=np.int8)
    if policy is None:
        return output, choices, {"interventions":0,"reason":"support_no_fit"}
    utility, nll = policy.scores(features)
    crossing = np.asarray(candidate).argmax(1) != original.argmax(1)
    active = crossing & np.isfinite(utility) & np.isfinite(nll) & (utility > 0) & (nll > 0)
    output[active] = np.asarray(candidate)[active]; choices[active] = 1
    return output, choices, {"interventions":int(active.sum())}
