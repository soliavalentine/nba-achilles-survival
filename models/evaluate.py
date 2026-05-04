"""
Survival model evaluation metrics.

Implements:
  - C-index (Harrell's concordance, competing-risks aware)
  - Brier Score (time-dependent, IPCW-weighted)
  - D-calibration (distributional calibration across deciles)
  - Integrated Brier Score (IBS)

All metrics follow the competing-risks formulation where cause=1 is
Achilles rupture and cause=2 is any other career-ending event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

try:
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


# ---------------------------------------------------------------------------
# C-index (competing risks)
# ---------------------------------------------------------------------------

def c_index_competing_risks(
    H: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    cause: int = 1,
    time_horizon: int | None = None,
) -> float:
    """
    Harrell's C-index adapted for competing risks.

    Compares predicted CIF at time_horizon (or observed time) between all
    valid comparable pairs where one subject experienced the cause of interest
    before the other's observed time.

    Args:
        H:            (n, n_causes, n_time_bins) joint PMF from DeepHit.forward().
        t:            (n,) observed time bins.
        e:            (n,) event indicators (0=censored, 1=cause_1, 2=cause_2, …).
        cause:        Which cause index to evaluate (1-indexed).
        time_horizon: Evaluate CIF at this time bin. Defaults to max(t).

    Returns:
        C-index scalar in [0, 1].
    """
    n = len(t)
    cause_idx = cause - 1  # 0-indexed into H

    if time_horizon is None:
        time_horizon = int(t.max())
    time_horizon = min(time_horizon, H.shape[2] - 1)

    # CIF at time_horizon for each subject
    cif = H[:, cause_idx, :time_horizon + 1].sum(axis=1)

    concordant = 0
    comparable = 0

    for i in range(n):
        for j in range(n):
            # Subject i had cause=1 event at t[i]; subject j either had event
            # at t[j] > t[i] or was censored at t[j] >= t[i]
            if e[i] == cause and (e[j] != cause or t[j] > t[i]):
                if t[j] > t[i]:
                    comparable += 1
                    if cif[i] > cif[j]:
                        concordant += 1
                    elif cif[i] == cif[j]:
                        concordant += 0.5

    return concordant / comparable if comparable > 0 else 0.5


# ---------------------------------------------------------------------------
# IPCW Brier Score
# ---------------------------------------------------------------------------

def _kaplan_meier_censoring(t: np.ndarray, e: np.ndarray) -> callable:
    """
    Estimate the censoring distribution G(t) = P(C > t) via Kaplan-Meier
    on the reverse event indicator (censoring times).
    """
    # Censored observations are "events" for the censoring KM
    censored = (e == 0).astype(int)
    unique_times = np.sort(np.unique(t))
    n = len(t)
    G = {}
    surv = 1.0
    for time in unique_times:
        at_risk = np.sum(t >= time)
        events = np.sum((t == time) & (censored == 1))
        if at_risk > 0:
            surv *= 1 - events / at_risk
        G[time] = max(surv, 1e-6)

    def G_func(query_t: float) -> float:
        valid = [v for k, v in G.items() if k <= query_t]
        return valid[-1] if valid else 1.0

    return G_func


def brier_score(
    H: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    eval_times: np.ndarray | None = None,
    cause: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-dependent Brier Score with IPCW weighting.

    Returns:
        (eval_times, brier_scores)  — one score per evaluation time point.
    """
    n, n_causes, n_bins = H.shape
    cause_idx = cause - 1

    if eval_times is None:
        eval_times = np.percentile(t, np.linspace(10, 90, 9)).astype(int)
        eval_times = np.unique(np.clip(eval_times, 0, n_bins - 1))

    G_func = _kaplan_meier_censoring(t, e)
    scores = []

    for tau in eval_times:
        cif_tau = H[:, cause_idx, : tau + 1].sum(axis=1)
        bs = 0.0
        n_valid = 0

        for i in range(n):
            G_t = G_func(t[i])
            G_tau = G_func(tau)

            if t[i] <= tau and e[i] == cause:
                # Subject i had the event before or at tau
                w = 1.0 / max(G_t, 1e-6)
                bs += w * (1 - cif_tau[i]) ** 2
                n_valid += 1
            elif t[i] > tau:
                # Subject i survived past tau
                w = 1.0 / max(G_tau, 1e-6)
                bs += w * cif_tau[i] ** 2
                n_valid += 1
            # else: event before tau but different cause — excluded

        scores.append(bs / n_valid if n_valid > 0 else np.nan)

    return eval_times, np.array(scores)


def integrated_brier_score(
    H: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    cause: int = 1,
) -> float:
    """Integrated Brier Score via trapezoidal rule over the time range."""
    times, scores = brier_score(H, t, e, cause=cause)
    valid = ~np.isnan(scores)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapz(scores[valid], times[valid]) / (times[valid][-1] - times[valid][0]))


# ---------------------------------------------------------------------------
# D-calibration
# ---------------------------------------------------------------------------

def d_calibration(
    H: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    cause: int = 1,
    n_bins: int = 10,
) -> dict:
    """
    D-calibration: tests whether events fall uniformly across predicted risk deciles.

    A well-calibrated model has observed event rate ≈ predicted CIF in each decile.

    Returns:
        dict with keys: decile_edges, observed_rates, predicted_rates, chi2_stat, p_value
    """
    from scipy import stats

    cause_idx = cause - 1
    event_mask = e == cause

    if event_mask.sum() == 0:
        return {}

    # Use CIF at each subject's observed time as the "predicted risk"
    cif_at_t = np.array([
        H[i, cause_idx, : int(t[i]) + 1].sum() for i in range(len(t))
    ])

    decile_edges = np.percentile(cif_at_t, np.linspace(0, 100, n_bins + 1))
    decile_edges[0] -= 1e-6
    decile_edges[-1] += 1e-6

    observed_counts = []
    expected_counts = []

    for lo, hi in zip(decile_edges[:-1], decile_edges[1:]):
        in_bin = (cif_at_t > lo) & (cif_at_t <= hi)
        observed_counts.append(event_mask[in_bin].sum())
        expected_counts.append(cif_at_t[in_bin].mean() * in_bin.sum() if in_bin.sum() > 0 else 0)

    observed_counts = np.array(observed_counts, dtype=float)
    expected_counts = np.array(expected_counts, dtype=float)

    # Chi-squared goodness-of-fit
    safe_expected = np.where(expected_counts > 0, expected_counts, 1e-6)
    chi2_stat = float(((observed_counts - expected_counts) ** 2 / safe_expected).sum())
    p_value = float(1 - stats.chi2.cdf(chi2_stat, df=n_bins - 1))

    total = observed_counts.sum()
    return {
        "decile_edges": decile_edges.tolist(),
        "observed_rates": (observed_counts / total).tolist(),
        "predicted_rates": (expected_counts / total).tolist(),
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "n_events": int(event_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def full_evaluation_report(
    H: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    cause: int = 1,
    output_json: str | None = None,
) -> dict:
    """Run all metrics and return a summary dict."""
    import json

    cindex = c_index_competing_risks(H, t, e, cause=cause)
    times, bs_scores = brier_score(H, t, e, cause=cause)
    ibs = integrated_brier_score(H, t, e, cause=cause)
    dcal = d_calibration(H, t, e, cause=cause)

    report = {
        "c_index": round(cindex, 4),
        "integrated_brier_score": round(float(ibs), 4) if not np.isnan(ibs) else None,
        "brier_score_by_time": {
            str(int(tm)): round(float(sc), 4)
            for tm, sc in zip(times, bs_scores)
            if not np.isnan(sc)
        },
        "d_calibration": dcal,
    }

    print("\n=== Evaluation Report ===")
    print(f"  C-index (cause {cause}):  {cindex:.4f}")
    print(f"  Integrated Brier Score: {ibs:.4f}" if not np.isnan(ibs) else "  IBS: N/A")
    if dcal:
        print(f"  D-calibration χ²={dcal['chi2_stat']:.2f}  p={dcal['p_value']:.4f}")

    if output_json:
        from pathlib import Path
        Path(output_json).write_text(json.dumps(report, indent=2))
        print(f"  Report saved → {output_json}")

    return report
