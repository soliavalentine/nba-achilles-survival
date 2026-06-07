#!/usr/bin/env python3
"""
Generate three publication-quality figures for the NBA Achilles survival analysis.

Outputs (300 dpi PNG, saved to figures/):
  shap_importance.png   — top-10 SHAP features, colored by category
  acwr_vs_rupture.png   — acwr_7_28 for ruptures vs controls in test set
  cindex_comparison.png — bar chart comparing demographics-only vs full model
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")                   # headless render
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.deephit import DeepHit
from models.train import load_data, temporal_split, make_loaders
from sklearn.preprocessing import StandardScaler

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       300,
})
DPI = 300

CATEGORY_COLORS = {
    "load":         "#D62728",   # red
    "recovery":     "#FF7F0E",   # orange
    "demographics": "#1F77B4",   # blue
    "workload":     "#AEC7E8",   # light blue (ACWR ratios)
}

FEATURE_CATEGORIES = {
    "games_last_7_days":   "load",
    "games_last_14_days":  "load",
    "acwr_3_21":           "workload",
    "acwr_7_28":           "workload",
    "acwr_14_56":          "workload",
    "acwr_spike_flag":     "workload",
    "days_since_last_game":"recovery",
    "age_at_observation":  "demographics",
    "position_encoded":    "demographics",
    "height_inches":       "demographics",
    "weight_lbs":          "demographics",
    "years_in_league":     "demographics",
}

FIGURES = Path(__file__).parent


# ── Load data + model ─────────────────────────────────────────────────────────

def load_all():
    X, t, e, feat_cols, n_bins, df = load_data()
    train_idx, val_idx, test_idx = temporal_split(df)

    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    X_train_sc = scaler.transform(X[train_idx]).astype(np.float32)
    X_test_sc  = scaler.transform(X[test_idx]).astype(np.float32)

    model = DeepHit(in_features=X.shape[1], n_time_bins=n_bins, n_causes=2)
    ckpt  = ROOT / "models" / "checkpoints" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    return (X, t, e, feat_cols, n_bins, df,
            train_idx, val_idx, test_idx,
            X_train_sc, X_test_sc, scaler, model)


# ── Figure 1: SHAP feature importance ────────────────────────────────────────

def fig_shap(X_train_sc, X_test_sc, feat_cols, n_bins, model):
    year_bin = min(int(n_bins * 365 / 1825), n_bins - 1)

    def predict_1yr_cif(X_arr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            h = model(torch.from_numpy(X_arr.astype(np.float32)))
        return h[:, 0, :year_bin].sum(dim=1).numpy()

    print("  computing SHAP values…")
    background = shap.sample(X_train_sc, min(40, len(X_train_sc)), random_state=42)
    explainer  = shap.KernelExplainer(predict_1yr_cif, background)
    shap_vals  = explainer.shap_values(X_test_sc, nsamples=150)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:10]   # top 10

    features = [feat_cols[i] for i in order]
    values   = mean_abs[order]
    colors   = [CATEGORY_COLORS[FEATURE_CATEGORIES.get(f, "demographics")]
                for f in features]

    # Human-readable labels
    label_map = {
        "days_since_last_game": "Days since last game",
        "games_last_14_days":   "Games — last 14 days",
        "games_last_7_days":    "Games — last 7 days",
        "acwr_14_56":           "ACWR 14:56",
        "acwr_7_28":            "ACWR 7:28",
        "acwr_3_21":            "ACWR 3:21",
        "acwr_spike_flag":      "ACWR spike flag",
        "height_inches":        "Height (inches)",
        "weight_lbs":           "Weight (lbs)",
        "age_at_observation":   "Age at observation",
        "position_encoded":     "Position",
        "years_in_league":      "Years in league",
    }
    labels = [label_map.get(f, f) for f in features]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(range(len(features)), values[::-1], color=colors[::-1],
                   height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("SHAP Feature Importance — Test Set", pad=12)
    ax.tick_params(axis="x", labelsize=9)

    # Legend
    legend_items = [
        mpatches.Patch(color=CATEGORY_COLORS["load"],         label="Recent load"),
        mpatches.Patch(color=CATEGORY_COLORS["recovery"],     label="Recovery"),
        mpatches.Patch(color=CATEGORY_COLORS["workload"],     label="Workload ratio (ACWR)"),
        mpatches.Patch(color=CATEGORY_COLORS["demographics"], label="Demographics"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()
    out = FIGURES / "shap_importance.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return shap_vals, mean_abs


# ── Figure 2: ACWR 7:28 — ruptures vs controls (test set) ───────────────────

def fig_acwr(df, test_idx, e):
    test_df = df.iloc[test_idx].copy()
    test_df["acwr_7_28"] = pd.to_numeric(test_df["acwr_7_28"], errors="coerce")

    rupt_vals = test_df.loc[test_df["event"] == 1, "acwr_7_28"].dropna()
    ctrl_vals = test_df.loc[test_df["event"] == 0, "acwr_7_28"].dropna()

    groups      = ["Ruptures\n(n=9)", "Matched controls\n(n=27)"]
    means       = [rupt_vals.mean(), ctrl_vals.mean()]
    stds        = [rupt_vals.std(ddof=1), ctrl_vals.std(ddof=1)]
    bar_colors  = ["#D62728", "#1F77B4"]

    fig, ax = plt.subplots(figsize=(5, 5))

    x = np.array([0, 1])
    bars = ax.bar(x, means, yerr=stds, color=bar_colors,
                  width=0.45, capsize=8, error_kw={"linewidth": 1.5},
                  edgecolor="white", linewidth=0.5, alpha=0.88)

    # Individual data points (jittered)
    rng = np.random.default_rng(42)
    for xi, vals, c in zip(x, [rupt_vals, ctrl_vals], bar_colors):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(xi + jitter, vals, color=c, s=28, zorder=3,
                   alpha=0.7, edgecolors="white", linewidths=0.5)

    # Threshold line at ACWR = 1.5
    ax.axhline(1.5, color="#888888", linestyle="--", linewidth=1.2,
               label="ACWR spike threshold (1.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("ACWR 7:28 at observation date", fontsize=11)
    ax.set_title("Workload Ratio at Time of Rupture\nvs Matched Controls (Test Set)", pad=12)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="#cccccc")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = FIGURES / "acwr_vs_rupture.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 3: C-index comparison ─────────────────────────────────────────────

def fig_cindex():
    labels = ["Demographics\nonly\n(5 features)", "Demographics\n+ ACWR\n(12 features)"]
    values = [0.46, 0.81]
    colors = ["#1F77B4", "#D62728"]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    x = np.array([0, 1])
    bars = ax.bar(x, values, color=colors, width=0.45,
                  edgecolor="white", linewidth=0.5, alpha=0.88)

    # Value labels on bars
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.2f}", ha="center", va="bottom", fontsize=13,
                fontweight="bold")

    # Chance line
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1.2,
               label="Chance (0.50)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Harrell C-index (test set)", fontsize=11)
    ax.set_title("C-index by Feature Set", pad=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor="#cccccc")

    # Annotation for delta
    ax.annotate(
        "",
        xy=(1, 0.81), xytext=(0, 0.46),
        arrowprops=dict(arrowstyle="-|>", color="#333333",
                        lw=1.5, mutation_scale=14),
    )
    ax.text(0.5, 0.64, "+0.35", ha="center", va="center",
            fontsize=11, color="#333333", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#cccccc", lw=0.8))

    fig.tight_layout()
    out = FIGURES / "cindex_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data and model…")
    (X, t, e, feat_cols, n_bins, df,
     train_idx, val_idx, test_idx,
     X_train_sc, X_test_sc, scaler, model) = load_all()

    print("\n[1/3] SHAP feature importance…")
    fig_shap(X_train_sc, X_test_sc, feat_cols, n_bins, model)

    print("\n[2/3] ACWR ruptures vs controls…")
    fig_acwr(df, test_idx, e)

    print("\n[3/3] C-index comparison…")
    fig_cindex()

    print("\nDone. Figures saved to figures/")


if __name__ == "__main__":
    main()
