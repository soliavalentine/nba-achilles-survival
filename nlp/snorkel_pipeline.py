"""
Snorkel weak supervision pipeline for prodromal Achilles injury labeling.

We have structured IL data (ground truth ruptures from ProSportsTransactions)
but we also want to surface "soft" precursor signals from:
  - beat-reporter game notes
  - injury report PDFs ("questionable – Achilles soreness")
  - player post-game transcripts
  - social media / news snippets

Snorkel lets us define labeling functions (LFs) that vote on each text snippet
and then trains a label model to combine noisy votes into probabilistic labels.

Labels:
  ABSTAIN  = -1
  HEALTHY  = 0   (no Achilles concern)
  PRODROMAL = 1  (Achilles soreness / tendinopathy signal)
  RUPTURE  = 2   (definitive rupture mention)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from snorkel.labeling import LabelingFunction, PandasLFApplier
    from snorkel.labeling.model import LabelModel
    SNORKEL_AVAILABLE = True
except ImportError:
    SNORKEL_AVAILABLE = False
    print("[snorkel] snorkel not installed — labeling functions defined but not runnable")

ABSTAIN = -1
HEALTHY = 0
PRODROMAL = 1
RUPTURE = 2

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Labeling functions
# ---------------------------------------------------------------------------

def _lf(name: str) -> Callable:
    """Decorator for a labeling function operating on a row with a 'text' field."""
    def decorator(fn: Callable) -> Callable:
        fn._lf_name = name
        return fn
    return decorator


@_lf("lf_rupture_explicit")
def lf_rupture_explicit(row) -> int:
    text = row["text"].lower()
    if re.search(r"ruptured?\s+achilles|achilles\s+ruptured?|torn\s+achilles|achilles\s+torn", text):
        return RUPTURE
    return ABSTAIN


@_lf("lf_rupture_season_ending")
def lf_rupture_season_ending(row) -> int:
    text = row["text"].lower()
    if "achilles" in text and re.search(r"season.ending|out\s+for\s+the\s+season|career.threatening", text):
        return RUPTURE
    return ABSTAIN


@_lf("lf_prodromal_soreness")
def lf_prodromal_soreness(row) -> int:
    text = row["text"].lower()
    if re.search(r"achilles\s+(soreness|sore|tightness|tight|discomfort|pain|ache)", text):
        return PRODROMAL
    if re.search(r"(soreness|sore|tightness|tight)\s+(in|of)\s+(his|her|the)\s+achilles", text):
        return PRODROMAL
    return ABSTAIN


@_lf("lf_prodromal_tendinopathy")
def lf_prodromal_tendinopathy(row) -> int:
    text = row["text"].lower()
    if re.search(r"achilles\s+tendinit|tendinopathy|tendinosis|achilles\s+inflam", text):
        return PRODROMAL
    return ABSTAIN


@_lf("lf_prodromal_load_management")
def lf_prodromal_load_management(row) -> int:
    text = row["text"].lower()
    # Load management or DNP + achilles
    if "achilles" in text and re.search(
        r"load\s+manag|rest\s+(day|ing)|dnp|did\s+not\s+play|questionable", text
    ):
        return PRODROMAL
    return ABSTAIN


@_lf("lf_healthy_no_injury")
def lf_healthy_no_injury(row) -> int:
    text = row["text"].lower()
    if re.search(r"no\s+injury|cleared|full\s+practice|no\s+limitations|back\s+to\s+(normal|full)", text):
        if "achilles" not in text:
            return HEALTHY
    return ABSTAIN


@_lf("lf_healthy_activated")
def lf_healthy_activated(row) -> int:
    text = row["text"].lower()
    if re.search(r"activated\s+from.*(il|injured\s+list)|returned?\s+to\s+(the\s+)?lineup", text):
        return HEALTHY
    return ABSTAIN


@_lf("lf_negative_context")
def lf_negative_context(row) -> int:
    # "no achilles issues" → healthy
    text = row["text"].lower()
    if re.search(r"no\s+achilles|achilles\s+(is\s+)?fine|achilles\s+cleared", text):
        return HEALTHY
    return ABSTAIN


ALL_LFS = [
    lf_rupture_explicit,
    lf_rupture_season_ending,
    lf_prodromal_soreness,
    lf_prodromal_tendinopathy,
    lf_prodromal_load_management,
    lf_healthy_no_injury,
    lf_healthy_activated,
    lf_negative_context,
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def apply_labeling_functions(df: pd.DataFrame) -> np.ndarray:
    """
    Apply all LFs to df and return the (n_samples, n_lfs) label matrix.
    Requires snorkel to be installed.
    """
    if not SNORKEL_AVAILABLE:
        raise ImportError("pip install snorkel")

    snorkel_lfs = [
        LabelingFunction(name=fn._lf_name, f=fn)
        for fn in ALL_LFS
    ]
    applier = PandasLFApplier(lfs=snorkel_lfs)
    return applier.apply(df)


def train_label_model(
    L: np.ndarray,
    n_epochs: int = 500,
    lr: float = 0.01,
    seed: int = 42,
) -> "LabelModel":
    """Fit a Snorkel LabelModel on the label matrix L."""
    if not SNORKEL_AVAILABLE:
        raise ImportError("pip install snorkel")
    model = LabelModel(cardinality=3, verbose=True)
    model.fit(L_train=L, n_epochs=n_epochs, lr=lr, seed=seed)
    return model


def run_snorkel_pipeline(
    text_csv: Path | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Full pipeline:
      1. Load raw text snippets.
      2. Apply LFs.
      3. Train label model.
      4. Attach probabilistic labels to each row.
      5. Save enriched CSV.
    """
    text_csv = text_csv or PROCESSED_DIR / "injury_text_snippets.csv"
    output_csv = output_csv or PROCESSED_DIR / "snorkel_labels.csv"

    if not text_csv.exists():
        print(f"[snorkel] {text_csv} not found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(text_csv)
    print(f"[snorkel] {len(df):,} snippets loaded")

    L = apply_labeling_functions(df)

    # Coverage / conflict analysis
    from snorkel.labeling import LFAnalysis
    analysis = LFAnalysis(L=L, lfs=[
        LabelingFunction(name=fn._lf_name, f=fn) for fn in ALL_LFS
    ])
    print(analysis.lf_summary())

    model = train_label_model(L)

    probs = model.predict_proba(L)
    df["prob_healthy"] = probs[:, HEALTHY]
    df["prob_prodromal"] = probs[:, PRODROMAL]
    df["prob_rupture"] = probs[:, RUPTURE]
    df["snorkel_label"] = model.predict(L, tie_break_policy="abstain")

    df.to_csv(output_csv, index=False)
    print(f"[snorkel] saved {len(df):,} rows → {output_csv}")
    return df
