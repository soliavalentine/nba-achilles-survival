"""
Play-style embedding from play-by-play data.

Computes per-game play-type distribution vectors and optionally reduces
them with PCA to a compact embedding.  The embedding captures movement
patterns (cuts, spot-up shooting, post-ups, pick-and-roll usage) that
are hypothesized to correlate with Achilles tendon loading.

Input:  NBA play-by-play data (from nba_api PBP or synergy-like breakdowns)
Output: per-(player, game_date) embedding DataFrame
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# Play types tracked  (mirrors Synergy / NBA Stats play-type labels)
PLAY_TYPES = [
    "transition",
    "isolation",
    "pick_and_roll_ball_handler",
    "pick_and_roll_man",
    "post_up",
    "spot_up",
    "handoff",
    "cut",
    "off_screen",
    "putback",
    "misc",
]

N_EMBEDDING_DIMS = 8  # PCA target dimension


def compute_play_type_distribution(
    pbp_df: pd.DataFrame,
    player_col: str = "player_id",
    date_col: str = "game_date",
    play_type_col: str = "play_type",
    poss_col: str = "possessions",
) -> pd.DataFrame:
    """
    Aggregate play-by-play possessions into per-(player, game) distributions.

    Args:
        pbp_df:         Play-by-play or play-type breakdown DataFrame.
        player_col:     Player identifier column.
        date_col:       Game date column.
        play_type_col:  Column with play-type label.
        poss_col:       Column with possession count (or 1 per row).

    Returns:
        DataFrame with columns [player_id, game_date, pt_transition, pt_iso, …]
        where each pt_* column is the fraction of possessions of that type.
    """
    pbp_df = pbp_df.copy()
    pbp_df[date_col] = pd.to_datetime(pbp_df[date_col])

    # Pivot to (player, game, play_type) → total possessions
    agg = (
        pbp_df.groupby([player_col, date_col, play_type_col])[poss_col]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Normalize to proportions within each row
    type_cols = [c for c in agg.columns if c not in (player_col, date_col)]
    row_sums = agg[type_cols].sum(axis=1).replace(0, np.nan)
    for col in type_cols:
        agg[f"pt_{col}"] = agg[col] / row_sums
    agg = agg.drop(columns=type_cols).fillna(0)

    return agg


def fit_play_style_pca(
    distribution_df: pd.DataFrame,
    player_col: str = "player_id",
    date_col: str = "game_date",
    n_components: int = N_EMBEDDING_DIMS,
    scaler: StandardScaler | None = None,
    pca: PCA | None = None,
) -> tuple[pd.DataFrame, StandardScaler, PCA]:
    """
    Fit a StandardScaler + PCA on the play-type distribution matrix.

    Returns:
        (embedding_df, fitted_scaler, fitted_pca)
        embedding_df has columns [player_id, game_date, ps_dim_0, …, ps_dim_N]
    """
    feat_cols = [c for c in distribution_df.columns
                 if c.startswith("pt_")]
    X = distribution_df[feat_cols].values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    if pca is None:
        pca = PCA(n_components=n_components, random_state=42)
        X_emb = pca.fit_transform(X_scaled)
    else:
        X_emb = pca.transform(X_scaled)

    emb_cols = [f"ps_dim_{i}" for i in range(n_components)]
    emb_df = distribution_df[[player_col, date_col]].copy()
    for i, col in enumerate(emb_cols):
        emb_df[col] = X_emb[:, i]

    return emb_df, scaler, pca


def build_play_style_features(
    pbp_csv: Path | None = None,
    output_csv: Path | None = None,
    n_components: int = N_EMBEDDING_DIMS,
) -> pd.DataFrame:
    """Full pipeline: raw PBP → play-type distribution → PCA embedding."""
    pbp_csv = pbp_csv or PROCESSED_DIR / "pbp_play_types.csv"
    output_csv = output_csv or PROCESSED_DIR / "play_style_embeddings.csv"

    if not pbp_csv.exists():
        print(f"[play_style] {pbp_csv} not found — skipping")
        return pd.DataFrame()

    pbp_df = pd.read_csv(pbp_csv)
    dist_df = compute_play_type_distribution(pbp_df)
    emb_df, _, _ = fit_play_style_pca(dist_df, n_components=n_components)
    emb_df.to_csv(output_csv, index=False)
    print(f"[play_style] {len(emb_df):,} rows → {output_csv}")
    return emb_df
