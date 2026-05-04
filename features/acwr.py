"""
Acute:Chronic Workload Ratio (ACWR) computation via EWMA.

Three window-scale variants:
  - Short:  acute=1w  chronic=4w   (standard sports-science baseline)
  - Medium: acute=2w  chronic=6w   (extended chronic window)
  - Long:   acute=4w  chronic=12w  (season-level chronic baseline)

All computations are done with exponentially weighted moving averages
(span parameterisation) rather than simple rolling means, which reduces
the spike artefact at window boundaries.

Input:  DataFrame with columns [player_id, game_date, workload]
        where workload is typically minutes_played or a composite load score.
Output: original DataFrame with added columns:
        acwr_short, acwr_medium, acwr_long
        plus the underlying ewma_acute_* and ewma_chronic_* columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (acute_days, chronic_days)  — EWMA span derived as 2*window_days - 1
WINDOW_CONFIGS = {
    "short":  (7, 28),
    "medium": (14, 42),
    "long":   (28, 84),
}

# Avoid division by zero: if chronic EWMA < this floor, ACWR = NaN
CHRONIC_FLOOR = 1e-3


def _ewma_span(days: int) -> float:
    """Convert a rolling-window width (days) to EWMA span parameter."""
    return float(2 * days - 1)


def compute_acwr(
    df: pd.DataFrame,
    player_col: str = "player_id",
    date_col: str = "game_date",
    load_col: str = "minutes_played",
    fill_gaps: bool = True,
) -> pd.DataFrame:
    """
    Compute ACWR at three window scales for every player.

    Args:
        df:           Game-level DataFrame.
        player_col:   Column identifying the player.
        date_col:     Date of each game (will be coerced to datetime).
        load_col:     Workload measure per game.
        fill_gaps:    If True, insert zero-load rows for calendar days
                      with no game, so EWMA decay is time-aware.

    Returns:
        df with new acwr_short, acwr_medium, acwr_long columns added.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    result_frames: list[pd.DataFrame] = []

    for pid, group in df.groupby(player_col):
        group = group.sort_values(date_col).copy()

        if fill_gaps:
            group = _fill_calendar_gaps(group, date_col, load_col)

        for scale, (acute_days, chronic_days) in WINDOW_CONFIGS.items():
            acute_span = _ewma_span(acute_days)
            chronic_span = _ewma_span(chronic_days)

            ewma_acute = (
                group[load_col]
                .ewm(span=acute_span, min_periods=1, adjust=True)
                .mean()
            )
            ewma_chronic = (
                group[load_col]
                .ewm(span=chronic_span, min_periods=1, adjust=True)
                .mean()
            )

            acwr = np.where(
                ewma_chronic < CHRONIC_FLOOR,
                np.nan,
                ewma_acute / ewma_chronic,
            )

            group[f"ewma_acute_{scale}"] = ewma_acute.values
            group[f"ewma_chronic_{scale}"] = ewma_chronic.values
            group[f"acwr_{scale}"] = acwr

        result_frames.append(group)

    out = pd.concat(result_frames, ignore_index=True)
    # Drop gap-fill rows (zero-load rows inserted for EWMA continuity)
    if fill_gaps:
        out = out[out["_gap_fill"].eq(False)].drop(columns=["_gap_fill"])
    return out.sort_values([player_col, date_col]).reset_index(drop=True)


def _fill_calendar_gaps(
    group: pd.DataFrame,
    date_col: str,
    load_col: str,
) -> pd.DataFrame:
    """Insert zero-load rows for every calendar day without a game."""
    if group.empty:
        group["_gap_fill"] = False
        return group

    group = group.copy()
    group["_gap_fill"] = False

    full_range = pd.date_range(
        start=group[date_col].min(),
        end=group[date_col].max(),
        freq="D",
    )
    existing_dates = set(group[date_col].dt.normalize())
    gap_dates = [d for d in full_range if d not in existing_dates]

    if not gap_dates:
        return group

    gap_rows = pd.DataFrame(
        {
            date_col: gap_dates,
            load_col: 0.0,
            "_gap_fill": True,
        }
    )
    # Propagate player_id and other needed columns
    for col in group.columns:
        if col not in gap_rows.columns and col != "_gap_fill":
            gap_rows[col] = group[col].iloc[0]

    return (
        pd.concat([group, gap_rows], ignore_index=True)
        .sort_values(date_col)
        .reset_index(drop=True)
    )


def add_workload_spikes(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Flag games where any ACWR scale exceeds threshold.
    Adds boolean columns: spike_short, spike_medium, spike_long, spike_any.
    """
    df = df.copy()
    for scale in WINDOW_CONFIGS:
        col = f"acwr_{scale}"
        if col in df.columns:
            df[f"spike_{scale}"] = df[col] > threshold
    spike_cols = [f"spike_{s}" for s in WINDOW_CONFIGS if f"spike_{s}" in df.columns]
    if spike_cols:
        df["spike_any"] = df[spike_cols].any(axis=1)
    return df


def compute_rolling_minutes(
    df: pd.DataFrame,
    player_col: str = "player_id",
    date_col: str = "game_date",
    load_col: str = "minutes_played",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Simple rolling-sum and rolling-mean of minutes over multiple windows.
    Complementary feature to ACWR.
    """
    if windows is None:
        windows = [7, 14, 30, 60]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    frames = []

    for pid, group in df.groupby(player_col):
        group = group.sort_values(date_col).copy()
        for w in windows:
            group[f"rolling_sum_{w}d"] = (
                group[load_col]
                .rolling(w, min_periods=1)
                .sum()
            )
            group[f"rolling_mean_{w}d"] = (
                group[load_col]
                .rolling(w, min_periods=1)
                .mean()
            )
        frames.append(group)

    return pd.concat(frames, ignore_index=True).sort_values(
        [player_col, date_col]
    ).reset_index(drop=True)
