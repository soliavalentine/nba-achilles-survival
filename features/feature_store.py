"""
Point-in-time feature store.

Joins all feature tables to a spine of (player_id, observation_date) pairs
with strict temporal discipline: no feature value from after observation_date
is ever used.

Key design decisions:
  - All joins use as-of semantics (merge_asof).
  - The "label" (time-to-event, event indicator) is attached separately
    and only from the ground-truth Achilles dataset.
  - Feature columns are versioned via a feature_version suffix so that
    re-engineered features can coexist with old ones during ablations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Spine builder
# ---------------------------------------------------------------------------

def build_observation_spine(
    profiles_csv: Path | None = None,
    achilles_csv: Path | None = None,
    observation_interval_days: int = 30,
) -> pd.DataFrame:
    """
    Build (player_id, observation_date) spine for survival analysis.

    For each player we generate one observation date per month from their
    NBA debut until either their Achilles event or the study end date
    (2024-10-01).  This creates the time-varying covariate structure
    needed for landmark survival analysis.

    Returns a DataFrame with columns:
      player_id, player_name, observation_date,
      event (0/1), time_to_event_days
    """
    if profiles_csv is None:
        profiles_csv = PROCESSED_DIR / "bbref_player_profiles.csv"
    if achilles_csv is None:
        achilles_csv = PROCESSED_DIR / "achilles_ground_truth.csv"

    profiles = pd.read_csv(profiles_csv, parse_dates=["birth_date"])
    achilles = pd.read_csv(achilles_csv, parse_dates=["date"])

    # Keep only ruptures as the primary event
    ruptures = achilles[achilles["severity"] == "rupture"].copy()
    ruptures = ruptures.sort_values("date").groupby("player_name").first().reset_index()
    ruptures = ruptures.rename(columns={"date": "event_date", "player_name": "name"})

    study_end = pd.Timestamp("2024-10-01")
    rows = []

    for _, player in profiles.iterrows():
        pid = player.get("player_id", player.get("name", ""))
        name = player.get("name", "")

        # Rough debut: use draft_year if available, else year_min
        debut_year = player.get("draft_year", player.get("year_min", 1990))
        debut_date = pd.Timestamp(f"{int(debut_year)}-10-01")

        # Check for Achilles event
        event_row = ruptures[ruptures["name"] == name]
        if not event_row.empty:
            event_date = event_row.iloc[0]["event_date"]
            end_date = event_date
            event = 1
        else:
            end_date = study_end
            event = 0

        # Generate monthly observation dates
        obs_dates = pd.date_range(debut_date, end_date, freq=f"{observation_interval_days}D")
        for obs_date in obs_dates:
            time_to_event = (end_date - obs_date).days
            rows.append(
                {
                    "player_id": pid,
                    "player_name": name,
                    "observation_date": obs_date,
                    "event": event,
                    "time_to_event_days": max(time_to_event, 0),
                }
            )

    spine = pd.DataFrame(rows)
    spine["observation_date"] = pd.to_datetime(spine["observation_date"])
    return spine.sort_values(["player_id", "observation_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Point-in-time feature join
# ---------------------------------------------------------------------------

def pit_join(
    spine: pd.DataFrame,
    feature_df: pd.DataFrame,
    player_col: str = "player_id",
    date_col: str = "observation_date",
    feature_date_col: str = "game_date",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Left-join feature_df onto the spine using as-of (last-known-value) semantics.

    For each row in the spine, this finds the most recent row in feature_df
    with feature_date_col <= observation_date, per player.

    Args:
        spine:            Observation spine with (player_id, observation_date).
        feature_df:       Time-varying features with (player_id, game_date, ...).
        player_col:       Player identifier column (must exist in both).
        date_col:         Date column in spine.
        feature_date_col: Date column in feature_df.
        feature_cols:     Subset of feature columns to join (None = all).

    Returns:
        spine enriched with the requested feature columns.
    """
    spine = spine.copy()
    feature_df = feature_df.copy()

    spine[date_col] = pd.to_datetime(spine[date_col])
    feature_df[feature_date_col] = pd.to_datetime(feature_df[feature_date_col])

    if feature_cols is not None:
        keep = [player_col, feature_date_col] + feature_cols
        feature_df = feature_df[keep]

    feature_df = feature_df.sort_values([player_col, feature_date_col])

    result_frames = []
    for pid, spine_group in spine.groupby(player_col):
        feat_group = feature_df[feature_df[player_col] == pid].sort_values(
            feature_date_col
        )
        if feat_group.empty:
            result_frames.append(spine_group)
            continue

        merged = pd.merge_asof(
            spine_group.sort_values(date_col),
            feat_group,
            left_on=date_col,
            right_on=feature_date_col,
            by=player_col,
            direction="backward",
        )
        result_frames.append(merged)

    return pd.concat(result_frames, ignore_index=True).sort_values(
        [player_col, date_col]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Static feature join (bio: age at observation, position, etc.)
# ---------------------------------------------------------------------------

def join_static_features(
    spine: pd.DataFrame,
    profiles_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Attach time-invariant player bio features, computing age dynamically.
    """
    if profiles_csv is None:
        profiles_csv = PROCESSED_DIR / "bbref_player_profiles.csv"

    profiles = pd.read_csv(profiles_csv, parse_dates=["birth_date"])
    profiles = profiles.rename(columns={"name": "player_name"})

    spine = spine.merge(
        profiles[["player_name", "birth_date", "position", "height_inches", "weight_lbs",
                   "college", "draft_year", "draft_pick"]],
        on="player_name",
        how="left",
    )
    spine["observation_date"] = pd.to_datetime(spine["observation_date"])

    # Compute age at each observation date (no leakage — birth_date is fixed)
    spine["age_at_observation"] = (
        (spine["observation_date"] - spine["birth_date"]).dt.days / 365.25
    )
    # BMI proxy
    spine["bmi"] = (spine["weight_lbs"] * 703) / (spine["height_inches"] ** 2)
    return spine


# ---------------------------------------------------------------------------
# Final feature matrix assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    acwr_csv: Path | None = None,
    gamelogs_csv: Path | None = None,
    playstyle_csv: Path | None = None,
    profiles_csv: Path | None = None,
    achilles_csv: Path | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: spine → static features → time-varying features → label.

    Output columns include everything needed to train the DeepHit model:
      player_id, observation_date, age_at_observation, position,
      height_inches, weight_lbs, bmi,
      acwr_short, acwr_medium, acwr_long,
      ewma_acute_*, ewma_chronic_*, spike_*,
      rolling_sum_*, rolling_mean_*,
      [playstyle embedding dims if available],
      event, time_to_event_days
    """
    acwr_csv = acwr_csv or PROCESSED_DIR / "acwr_features.csv"
    profiles_csv = profiles_csv or PROCESSED_DIR / "bbref_player_profiles.csv"
    achilles_csv = achilles_csv or PROCESSED_DIR / "achilles_ground_truth.csv"
    output_csv = output_csv or PROCESSED_DIR / "feature_matrix.csv"

    print("[feature_store] building observation spine…")
    spine = build_observation_spine(profiles_csv, achilles_csv)
    print(f"  spine: {len(spine):,} rows, {spine['player_id'].nunique()} players")

    print("[feature_store] joining static bio features…")
    spine = join_static_features(spine, profiles_csv)

    if acwr_csv.exists():
        print("[feature_store] joining ACWR / workload features…")
        acwr_df = pd.read_csv(acwr_csv, parse_dates=["game_date"])
        spine = pit_join(
            spine,
            acwr_df,
            player_col="player_id",
            date_col="observation_date",
            feature_date_col="game_date",
        )

    if playstyle_csv and Path(playstyle_csv).exists():
        print("[feature_store] joining play-style embeddings…")
        ps_df = pd.read_csv(playstyle_csv, parse_dates=["game_date"])
        spine = pit_join(spine, ps_df, feature_date_col="game_date")

    if gamelogs_csv and Path(gamelogs_csv).exists():
        print("[feature_store] joining game-log aggregates…")
        gl_df = pd.read_csv(gamelogs_csv, parse_dates=["game_date"])
        spine = pit_join(spine, gl_df, feature_date_col="game_date")

    spine.to_csv(output_csv, index=False)
    print(f"[feature_store] saved {len(spine):,} rows → {output_csv}")
    return spine
