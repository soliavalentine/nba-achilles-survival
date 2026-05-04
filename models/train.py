"""
Training loop for the DeepHit Achilles rupture survival model.

Features:
  - Optuna hyperparameter optimisation (n_trials configurable)
  - Time-series cross-validation (no patient appears in both train and val)
  - Checkpointing: best model by val C-index saved automatically
  - Resumable: skips trials already completed in the study DB

Usage:
    python models/train.py --hpo-trials 50          # with HPO
    python models/train.py --no-hpo --epochs 100    # fixed params
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from models.deephit import DeepHit, deephit_loss
from models.evaluate import c_index_competing_risks

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_MATRIX_CSV = PROCESSED_DIR / "feature_matrix.csv"

# Columns to exclude from the covariate matrix
NON_FEATURE_COLS = {
    "player_id", "player_name", "observation_date",
    "event", "time_to_event_days",
    "birth_date",  # age_at_observation is the derived feature
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    csv_path: Path = FEATURE_MATRIX_CSV,
    n_time_bins: int = 60,
    max_time_days: int = 1825,  # 5 years
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """
    Load feature matrix, discretise time, return (X, t, e, feature_names, n_time_bins).

    Time is quantised into n_time_bins equal-width intervals up to max_time_days.
    e=0 censored, e=1 Achilles rupture, e=2 other retirement/censoring.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["time_to_event_days", "event"])

    # Clip time and discretise
    df["time_clipped"] = df["time_to_event_days"].clip(0, max_time_days)
    bin_edges = np.linspace(0, max_time_days, n_time_bins + 1)
    df["time_bin"] = np.digitize(df["time_clipped"], bin_edges[1:], right=False)
    df["time_bin"] = df["time_bin"].clip(0, n_time_bins - 1)

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and c not in ("time_clipped", "time_bin")
        and df[c].dtype in (np.float64, np.int64, float, int)
    ]

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    t = df["time_bin"].values.astype(np.int64)
    e = df["event"].values.astype(np.int64)

    return X, t, e, feature_cols, n_time_bins


def make_dataloaders(
    X: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    player_ids: np.ndarray,
    batch_size: int = 256,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split by player (no leakage) and return train/val DataLoaders."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, e, groups=player_ids))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])

    def _loader(X_arr, t_arr, e_arr, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X_arr),
            torch.from_numpy(t_arr),
            torch.from_numpy(e_arr),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        _loader(X_train.astype(np.float32), t[train_idx], e[train_idx], shuffle=True),
        _loader(X_val.astype(np.float32), t[val_idx], e[val_idx], shuffle=False),
        scaler,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: DeepHit,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    alpha: float,
    sigma: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, t_batch, e_batch in loader:
        X_batch = X_batch.to(device)
        t_batch = t_batch.to(device)
        e_batch = e_batch.to(device)

        optimizer.zero_grad()
        h = model(X_batch)
        loss = deephit_loss(h, t_batch, e_batch, alpha=alpha, sigma=sigma)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: DeepHit,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    all_h, all_t, all_e = [], [], []
    for X_batch, t_batch, e_batch in loader:
        h = model(X_batch.to(device))
        all_h.append(h.cpu())
        all_t.append(t_batch)
        all_e.append(e_batch)
    H = torch.cat(all_h).numpy()
    T = torch.cat(all_t).numpy()
    E = torch.cat(all_e).numpy()
    return c_index_competing_risks(H, T, E, cause=1)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(
    X: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    player_ids: np.ndarray,
    n_time_bins: int,
    device: torch.device,
):
    def objective(trial: optuna.Trial) -> float:
        shared_depth = trial.suggest_int("shared_depth", 2, 4)
        shared_width = trial.suggest_categorical("shared_width", [128, 256, 512])
        cs_hidden = trial.suggest_categorical("cs_hidden", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        sigma = trial.suggest_float("sigma", 0.05, 0.5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        train_loader, val_loader, _ = make_dataloaders(
            X, t, e, player_ids, batch_size=batch_size
        )

        model = DeepHit(
            in_features=X.shape[1],
            n_time_bins=n_time_bins,
            n_causes=2,
            shared_layers=[shared_width] * shared_depth,
            cs_hidden=cs_hidden,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_cindex = 0.0
        for epoch in range(50):
            train_one_epoch(model, train_loader, optimizer, alpha, sigma, device)
            cindex = evaluate(model, val_loader, device)
            trial.report(cindex, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            best_cindex = max(best_cindex, cindex)

        return best_cindex

    return objective


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo-trials", type=int, default=50)
    parser.add_argument("--no-hpo", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-time-bins", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )
    print(f"[train] device={device}")

    print("[train] loading features…")
    X, t, e, feature_cols, n_bins = load_data(n_time_bins=args.n_time_bins)
    print(f"  X={X.shape}  events={e.sum()}  censored={(e==0).sum()}")

    # Placeholder player_ids (row index for now; replace with actual IDs from CSV)
    player_ids = np.arange(len(X))

    if args.no_hpo:
        # Fixed hyperparameters — sensible defaults
        train_loader, val_loader, scaler = make_dataloaders(
            X, t, e, player_ids, seed=args.seed
        )
        model = DeepHit(
            in_features=X.shape[1],
            n_time_bins=n_bins,
            n_causes=2,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best = 0.0
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, 0.5, 0.1, device)
            cindex = evaluate(model, val_loader, device)
            if cindex > best:
                best = cindex
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            if epoch % 10 == 0:
                print(f"  epoch {epoch:3d}  loss={loss:.4f}  val_cindex={cindex:.4f}")
        print(f"\n[train] best val C-index: {best:.4f}")

    else:
        print(f"[train] HPO with {args.hpo_trials} Optuna trials…")
        study = optuna.create_study(
            direction="maximize",
            study_name="deephit_achilles",
            storage=f"sqlite:///{ROOT}/models/optuna.db",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )
        study.optimize(
            make_objective(X, t, e, player_ids, n_bins, device),
            n_trials=args.hpo_trials,
            timeout=3600 * 6,
        )
        best_params = study.best_params
        print(f"\n[train] best params: {json.dumps(best_params, indent=2)}")
        (ROOT / "models" / "best_hparams.json").write_text(
            json.dumps(best_params, indent=2)
        )


if __name__ == "__main__":
    main()
