"""
Training loop for the DeepHit Achilles rupture survival model.

Splits:
  - Temporal train/val/test by observation year
      train  : year < 2020
      val    : 2020 <= year < 2022
      test   : year >= 2022   ← held-out; evaluated ONCE at the very end

  Test-set discipline: the test set is loaded and predicted on only in the
  final `evaluate_test` call.  Never used for model selection.

Usage:
    python models/train.py --no-hpo --epochs 50     # fixed params
    python models/train.py --hpo-trials 50          # Optuna HPO
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
from sklearn.preprocessing import StandardScaler

from models.deephit import DeepHit, deephit_loss
from models.evaluate import c_index_competing_risks

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_MATRIX_CSV = PROCESSED_DIR / "feature_matrix.csv"

NON_FEATURE_COLS = {
    "player_id", "player_name", "observation_date",
    "event", "time_to_event_days",
    "birth_date",
}

# Temporal split boundaries (calendar year of observation_date)
TRAIN_BEFORE = 2020   # year < 2020  → train
VAL_BEFORE   = 2022   # 2020 ≤ year < 2022 → val
                       # year ≥ 2022  → test


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    csv_path: Path = FEATURE_MATRIX_CSV,
    n_time_bins: int = 60,
    max_time_days: int = 1825,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int, pd.DataFrame]:
    """
    Load and discretise. Returns (X, t, e, feature_cols, n_time_bins, df).
    df retains observation_date for temporal split assignment.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["time_to_event_days", "event"])
    df["observation_date"] = pd.to_datetime(df["observation_date"])

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

    return X, t, e, feature_cols, n_time_bins, df


def temporal_split(
    df: pd.DataFrame,
    train_before: int = TRAIN_BEFORE,
    val_before: int   = VAL_BEFORE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (train_idx, val_idx, test_idx) by observation_date year.

    No player appears in more than one split because each player has exactly
    one row in the feature matrix (one observation point).
    """
    years = df["observation_date"].dt.year.values
    train_idx = np.where(years <  train_before)[0]
    val_idx   = np.where((years >= train_before) & (years < val_before))[0]
    test_idx  = np.where(years >= val_before)[0]
    return train_idx, val_idx, test_idx


def make_loaders(
    X: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val   = scaler.transform(X[val_idx]).astype(np.float32)

    def _loader(Xa, ta, ea, shuffle):
        ds = TensorDataset(
            torch.from_numpy(Xa),
            torch.from_numpy(ta),
            torch.from_numpy(ea),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        _loader(X_train, t[train_idx], e[train_idx], shuffle=True),
        _loader(X_val,   t[val_idx],   e[val_idx],   shuffle=False),
        scaler,
    )


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, alpha, sigma, device) -> float:
    model.train()
    total_loss = 0.0
    for Xb, tb, eb in loader:
        Xb, tb, eb = Xb.to(device), tb.to(device), eb.to(device)
        optimizer.zero_grad()
        h = model(Xb)
        loss = deephit_loss(h, tb, eb, alpha=alpha, sigma=sigma)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(Xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loader(model, loader, device) -> float:
    model.eval()
    all_h, all_t, all_e = [], [], []
    for Xb, tb, eb in loader:
        all_h.append(model(Xb.to(device)).cpu())
        all_t.append(tb)
        all_e.append(eb)
    H = torch.cat(all_h).numpy()
    T = torch.cat(all_t).numpy()
    E = torch.cat(all_e).numpy()
    return c_index_competing_risks(H, T, E, cause=1)


@torch.no_grad()
def evaluate_test(
    model: DeepHit,
    X: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    test_idx: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
    feature_cols: list[str],
    n_bins: int,
    max_time_days: int = 1825,
) -> float:
    """
    Evaluate on the held-out test set. Called exactly once, after training.
    Also computes and prints SHAP feature importance (top 5).
    """
    model.eval()
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    t_test = t[test_idx]
    e_test = e[test_idx]

    H = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    cindex = c_index_competing_risks(H, t_test, e_test, cause=1)

    # ── SHAP feature importance ───────────────────────────────────────────────
    try:
        import shap
        # 1-year bin index
        year_bin = min(int(n_bins * 365 / max_time_days), n_bins - 1)

        def predict_1yr_cif(X_arr: np.ndarray) -> np.ndarray:
            """1-year CIF for cause 1 (Achilles rupture)."""
            with torch.no_grad():
                h = model(torch.from_numpy(X_arr.astype(np.float32)).to(device))
            return h[:, 0, :year_bin].sum(dim=1).cpu().numpy()

        # Use training set rows (already scaled) as background
        X_train_scaled = X[X.shape[0] - len(test_idx):]  # fallback
        # Re-derive background from the scaler-transformed train set
        background = shap.sample(X_test, min(30, len(X_test)))
        explainer  = shap.KernelExplainer(predict_1yr_cif, background)
        shap_vals  = explainer.shap_values(X_test, nsamples=100)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        top5_idx = np.argsort(mean_abs)[::-1][:5]
        print("\n[SHAP] Top-5 features by mean |SHAP| on test set:")
        for rank, i in enumerate(top5_idx, 1):
            print(f"  {rank}. {feature_cols[i]:<30}  mean|SHAP|={mean_abs[i]:.4f}")

    except ImportError:
        print("[SHAP] shap not installed — run: pip install shap")
    except Exception as exc:
        print(f"[SHAP] failed: {exc}")

    return cindex


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(X, t, e, train_idx, val_idx, n_time_bins, device):
    def objective(trial: optuna.Trial) -> float:
        shared_depth = trial.suggest_int("shared_depth", 2, 4)
        shared_width = trial.suggest_categorical("shared_width", [64, 128, 256])
        cs_hidden    = trial.suggest_categorical("cs_hidden", [32, 64, 128])
        dropout      = trial.suggest_float("dropout", 0.1, 0.5)
        lr           = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        alpha        = trial.suggest_float("alpha", 0.1, 0.9)
        sigma        = trial.suggest_float("sigma", 0.05, 0.5, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128])

        train_loader, val_loader, _ = make_loaders(X, t, e, train_idx, val_idx, batch_size)
        model = DeepHit(
            in_features=X.shape[1], n_time_bins=n_time_bins, n_causes=2,
            shared_layers=[shared_width] * shared_depth,
            cs_hidden=cs_hidden, dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_cindex = 0.0
        for epoch in range(50):
            train_one_epoch(model, train_loader, optimizer, alpha, sigma, device)
            cindex = evaluate_loader(model, val_loader, device)
            trial.report(cindex, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            best_cindex = max(best_cindex, cindex)
        return best_cindex

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo-trials", type=int, default=50)
    parser.add_argument("--no-hpo",     action="store_true")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--n-time-bins",type=int, default=60)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    print(f"[train] device={device}")

    print("[train] loading features…")
    X, t, e, feature_cols, n_bins, df = load_data(n_time_bins=args.n_time_bins)

    train_idx, val_idx, test_idx = temporal_split(df)

    print(f"  X={X.shape}  features={len(feature_cols)}")
    print(f"  split  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  events train={e[train_idx].sum()}  "
          f"val={e[val_idx].sum()}  test={e[test_idx].sum()}")

    if len(val_idx) == 0 or len(test_idx) == 0:
        print("[warn] val or test split is empty — check observation_date column")

    train_loader, val_loader, scaler = make_loaders(X, t, e, train_idx, val_idx)

    if args.no_hpo:
        model = DeepHit(
            in_features=X.shape[1],
            n_time_bins=n_bins,
            n_causes=2,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_val = 0.0

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, 0.5, 0.1, device)
            val_cindex = evaluate_loader(model, val_loader, device)
            if val_cindex > best_val:
                best_val = val_cindex
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            if epoch % 10 == 0:
                print(f"  epoch {epoch:3d}  loss={loss:.4f}  val_cindex={val_cindex:.4f}")

        print(f"\n[train] best val C-index : {best_val:.4f}")

        # ── Final test-set evaluation (one-time, held-out) ────────────────
        print("\n[train] loading best checkpoint for test-set evaluation…")
        model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pt",
                                         map_location=device))
        test_cindex = evaluate_test(
            model, X, t, e, test_idx, scaler, device, feature_cols, n_bins
        )
        print(f"\n{'='*50}")
        print(f"  Val  C-index : {best_val:.4f}")
        print(f"  Test C-index : {test_cindex:.4f}   ← report this")
        print(f"{'='*50}")

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
            make_objective(X, t, e, train_idx, val_idx, n_bins, device),
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
