"""
BioBERT fine-tuning for prodromal Achilles injury scoring.

Fine-tunes dmis-lab/biobert-base-cased-v1.2 on the snorkel-labeled
text snippets to produce a continuous "prodromal risk score" per text.

The task is framed as 3-class sequence classification:
  0 = HEALTHY, 1 = PRODROMAL, 2 = RUPTURE

At inference time we use the softmax probability of class 1 (PRODROMAL)
as a continuous risk signal that feeds into the DeepHit feature matrix.

Usage:
    python biobert_finetune.py --train   # fine-tune from snorkel labels
    python biobert_finetune.py --infer   # score all unlabeled snippets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "biobert_finetuned"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "dmis-lab/biobert-base-cased-v1.2"
MAX_LEN = 256
BATCH_SIZE = 16
NUM_LABELS = 3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AchillesTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer,
        max_len: int = MAX_LEN,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    from sklearn.metrics import f1_score, accuracy_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_prodromal": f1_score(
            labels, preds, labels=[1], average="macro", zero_division=0
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    labels_csv: Path | None = None,
    output_dir: Path | None = None,
    num_epochs: int = 5,
    seed: int = 42,
) -> None:
    labels_csv = labels_csv or PROCESSED_DIR / "snorkel_labels.csv"
    output_dir = output_dir or MODEL_DIR

    df = pd.read_csv(labels_csv)
    # Drop abstains
    df = df[df["snorkel_label"] >= 0].copy()
    print(f"[biobert] {len(df):,} labeled snippets (abstains excluded)")

    texts = df["text"].tolist()
    labels = df["snorkel_label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=seed
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS
    )

    train_dataset = AchillesTextDataset(train_texts, train_labels, tokenizer)
    val_dataset = AchillesTextDataset(val_texts, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_prodromal",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[biobert] model saved → {output_dir}")

    # Validation report
    val_preds = np.argmax(trainer.predict(val_dataset).predictions, axis=-1)
    print(classification_report(val_labels, val_preds,
                                 target_names=["healthy", "prodromal", "rupture"]))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(
    text_csv: Path | None = None,
    model_dir: Path | None = None,
    output_csv: Path | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    text_csv = text_csv or PROCESSED_DIR / "injury_text_snippets.csv"
    model_dir = model_dir or MODEL_DIR
    output_csv = output_csv or PROCESSED_DIR / "biobert_scores.csv"

    df = pd.read_csv(text_csv)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = AchillesTextDataset(df["text"].tolist(), None, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    prob_matrix = np.vstack(all_probs)
    df["biobert_prob_healthy"] = prob_matrix[:, 0]
    df["biobert_prob_prodromal"] = prob_matrix[:, 1]
    df["biobert_prob_rupture"] = prob_matrix[:, 2]
    df["biobert_prodromal_score"] = df["biobert_prob_prodromal"]  # primary feature

    df.to_csv(output_csv, index=False)
    print(f"[biobert] scored {len(df):,} snippets → {output_csv}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    if args.train:
        train(output_dir=args.model_dir, num_epochs=args.epochs)
    if args.infer:
        infer(model_dir=args.model_dir)
    if not args.train and not args.infer:
        parser.print_help()


if __name__ == "__main__":
    main()
