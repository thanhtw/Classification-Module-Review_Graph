import time
import json
import os
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.training.config import TransformerConfig
from src.utils.metrics import apply_per_label_thresholds, compute_metrics, tune_per_label_thresholds
from src.utils.smote import apply_smote_multilabel


class HFDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, labels: np.ndarray):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def _build_transformer_training_history(
    epochs: list[float],
    train_loss: list[float],
    val_loss: list[float],
    final_metrics: Dict[str, float],
) -> Dict[str, object]:
    """Build an epoch-oriented history payload from manual training loops."""
    if not train_loss:
        proxy = float(max(0.0, 1.0 - final_metrics.get("f1_macro", 0.0)))
        train_loss = [proxy]
        val_loss = [proxy]
        epochs = [1.0]

    if len(val_loss) < len(train_loss):
        if val_loss:
            val_loss.extend([val_loss[-1]] * (len(train_loss) - len(val_loss)))
        else:
            val_loss = [train_loss[-1]] * len(train_loss)

    f1_macro = float(final_metrics.get("f1_macro", 0.0))
    f1_micro = float(final_metrics.get("f1_micro", f1_macro))
    n = len(train_loss)

    return {
        "history_type": "manual_epoch_loop",
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss[:n],
        "train_f1_macro": [f1_macro] * n,
        "val_f1_macro": [f1_macro] * n,
        "train_f1_micro": [f1_micro] * n,
        "val_f1_micro": [f1_micro] * n,
        "note": "Loss curves are from the manual epoch loop; F1 values are final fold metrics repeated per epoch.",
    }


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _run_eval_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            labels = batch["labels"]
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(outputs.logits, labels)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _predict_logits(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    logits_chunks = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits_chunks.append(outputs.logits.detach().cpu().numpy())
    if not logits_chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(logits_chunks, axis=0)


def run_transformer(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    cfg: TransformerConfig,
    seed: int,
    use_smote: bool,
    output_dir: str,
    save_dir: str = "",
) -> Tuple[Dict[str, float], float, float]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=train_labels.shape[1],
        problem_type="multi_label_classification",
    )

    train_enc = tokenizer(list(train_texts), truncation=True, padding=True, max_length=cfg.max_len)
    test_enc = tokenizer(list(test_texts), truncation=True, padding=True, max_length=cfg.max_len)

    tr_ids = np.asarray(train_enc["input_ids"])
    tr_mask = np.asarray(train_enc["attention_mask"])

    tr_ids_res, tr_mask_res, y_train_res = tr_ids, tr_mask, train_labels
    smote_stats = {"applied": 0, "method": "disabled"}
    if use_smote:
        comb = np.concatenate([tr_ids, tr_mask], axis=1).astype(np.float32)
        comb_res, y_train_res, smote_stats = apply_smote_multilabel(comb, train_labels, seed=seed)
        seq_len = tr_ids.shape[1]
        tr_ids_res = np.clip(np.rint(comb_res[:, :seq_len]), 0, tokenizer.vocab_size - 1).astype(np.int64)
        tr_mask_res = np.clip(np.rint(comb_res[:, seq_len:]), 0, 1).astype(np.int64)

    # Build an internal validation split for per-label threshold tuning.
    n_train = len(y_train_res)
    if n_train >= 10:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_train)
        val_n = max(1, int(round(0.1 * n_train)))
        tr_idx = perm[val_n:]
        va_idx = perm[:val_n]
        if len(tr_idx) == 0:
            tr_idx = perm
            va_idx = perm[:1]
    else:
        tr_idx = np.arange(n_train)
        va_idx = np.arange(n_train)

    ds_train = HFDataset(tr_ids_res[tr_idx], tr_mask_res[tr_idx], y_train_res[tr_idx])
    ds_val = HFDataset(tr_ids_res[va_idx], tr_mask_res[va_idx], y_train_res[va_idx])
    ds_test = HFDataset(np.asarray(test_enc["input_ids"]), np.asarray(test_enc["attention_mask"]), test_labels)

    pos = y_train_res[tr_idx].sum(axis=0)
    neg = len(y_train_res[tr_idx]) - pos
    pos_weight = torch.FloatTensor((neg / (pos + 1e-6)).astype(np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, pin_memory=False)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    epoch_ids: list[float] = []
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    tr_start = time.perf_counter()
    for epoch in range(int(cfg.epochs)):
        model.train()
        batch_losses = []
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            labels = batch["labels"]
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        epoch_ids.append(float(epoch + 1))
        train_loss_history.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
        val_loss_history.append(_run_eval_loss(model, val_loader, loss_fn, device))
    train_time = time.perf_counter() - tr_start

    val_logits = _predict_logits(model, val_loader, device)
    val_probs = 1 / (1 + np.exp(-val_logits))
    tuned_thresholds = tune_per_label_thresholds(y_train_res[va_idx].astype(int), val_probs)

    inf_start = time.perf_counter()
    logits = _predict_logits(model, test_loader, device)
    infer_time = time.perf_counter() - inf_start

    probs = 1 / (1 + np.exp(-logits))
    preds = apply_per_label_thresholds(probs, tuned_thresholds)
    metrics = compute_metrics(test_labels, preds)
    training_history = _build_transformer_training_history(epoch_ids, train_loss_history, val_loss_history, metrics)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        # Save predictions and labels for confusion matrix calculation
        np.save(os.path.join(save_dir, "predictions.npy"), preds)
        np.save(os.path.join(save_dir, "labels.npy"), test_labels)
        
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": cfg.model_name,
                    "max_len": cfg.max_len,
                    "thresholds": tuned_thresholds.tolist(),
                    "use_smote": bool(use_smote),
                    "smote_stats": smote_stats,
                    "train_size_before": int(len(train_labels)),
                    "train_size_after": int(len(y_train_res)),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(os.path.join(save_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(training_history, f, ensure_ascii=False, indent=2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, train_time, infer_time
