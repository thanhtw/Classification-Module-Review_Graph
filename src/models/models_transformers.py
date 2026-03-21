import time
import json
import os
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

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


class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight.to(logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def _extract_logits(pred_output) -> np.ndarray:
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return np.asarray(logits)


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

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        save_strategy="no",
        report_to="none",
        disable_tqdm=False,
        dataloader_pin_memory=False,
        use_cpu=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedBCETrainer(
        pos_weight=pos_weight,
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
    )

    tr_start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - tr_start

    val_pred_out = trainer.predict(ds_val)
    val_logits = _extract_logits(val_pred_out)
    val_probs = 1 / (1 + np.exp(-val_logits))
    tuned_thresholds = tune_per_label_thresholds(y_train_res[va_idx].astype(int), val_probs)

    inf_start = time.perf_counter()
    pred_out = trainer.predict(ds_test)
    infer_time = time.perf_counter() - inf_start

    logits = _extract_logits(pred_out)
    probs = 1 / (1 + np.exp(-logits))
    preds = apply_per_label_thresholds(probs, tuned_thresholds)
    metrics = compute_metrics(test_labels, preds)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(save_dir)
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, train_time, infer_time
