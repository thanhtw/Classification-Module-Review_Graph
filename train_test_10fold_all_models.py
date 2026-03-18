import argparse
import json
import os
import pickle
import random
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

try:
    import jieba  # type: ignore
except ModuleNotFoundError:
    jieba = None
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from imblearn.over_sampling import RandomOverSampler
except ModuleNotFoundError:
    RandomOverSampler = None

LABEL_COLUMNS = ["relevance", "concreteness", "constructive"]
METRIC_COLUMNS = [
    "subset_accuracy",
    "accuracy_micro",
    "accuracy_macro",
    "hamming_score",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]
AVAILABLE_MODELS = ["bert", "roberta", "lstm", "bilstm", "svm", "decision_tree"]
DEEP_MODELS = {"bert", "roberta", "lstm", "bilstm"}
MACRO_MICRO_COMPARE_COLUMNS = [
    "accuracy_micro",
    "accuracy_macro",
    "precision_micro",
    "precision_macro",
    "recall_micro",
    "recall_macro",
    "f1_micro",
    "f1_macro",
]


def apply_paper_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F6F8FB",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": "#1F2937",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "grid.color": "#E5E7EB",
            "grid.linestyle": "--",
            "grid.alpha": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
        }
    )


def tokenize_text(text: str) -> List[str]:
    if jieba is not None:
        return list(jieba.cut(text))
    # Fallback: whitespace tokens if present, otherwise char-level tokens for CJK text.
    parts = text.split()
    return parts if parts else list(text)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")

    required_columns = ["text"] + LABEL_COLUMNS
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.strip()
    )
    df = df[df["text"] != ""].copy()

    for col in LABEL_COLUMNS:
        df[col] = df[col].fillna(0).astype(int)

    return df.reset_index(drop=True)


def encode_label_combinations(labels: np.ndarray) -> np.ndarray:
    combos = []
    n_labels = labels.shape[1]
    for row in labels:
        binary_str = "".join(str(int(v)) for v in row)
        combos.append(int(binary_str, 2))
    return np.asarray(combos, dtype=np.int64)


def decode_label_combinations(combo_labels: np.ndarray, n_labels: int) -> np.ndarray:
    decoded = np.zeros((len(combo_labels), n_labels), dtype=np.int64)
    for i, combo in enumerate(combo_labels):
        bits = format(int(combo), f"0{n_labels}b")
        decoded[i] = np.array([int(b) for b in bits], dtype=np.int64)
    return decoded


def smote_resample_features(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    feature_postprocess: str = "float",
    clip_min: int = 0,
    clip_max: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)

    combo_labels = encode_label_combinations(labels)
    unique, counts = np.unique(combo_labels, return_counts=True)
    min_count = int(np.min(counts))
    max_count = int(np.max(counts))

    stats = {
        "n_before": int(len(labels)),
        "n_after": int(len(labels)),
        "min_combo_before": min_count,
        "min_combo_after": min_count,
        "target_count_per_combo": max_count,
        "k_neighbors": 0,
        "applied": 0,
        "method": "none",
        "mlsmote_generated": 0,
    }

    if len(unique) < 2:
        return features, labels, stats

    # MLSMOTE stage: generate synthetic samples from tail-label minority instances.
    pos_counts = labels.sum(axis=0).astype(float)
    valid = pos_counts > 0
    if np.any(valid):
        max_pos = float(np.max(pos_counts[valid]))
        irpl = np.where(valid, max_pos / pos_counts, np.inf)
        mir = float(np.mean(irpl[np.isfinite(irpl)]))
        tail_idx = np.where((irpl > mir) & np.isfinite(irpl))[0]
        if len(tail_idx) == 0:
            tail_idx = np.where(valid)[0]
    else:
        tail_idx = np.array([], dtype=int)

    minority_mask = (labels[:, tail_idx] == 1).any(axis=1) if len(tail_idx) > 0 else np.zeros(len(labels), dtype=bool)
    minority_indices = np.where(minority_mask)[0]

    target_count = max_count
    needed = int(sum(target_count - int(c) for c in counts if int(c) < target_count))

    x_work = features.astype(np.float32, copy=True)
    y_work = labels.astype(np.int64, copy=True)
    mlsmote_generated = 0
    k_neighbors = 0
    method = "none"

    if needed > 0 and len(minority_indices) >= 2:
        x_min = x_work[minority_indices]
        y_min = y_work[minority_indices]
        k_neighbors = min(5, len(x_min) - 1)
        if k_neighbors >= 1:
            nbs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine", algorithm="brute").fit(x_min)
            _, knn_idx = nbs.kneighbors(x_min)

            syn_x = np.zeros((needed, x_work.shape[1]), dtype=np.float32)
            syn_y = np.zeros((needed, y_work.shape[1]), dtype=np.int64)
            vote_threshold = int(np.ceil((k_neighbors + 1) / 2.0))

            for i in range(needed):
                ref = int(rng.integers(0, len(x_min)))
                neigh_pool = knn_idx[ref, 1:]
                if len(neigh_pool) == 0:
                    neigh = ref
                else:
                    neigh = int(neigh_pool[int(rng.integers(0, len(neigh_pool)))])
                ratio = float(rng.random())
                syn_x[i] = x_min[ref] + ratio * (x_min[ref] - x_min[neigh])

                neighborhood = knn_idx[ref]
                votes = y_min[neighborhood].sum(axis=0)
                lab = (votes >= vote_threshold).astype(np.int64)
                if lab.sum() == 0:
                    lab = y_min[ref]
                syn_y[i] = lab

            x_work = np.vstack([x_work, syn_x])
            y_work = np.vstack([y_work, syn_y])
            mlsmote_generated = int(needed)
            method = "mlsmote"

    # Equalize 3-bit combo counts to the same target using ROS after MLSMOTE.
    combo_work = encode_label_combinations(y_work)
    uniq_w, cnt_w = np.unique(combo_work, return_counts=True)
    target_final = int(np.max(cnt_w)) if len(cnt_w) else max_count
    sampling_strategy = {
        int(combo): target_final
        for combo, count in zip(uniq_w, cnt_w)
        if int(count) < target_final
    }

    if sampling_strategy and RandomOverSampler is not None:
        ros = RandomOverSampler(random_state=seed, sampling_strategy=sampling_strategy)
        x_res, y_combo_res = ros.fit_resample(x_work, combo_work)
        method = "mlsmote+ros_equalize" if method != "none" else "random_oversampler_only"
    else:
        x_res = x_work
        y_combo_res = combo_work

    if feature_postprocess == "int":
        x_res = np.rint(x_res).astype(np.int64)
        x_res = np.clip(x_res, clip_min, clip_max)

    y_res = decode_label_combinations(np.asarray(y_combo_res), labels.shape[1]).astype(np.int64)
    _, counts_after = np.unique(y_combo_res, return_counts=True)

    stats = {
        "n_before": int(len(labels)),
        "n_after": int(len(y_res)),
        "min_combo_before": min_count,
        "min_combo_after": int(np.min(counts_after)),
        "target_count_per_combo": int(np.max(counts_after)),
        "k_neighbors": int(k_neighbors),
        "applied": int(len(y_res) > len(labels)),
        "method": method,
        "mlsmote_generated": int(mlsmote_generated),
    }
    return x_res, y_res, stats


def compute_label_distribution(labels: np.ndarray, split_name: str) -> pd.DataFrame:
    rows = []
    total = len(labels)
    for i, col in enumerate(LABEL_COLUMNS):
        pos = int(labels[:, i].sum())
        neg = int(total - pos)
        rows.append(
            {
                "split": split_name,
                "label": col,
                "positive_count": pos,
                "negative_count": neg,
                "positive_ratio": float(pos / total if total else 0.0),
            }
        )
    return pd.DataFrame(rows)


def compute_combo_distribution(labels: np.ndarray, split_name: str) -> pd.DataFrame:
    combos = encode_label_combinations(labels)
    unique, counts = np.unique(combos, return_counts=True)
    rows = []
    total = len(labels)
    for combo, count in zip(unique, counts):
        combo_bin = format(int(combo), f"0{labels.shape[1]}b")
        rows.append(
            {
                "split": split_name,
                "combo_int": int(combo),
                "combo_bin": combo_bin,
                "count": int(count),
                "ratio": float(count / total if total else 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def export_smote_dataset_analysis(texts: Sequence[str], labels: np.ndarray, output_dir: str, seed: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    combo_def_rows = []
    for combo in range(2 ** len(LABEL_COLUMNS)):
        bits = format(combo, f"0{len(LABEL_COLUMNS)}b")
        combo_def_rows.append(
            {
                "combo_int": combo,
                "combo_bin": bits,
                "bit1_relevance": int(bits[0]),
                "bit2_concreteness": int(bits[1]),
                "bit3_constructive": int(bits[2]),
            }
        )
    pd.DataFrame(combo_def_rows).to_csv(
        os.path.join(output_dir, "label_combo_definition.csv"),
        index=False,
        encoding="utf-8",
    )

    # Use TF-IDF space to create a reproducible before/after SMOTE dataset report.
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_for_tfidf,
        token_pattern=None,
        max_features=5000,
        ngram_range=(1, 2),
    )
    x = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    x_smote, y_smote, smote_stats = smote_resample_features(x, labels, seed=seed)

    label_before = compute_label_distribution(labels, "before_smote")
    label_after = compute_label_distribution(y_smote, "after_smote")
    combo_before = compute_combo_distribution(labels, "before_smote")
    combo_after = compute_combo_distribution(y_smote, "after_smote")

    label_before.to_csv(os.path.join(output_dir, "label_distribution_before_smote.csv"), index=False, encoding="utf-8")
    label_after.to_csv(os.path.join(output_dir, "label_distribution_after_smote.csv"), index=False, encoding="utf-8")
    combo_before.to_csv(os.path.join(output_dir, "label_combo_before_smote.csv"), index=False, encoding="utf-8")
    combo_after.to_csv(os.path.join(output_dir, "label_combo_after_smote.csv"), index=False, encoding="utf-8")

    with open(os.path.join(output_dir, "smote_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(smote_stats, f, ensure_ascii=False, indent=2)

    apply_paper_style()
    merged = label_before.merge(label_after, on="label", suffixes=("_before", "_after"))

    plt.figure(figsize=(12, 5))
    x_idx = np.arange(len(LABEL_COLUMNS))
    w = 0.35
    plt.subplot(1, 2, 1)
    plt.bar(x_idx - w / 2, merged["positive_count_before"], width=w, color="#2563EB", label="Before")
    plt.bar(x_idx + w / 2, merged["positive_count_after"], width=w, color="#14B8A6", label="After")
    plt.xticks(x_idx, merged["label"], rotation=20)
    plt.title("Positive Label Counts")
    plt.ylabel("Count")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(x_idx - w / 2, merged["positive_ratio_before"], width=w, color="#2563EB", label="Before")
    plt.bar(x_idx + w / 2, merged["positive_ratio_after"], width=w, color="#14B8A6", label="After")
    plt.xticks(x_idx, merged["label"], rotation=20)
    plt.title("Positive Label Ratios")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paper_label_distribution_before_after_smote.png"), dpi=300)
    plt.close()

    combo_union = pd.concat([
        combo_before[["combo_bin", "count"]].assign(source="before"),
        combo_after[["combo_bin", "count"]].assign(source="after"),
    ])
    top_combos = (
        combo_union.groupby("combo_bin", as_index=False)["count"].sum()
        .sort_values("count", ascending=False)
        .head(10)["combo_bin"]
        .tolist()
    )
    before_top = combo_before.set_index("combo_bin").reindex(top_combos).fillna(0)
    after_top = combo_after.set_index("combo_bin").reindex(top_combos).fillna(0)

    plt.figure(figsize=(12, 5))
    idx = np.arange(len(top_combos))
    plt.bar(idx - w / 2, before_top["count"], width=w, color="#2563EB", label="Before")
    plt.bar(idx + w / 2, after_top["count"], width=w, color="#14B8A6", label="After")
    plt.xticks(idx, top_combos, rotation=30)
    plt.ylabel("Count")
    plt.title("Top Label Combination Distribution Before/After SMOTE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paper_label_combo_before_after_smote.png"), dpi=300)
    plt.close()


def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for t, p in zip(y_true, y_pred):
        union = np.logical_or(t, p).sum()
        if union == 0:
            scores.append(1.0)
        else:
            scores.append(np.logical_and(t, p).sum() / union)
    return float(np.mean(scores))


def micro_macro_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    # Micro accuracy: flatten all label decisions.
    micro = float((y_true == y_pred).mean())

    # Macro accuracy: per-label binary accuracy then averaged.
    per_label_acc = []
    for i in range(y_true.shape[1]):
        per_label_acc.append(float((y_true[:, i] == y_pred[:, i]).mean()))
    macro = float(np.mean(per_label_acc))
    return micro, macro


def score_from_metrics(metrics: Dict[str, float]) -> float:
    return float(np.mean([metrics[c] for c in MACRO_MICRO_COMPARE_COLUMNS]))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc_micro, acc_macro = micro_macro_accuracy(y_true, y_pred)
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_micro": acc_micro,
        "accuracy_macro": acc_macro,
        "hamming_score": float(hamming_score(y_true, y_pred)),
        "precision_micro": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "recall_micro": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def make_result_row(
    model_name: str,
    fold_idx: int,
    split: str,
    metrics: Dict[str, float],
    train_time_sec: float,
    inference_time_sec: float,
    n_samples: int,
) -> Dict[str, float]:
    row = {
        "model": model_name,
        "fold": fold_idx,
        "split": split,
        "train_time_sec": train_time_sec,
        "inference_time_sec": inference_time_sec,
        "n_samples": n_samples,
    }
    row.update(metrics)
    return row


def tokenize_for_tfidf(text: str) -> List[str]:
    return tokenize_text(text)


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.x = torch.LongTensor(sequences)
        self.y = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TextRNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        bidirectional: bool,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        pooled = out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


class VocabBuilder:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.pad = "<PAD>"
        self.unk = "<UNK>"

    def build(self, tokenized_texts: Sequence[Sequence[str]]) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for tokens in tokenized_texts:
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1

        vocab = {self.pad: 0, self.unk: 1}
        for token, count in freq.items():
            if count >= self.min_freq:
                vocab[token] = len(vocab)
        return vocab


class HFDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, labels: np.ndarray):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
        weight = self.pos_weight.to(logits.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=weight,
        )
        return (loss, outputs) if return_outputs else loss


@dataclass
class RNNConfig:
    max_len: int
    embedding_dim: int
    hidden_dim: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float


@dataclass
class TransformerConfig:
    model_name: str
    max_len: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float


def texts_to_sequences(
    tokenized_texts: Sequence[Sequence[str]],
    vocab: Dict[str, int],
    max_len: int,
) -> np.ndarray:
    unk_id = vocab["<UNK>"]
    output = []
    for tokens in tokenized_texts:
        ids = [vocab.get(t, unk_id) for t in tokens]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [0] * (max_len - len(ids))
        output.append(ids)
    return np.array(output)


def predict_rnn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    model.eval()
    all_logits = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_logits.append(logits.cpu().numpy())
    infer_time = time.perf_counter() - start
    logits = np.vstack(all_logits)
    preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    return preds, infer_time


def run_lstm_family_fold(
    model_label: str,
    bidirectional: bool,
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    config: RNNConfig,
    seed: int,
    use_smote: bool,
    save_dir: str = "",
) -> Tuple[Dict[str, float], Dict[str, float], float, float, float]:
    set_seed(seed)

    tokenized_train = [tokenize_text(t) for t in train_texts]
    tokenized_test = [tokenize_text(t) for t in test_texts]

    vocab = VocabBuilder(min_freq=2).build(tokenized_train)
    x_train = texts_to_sequences(tokenized_train, vocab, config.max_len)
    x_test = texts_to_sequences(tokenized_test, vocab, config.max_len)

    x_train_res, y_train_res = x_train, train_labels
    if use_smote:
        x_train_res, y_train_res, _ = smote_resample_features(
            features=x_train.astype(np.float32),
            labels=train_labels,
            seed=seed,
            feature_postprocess="int",
            clip_min=0,
            clip_max=len(vocab) - 1,
        )
        y_train_res = y_train_res.astype(np.int64)

    train_ds = SequenceDataset(x_train_res, y_train_res)
    test_ds = SequenceDataset(x_test, test_labels)
    train_orig_ds = SequenceDataset(x_train, train_labels)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    eval_train_loader = DataLoader(train_orig_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextRNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_labels=len(LABEL_COLUMNS),
        bidirectional=bidirectional,
        dropout=config.dropout,
    ).to(device)

    pos = y_train_res.sum(axis=0)
    neg = len(y_train_res) - pos
    pos_weight = torch.FloatTensor((neg / (pos + 1e-6)).astype(np.float32)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_start = time.perf_counter()
    model.train()
    for _ in range(config.epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    train_time = time.perf_counter() - train_start

    train_preds, train_infer = predict_rnn(model, eval_train_loader, device)
    test_preds, test_infer = predict_rnn(model, test_loader, device)

    train_metrics = compute_metrics(train_labels, train_preds)
    test_metrics = compute_metrics(test_labels, test_preds)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        artifact = {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "config": {
                "model_label": model_label,
                "bidirectional": bidirectional,
                "max_len": config.max_len,
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "dropout": config.dropout,
                "label_columns": LABEL_COLUMNS,
            },
        }
        torch.save(artifact, os.path.join(save_dir, "model.pt"))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_metrics, test_metrics, train_time, train_infer, test_infer


def _extract_logits(pred_output) -> np.ndarray:
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return np.asarray(logits)


def run_transformer_fold(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    config: TransformerConfig,
    seed: int,
    output_dir: str,
    use_smote: bool,
    save_dir: str = "",
) -> Tuple[Dict[str, float], Dict[str, float], float, float, float]:
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
    )

    train_enc = tokenizer(
        list(train_texts),
        truncation=True,
        padding=True,
        max_length=config.max_len,
    )
    test_enc = tokenizer(
        list(test_texts),
        truncation=True,
        padding=True,
        max_length=config.max_len,
    )

    train_input_ids = np.asarray(train_enc["input_ids"])
    train_attention = np.asarray(train_enc["attention_mask"])

    train_input_ids_res, train_attention_res, train_labels_res = train_input_ids, train_attention, train_labels
    if use_smote:
        combined = np.concatenate([train_input_ids, train_attention], axis=1).astype(np.float32)
        combined_res, train_labels_res, _ = smote_resample_features(
            features=combined,
            labels=train_labels,
            seed=seed,
            feature_postprocess="float",
        )
        seq_len = train_input_ids.shape[1]
        train_input_ids_res = np.clip(np.rint(combined_res[:, :seq_len]), 0, tokenizer.vocab_size - 1).astype(np.int64)
        train_attention_res = np.clip(np.rint(combined_res[:, seq_len:]), 0, 1).astype(np.int64)
        train_labels_res = train_labels_res.astype(np.int64)

    train_ds = HFDataset(
        train_input_ids_res,
        train_attention_res,
        train_labels_res,
    )
    train_orig_ds = HFDataset(
        train_input_ids,
        train_attention,
        train_labels,
    )
    test_ds = HFDataset(
        np.asarray(test_enc["input_ids"]),
        np.asarray(test_enc["attention_mask"]),
        test_labels,
    )

    pos = train_labels_res.sum(axis=0)
    neg = len(train_labels_res) - pos
    pos_weight = torch.FloatTensor((neg / (pos + 1e-6)).astype(np.float32))

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
        use_cpu=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedBCETrainer(
        pos_weight=pos_weight,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    train_start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - train_start

    train_pred_start = time.perf_counter()
    train_pred_output = trainer.predict(train_orig_ds)
    train_infer = time.perf_counter() - train_pred_start

    test_pred_start = time.perf_counter()
    test_pred_output = trainer.predict(test_ds)
    test_infer = time.perf_counter() - test_pred_start

    train_logits = _extract_logits(train_pred_output)
    test_logits = _extract_logits(test_pred_output)

    train_preds = (1 / (1 + np.exp(-train_logits)) > 0.5).astype(int)
    test_preds = (1 / (1 + np.exp(-test_logits)) > 0.5).astype(int)

    train_metrics = compute_metrics(train_labels, train_preds)
    test_metrics = compute_metrics(test_labels, test_preds)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": config.model_name,
                    "max_len": config.max_len,
                    "label_columns": LABEL_COLUMNS,
                    "threshold": 0.5,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_metrics, test_metrics, train_time, train_infer, test_infer


def run_traditional_fold(
    model_name: str,
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    tfidf_max_features: int,
    seed: int,
    use_smote: bool,
    save_dir: str = "",
) -> Tuple[Dict[str, float], Dict[str, float], float, float, float]:
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_for_tfidf,
        token_pattern=None,
        max_features=tfidf_max_features,
        ngram_range=(1, 2),
    )

    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    x_train_fit = x_train
    y_train_fit = train_labels
    x_train_eval = x_train
    x_test_eval = x_test
    if use_smote:
        x_res, y_res, _ = smote_resample_features(
            features=x_train.toarray().astype(np.float32),
            labels=train_labels,
            seed=seed,
            feature_postprocess="float",
        )
        x_train_fit = x_res
        y_train_fit = y_res
        x_train_eval = x_train.toarray().astype(np.float32)
        x_test_eval = x_test.toarray().astype(np.float32)

    if model_name == "svm":
        base = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    elif model_name == "decision_tree":
        base = DecisionTreeClassifier(
            random_state=seed,
            class_weight="balanced",
            max_depth=30,
        )
    else:
        raise ValueError(f"Unsupported traditional model: {model_name}")

    model = MultiOutputClassifier(base)

    train_start = time.perf_counter()
    model.fit(x_train_fit, y_train_fit)
    train_time = time.perf_counter() - train_start

    train_pred_start = time.perf_counter()
    train_preds = model.predict(x_train_eval)
    train_infer = time.perf_counter() - train_pred_start

    test_pred_start = time.perf_counter()
    test_preds = model.predict(x_test_eval)
    test_infer = time.perf_counter() - test_pred_start

    train_metrics = compute_metrics(train_labels, train_preds)
    test_metrics = compute_metrics(test_labels, test_preds)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "vectorizer": vectorizer,
                    "label_columns": LABEL_COLUMNS,
                },
                f,
            )

    return train_metrics, test_metrics, train_time, train_infer, test_infer


def export_visualizations(fold_df: pd.DataFrame, output_dir: str, best_model: str) -> None:
    apply_paper_style()
    test_df = fold_df[fold_df["split"] == "test"].copy()
    stats_df = (
        test_df.groupby("model", as_index=False)
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", lambda x: float(np.std(x, ddof=0))),
            subset_accuracy_mean=("subset_accuracy", "mean"),
            subset_accuracy_std=("subset_accuracy", lambda x: float(np.std(x, ddof=0))),
            precision_macro_mean=("precision_macro", "mean"),
            recall_macro_mean=("recall_macro", "mean"),
            inference_time_mean=("inference_time_sec", "mean"),
            train_time_mean=("train_time_sec", "mean"),
        )
        .sort_values("f1_macro_mean", ascending=False)
    )

    colors = {
        "bert": "#1D4ED8",
        "roberta": "#0F766E",
        "lstm": "#C2410C",
        "bilstm": "#7C3AED",
        "svm": "#B45309",
        "decision_tree": "#475569",
    }

    plt.figure(figsize=(12, 6))
    x = np.arange(len(stats_df))
    width = 0.38
    bars1 = plt.bar(
        x - width / 2,
        stats_df["f1_macro_mean"],
        width,
        yerr=stats_df["f1_macro_std"],
        capsize=4,
        color="#2563EB",
        edgecolor="#1E3A8A",
        label="F1-macro",
    )
    bars2 = plt.bar(
        x + width / 2,
        stats_df["subset_accuracy_mean"],
        width,
        yerr=stats_df["subset_accuracy_std"],
        capsize=4,
        color="#14B8A6",
        edgecolor="#115E59",
        label="Subset Accuracy",
    )
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, stats_df["model"], rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paper_model_performance_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    model_order = stats_df["model"].tolist()
    data_for_box = [test_df[test_df["model"] == m]["f1_macro"].values for m in model_order]
    box = plt.boxplot(
        data_for_box,
        tick_labels=model_order,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "#DBEAFE", "color": "#1E40AF"},
        medianprops={"color": "#B91C1C", "linewidth": 1.5},
        whiskerprops={"color": "#1E40AF"},
        capprops={"color": "#1E40AF"},
        meanprops={"marker": "o", "markerfacecolor": "#111827", "markeredgecolor": "#111827"},
    )
    for patch, m in zip(box["boxes"], model_order):
        patch.set_facecolor(colors.get(m, "#CBD5E1"))
        patch.set_alpha(0.35)
    plt.xticks(rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1-macro")
    plt.title("F1-macro Distribution Across Splits/Folds (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paper_model_f1_distribution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6), facecolor="#F6F8FB")
    for _, row in stats_df.iterrows():
        color = "#DC2626" if row["model"] == best_model else colors.get(row["model"], "#334155")
        bubble_size = max(100, row["train_time_mean"] * 8)
        plt.scatter(
            row["inference_time_mean"],
            row["f1_macro_mean"],
            s=bubble_size,
            c=color,
            alpha=0.85,
            edgecolors="#111827",
            linewidths=0.7,
        )
        plt.annotate(
            row["model"],
            (row["inference_time_mean"], row["f1_macro_mean"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    plt.xlabel("Average Inference Time (sec)")
    plt.ylabel("Average F1-macro (test)")
    plt.title("Performance vs Inference Efficiency (bubble size = train time)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paper_efficiency_tradeoff.png"), dpi=300)
    plt.close()

    # Heatmap-like matrix for paper appendix.
    metric_matrix = stats_df[["model", "f1_macro_mean", "subset_accuracy_mean", "precision_macro_mean", "recall_macro_mean"]].copy()
    matrix_values = metric_matrix.drop(columns=["model"]).values
    plt.figure(figsize=(9, 5))
    im = plt.imshow(matrix_values, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Score")
    plt.xticks(
        ticks=np.arange(matrix_values.shape[1]),
        labels=["F1-macro", "Subset Acc", "Precision-macro", "Recall-macro"],
        rotation=20,
    )
    plt.yticks(ticks=np.arange(len(metric_matrix)), labels=metric_matrix["model"])
    for i in range(matrix_values.shape[0]):
        for j in range(matrix_values.shape[1]):
            plt.text(j, i, f"{matrix_values[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#111827")
    plt.title("Test Metric Matrix by Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paper_metric_matrix.png"), dpi=300)
    plt.close()


def export_results(rows: List[Dict[str, float]], output_dir: str, best_artifact_paths: Dict[str, str]) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fold_df = pd.DataFrame(rows)
    fold_path = os.path.join(output_dir, "all_models_10fold_train_test_metrics.csv")
    fold_df.to_csv(fold_path, index=False, encoding="utf-8")

    # Mean and std per model/split for each paper metric.
    summary_parts = []
    for (model, split), grp in fold_df.groupby(["model", "split"]):
        summary = {"model": model, "split": split}
        for col in METRIC_COLUMNS + ["train_time_sec", "inference_time_sec"]:
            summary[f"{col}_mean"] = float(grp[col].mean())
            summary[f"{col}_std"] = float(grp[col].std(ddof=0))
        summary_parts.append(summary)

    summary_df = pd.DataFrame(summary_parts)
    summary_path = os.path.join(output_dir, "all_models_summary_mean_std.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    # Paper-ready table on test split with mean +- std formatting.
    test_summary = summary_df[summary_df["split"] == "test"].copy()
    paper_table = pd.DataFrame({"model": test_summary["model"]})
    for metric in ["subset_accuracy", "f1_micro", "f1_macro", "precision_macro", "recall_macro"]:
        paper_table[metric] = test_summary.apply(
            lambda r: f"{r[f'{metric}_mean']:.4f} +- {r[f'{metric}_std']:.4f}", axis=1
        )

    paper_csv_path = os.path.join(output_dir, "paper_test_table.csv")
    paper_table.to_csv(paper_csv_path, index=False, encoding="utf-8")

    paper_tex_path = os.path.join(output_dir, "paper_test_table.tex")
    with open(paper_tex_path, "w", encoding="utf-8") as f:
        f.write(paper_table.to_latex(index=False, escape=False))

    # Compare the best fold of each model using macro/micro Acc/Prec/Rec/F1.
    test_rows = fold_df[fold_df["split"] == "test"].copy()
    best_fold_per_model = (
        test_rows.sort_values(
            ["model", "f1_macro", "f1_micro", "accuracy_macro", "accuracy_micro"],
            ascending=[True, False, False, False, False],
        )
        .drop_duplicates(subset=["model"], keep="first")
        .reset_index(drop=True)
    )
    best_fold_per_model["macro_micro_score"] = best_fold_per_model.apply(
        lambda r: float(np.mean([r[c] for c in MACRO_MICRO_COMPARE_COLUMNS])), axis=1
    )
    best_fold_per_model = best_fold_per_model.sort_values(
        ["macro_micro_score", "f1_macro", "f1_micro"], ascending=False
    ).reset_index(drop=True)

    best_compare_cols = [
        "model",
        "fold",
        "subset_accuracy",
        "accuracy_micro",
        "accuracy_macro",
        "precision_micro",
        "precision_macro",
        "recall_micro",
        "recall_macro",
        "f1_micro",
        "f1_macro",
        "macro_micro_score",
    ]
    best_model_compare_path = os.path.join(output_dir, "best_per_model_comparison.csv")
    best_fold_per_model[best_compare_cols].to_csv(best_model_compare_path, index=False, encoding="utf-8")

    best_model_compare_tex = os.path.join(output_dir, "best_per_model_comparison.tex")
    with open(best_model_compare_tex, "w", encoding="utf-8") as f:
        f.write(best_fold_per_model[best_compare_cols].to_latex(index=False, float_format="%.4f", escape=False))

    best_row = best_fold_per_model.iloc[0]
    best_model = str(best_row["model"])
    best_artifact = best_artifact_paths.get(best_model, "")

    if best_artifact and os.path.exists(best_artifact):
        selected_dir = os.path.join(output_dir, "best_model_artifact")
        if os.path.exists(selected_dir):
            shutil.rmtree(selected_dir)
        shutil.copytree(best_artifact, selected_dir)
        best_artifact = selected_dir

    best_info = {
        "best_model": best_model,
        "selection_rule": "best fold per model, then highest mean of macro/micro accuracy+precision+recall+f1",
        "best_fold_of_best_model": int(best_row["fold"]),
        "metrics": {
            "subset_accuracy": float(best_row["subset_accuracy"]),
            "accuracy_micro": float(best_row["accuracy_micro"]),
            "accuracy_macro": float(best_row["accuracy_macro"]),
            "precision_micro": float(best_row["precision_micro"]),
            "precision_macro": float(best_row["precision_macro"]),
            "recall_micro": float(best_row["recall_micro"]),
            "recall_macro": float(best_row["recall_macro"]),
            "f1_micro": float(best_row["f1_micro"]),
            "f1_macro": float(best_row["f1_macro"]),
            "macro_micro_score": float(best_row["macro_micro_score"]),
        },
        "artifact_path": best_artifact,
    }
    best_info_path = os.path.join(output_dir, "best_model_selection.json")
    with open(best_info_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    export_visualizations(fold_df, output_dir, best_model)

    print("\nExport complete:")
    print(f"1) Per-fold metrics: {fold_path}")
    print(f"2) Mean/std summary: {summary_path}")
    print(f"3) Paper test table (CSV): {paper_csv_path}")
    print(f"4) Paper test table (LaTeX): {paper_tex_path}")
    print(f"5) Best model selection: {best_info_path}")
    print(f"6) Best-per-model comparison: {best_model_compare_path}")
    print(f"7) Best-per-model comparison LaTeX: {best_model_compare_tex}")
    print("8) Paper figures:")
    print(f"   - {os.path.join(output_dir, 'paper_model_performance_comparison.png')}")
    print(f"   - {os.path.join(output_dir, 'paper_model_f1_distribution.png')}")
    print(f"   - {os.path.join(output_dir, 'paper_efficiency_tradeoff.png')}")
    print(f"   - {os.path.join(output_dir, 'paper_metric_matrix.png')}")
    print("9) Dataset analysis (before/after SMOTE):")
    print(f"   - {os.path.join(output_dir, 'dataset_analysis/label_combo_definition.csv')}")
    print(f"   - {os.path.join(output_dir, 'dataset_analysis/label_distribution_before_smote.csv')}")
    print(f"   - {os.path.join(output_dir, 'dataset_analysis/label_distribution_after_smote.csv')}")
    print(f"   - {os.path.join(output_dir, 'dataset_analysis/label_combo_before_smote.csv')}")
    print(f"   - {os.path.join(output_dir, 'dataset_analysis/label_combo_after_smote.csv')}")
    print(f"   - {os.path.join(output_dir, 'dataset_analysis/paper_label_distribution_before_after_smote.png')}")
    print(f"   - {os.path.join(output_dir, 'dataset_analysis/paper_label_combo_before_after_smote.png')}")
    print(f"\nRecommended solution model: {best_model}")
    if best_artifact:
        print(f"Reusable inference artifact: {best_artifact}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="10-fold train/test benchmark for BERT, RoBERTa, LSTM, BiLSTM, SVM, Decision Tree"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/cleaned_3label_data.csv",
        help="Path to cleaned_3label_data.csv",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=AVAILABLE_MODELS,
        choices=AVAILABLE_MODELS,
        help="Models to run",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="kfold",
        choices=["holdout", "kfold"],
        help="holdout uses 8:2 style train/test split; kfold uses cross-validation",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="test ratio for holdout split")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/all_models_10fold")
    parser.add_argument(
        "--use_smote",
        action="store_true",
        default=True,
        help="Apply MLSMOTE-based resampling on each training split to handle imbalanced data",
    )
    parser.add_argument(
        "--no_smote",
        action="store_false",
        dest="use_smote",
        help="Disable MLSMOTE-based resampling",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow deep models to run on CPU when GPU is unavailable",
    )

    parser.add_argument("--tfidf_max_features", type=int, default=10000)

    parser.add_argument("--rnn_max_len", type=int, default=200)
    parser.add_argument("--rnn_embedding_dim", type=int, default=100)
    parser.add_argument("--rnn_hidden_dim", type=int, default=128)
    parser.add_argument("--rnn_dropout", type=float, default=0.3)
    parser.add_argument("--rnn_batch_size", type=int, default=32)
    parser.add_argument("--rnn_epochs", type=int, default=8)
    parser.add_argument("--rnn_lr", type=float, default=1e-3)

    parser.add_argument("--bert_max_len", type=int, default=256)
    parser.add_argument("--bert_batch_size", type=int, default=8)
    parser.add_argument("--bert_epochs", type=int, default=3)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--bert_weight_decay", type=float, default=1e-3)

    parser.add_argument("--roberta_max_len", type=int, default=256)
    parser.add_argument("--roberta_batch_size", type=int, default=8)
    parser.add_argument("--roberta_epochs", type=int, default=3)
    parser.add_argument("--roberta_lr", type=float, default=2e-5)
    parser.add_argument("--roberta_weight_decay", type=float, default=1e-3)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    deep_requested = any(m in DEEP_MODELS for m in args.models)
    if deep_requested and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "GPU is required for deep models (BERT/RoBERTa/LSTM/BiLSTM). "
            "Use a CUDA-enabled environment or pass --allow_cpu."
        )
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU unavailable: deep models will run on CPU due to --allow_cpu")

    if args.use_smote and RandomOverSampler is None:
        raise ModuleNotFoundError(
            "MLSMOTE pipeline needs imbalanced-learn for equalization fallback. Run: pip install imbalanced-learn"
        )

    df = load_and_clean_data(args.data_path)
    texts = df["text"].tolist()
    labels = df[LABEL_COLUMNS].values.astype(int)

    analysis_dir = os.path.join(args.output_dir, "dataset_analysis")
    export_smote_dataset_analysis(texts, labels, analysis_dir, seed=args.seed)

    indices = np.arange(len(texts))
    if args.split_mode == "holdout":
        train_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.seed,
            shuffle=True,
        )
        splits = [(np.asarray(train_idx), np.asarray(test_idx))]
    else:
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(kf.split(texts))

    rnn_cfg = RNNConfig(
        max_len=args.rnn_max_len,
        embedding_dim=args.rnn_embedding_dim,
        hidden_dim=args.rnn_hidden_dim,
        dropout=args.rnn_dropout,
        batch_size=args.rnn_batch_size,
        epochs=args.rnn_epochs,
        lr=args.rnn_lr,
    )

    bert_cfg = TransformerConfig(
        model_name="bert-base-chinese",
        max_len=args.bert_max_len,
        batch_size=args.bert_batch_size,
        epochs=args.bert_epochs,
        lr=args.bert_lr,
        weight_decay=args.bert_weight_decay,
    )

    roberta_cfg = TransformerConfig(
        model_name="hfl/chinese-roberta-wwm-ext",
        max_len=args.roberta_max_len,
        batch_size=args.roberta_batch_size,
        epochs=args.roberta_epochs,
        lr=args.roberta_lr,
        weight_decay=args.roberta_weight_decay,
    )

    all_rows: List[Dict[str, float]] = []
    best_by_model = {m: {"score": -1.0, "artifact": ""} for m in args.models}

    for model_name in args.models:
        print("\n" + "=" * 70)
        print(f"Running model: {model_name}")
        print("=" * 70)

        total_splits = len(splits)
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            split_name = "Split" if args.split_mode == "holdout" else "Fold"
            print(f"\n{split_name} {fold_idx}/{total_splits}")

            train_texts = [texts[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]
            artifact_dir = os.path.join(args.output_dir, "artifacts", model_name, f"fold_{fold_idx}")

            fold_seed = args.seed + fold_idx

            if model_name == "svm" or model_name == "decision_tree":
                (
                    train_metrics,
                    test_metrics,
                    train_time,
                    train_infer,
                    test_infer,
                ) = run_traditional_fold(
                    model_name=model_name,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    test_texts=test_texts,
                    test_labels=test_labels,
                    tfidf_max_features=args.tfidf_max_features,
                    seed=fold_seed,
                    use_smote=args.use_smote,
                    save_dir=artifact_dir,
                )
            elif model_name == "lstm":
                (
                    train_metrics,
                    test_metrics,
                    train_time,
                    train_infer,
                    test_infer,
                ) = run_lstm_family_fold(
                    model_label="LSTM",
                    bidirectional=False,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    test_texts=test_texts,
                    test_labels=test_labels,
                    config=rnn_cfg,
                    seed=fold_seed,
                    use_smote=args.use_smote,
                    save_dir=artifact_dir,
                )
            elif model_name == "bilstm":
                (
                    train_metrics,
                    test_metrics,
                    train_time,
                    train_infer,
                    test_infer,
                ) = run_lstm_family_fold(
                    model_label="BiLSTM",
                    bidirectional=True,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    test_texts=test_texts,
                    test_labels=test_labels,
                    config=rnn_cfg,
                    seed=fold_seed,
                    use_smote=args.use_smote,
                    save_dir=artifact_dir,
                )
            elif model_name == "bert":
                fold_output = os.path.join(args.output_dir, "temp", "bert", f"fold_{fold_idx}")
                (
                    train_metrics,
                    test_metrics,
                    train_time,
                    train_infer,
                    test_infer,
                ) = run_transformer_fold(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    test_texts=test_texts,
                    test_labels=test_labels,
                    config=bert_cfg,
                    seed=fold_seed,
                    output_dir=fold_output,
                    use_smote=args.use_smote,
                    save_dir=artifact_dir,
                )
            elif model_name == "roberta":
                fold_output = os.path.join(args.output_dir, "temp", "roberta", f"fold_{fold_idx}")
                (
                    train_metrics,
                    test_metrics,
                    train_time,
                    train_infer,
                    test_infer,
                ) = run_transformer_fold(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    test_texts=test_texts,
                    test_labels=test_labels,
                    config=roberta_cfg,
                    seed=fold_seed,
                    output_dir=fold_output,
                    use_smote=args.use_smote,
                    save_dir=artifact_dir,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            current_score = score_from_metrics(test_metrics)
            if current_score > best_by_model[model_name]["score"]:
                best_by_model[model_name]["score"] = current_score
                best_by_model[model_name]["artifact"] = artifact_dir

            print(
                f"Train F1-macro: {train_metrics['f1_macro']:.4f} | "
                f"Test F1-macro: {test_metrics['f1_macro']:.4f}"
            )

            all_rows.append(
                make_result_row(
                    model_name=model_name,
                    fold_idx=fold_idx,
                    split="train",
                    metrics=train_metrics,
                    train_time_sec=train_time,
                    inference_time_sec=train_infer,
                    n_samples=len(train_idx),
                )
            )
            all_rows.append(
                make_result_row(
                    model_name=model_name,
                    fold_idx=fold_idx,
                    split="test",
                    metrics=test_metrics,
                    train_time_sec=train_time,
                    inference_time_sec=test_infer,
                    n_samples=len(test_idx),
                )
            )

    best_artifact_paths = {
        model: meta["artifact"] for model, meta in best_by_model.items() if meta["artifact"]
    }
    export_results(all_rows, args.output_dir, best_artifact_paths)


if __name__ == "__main__":
    main()
