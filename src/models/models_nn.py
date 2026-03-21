import time
import json
import os
import copy
import gzip
from typing import Dict, Sequence, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.config import CNNConfig, RNNConfig
from src.data.preprocessor import VocabBuilder, texts_to_sequences, tokenize_text
from src.utils.metrics import compute_metrics
from src.utils.smote import apply_smote_multilabel


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.LongTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    """
    LSTM/BiLSTM Classifier for multi-label text classification.
    
    Architecture:
    - LSTM: Embedding -> LSTM -> Mean Pooling -> Dense -> Sigmoid
    - BiLSTM: Embedding -> BiLSTM -> Mean Pooling -> Dense -> Sigmoid
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        n_labels: int,
        bidirectional: bool,
        dropout: float,
        embeddings: np.ndarray | None = None,
        embedding_trainable: bool = True,
    ):
        super().__init__()
        # Embedding layer (initialized with GloVe or random)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings.astype(np.float32)))
        self.embedding.weight.requires_grad = embedding_trainable
        
        # LSTM/BiLSTM layer
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Dense layer -> output logits (sigmoid applied via BCEWithLogitsLoss)
        self.fc = nn.Linear(out_dim, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Embedding -> LSTM -> Mean Pooling -> Dropout -> Dense
        
        Args:
            x: Input sequence of token IDs (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, n_labels) - sigmoid applied in loss function
        """
        # Step 1: Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, emb_dim)
        
        # Step 2: LSTM/BiLSTM
        lstm_out, (hidden, cell) = self.rnn(embedded)  # (batch_size, seq_len, hidden_dim*2 or hidden_dim)
        
        # Step 3: Mean pooling over time dimension
        pooled = lstm_out.mean(dim=1)  # (batch_size, hidden_dim*2 or hidden_dim)
        
        # Step 4: Dropout
        pooled = self.dropout(pooled)
        
        # Step 5: Dense layer (logits)
        logits = self.fc(pooled)  # (batch_size, n_labels)
        
        return logits


class CNNAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        n_filters: int,
        filter_sizes: Sequence[int],
        n_labels: int,
        dropout: float,
        embeddings: np.ndarray | None = None,
        embedding_trainable: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings.astype(np.float32)))
        self.embedding.weight.requires_grad = embedding_trainable
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k) for k in filter_sizes])
        att_dim = n_filters * len(filter_sizes)
        self.attn = nn.Linear(att_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(att_dim, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        conv_outs = [torch.relu(c(emb)) for c in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]
        feat = torch.cat(pooled, dim=1)
        weight = torch.softmax(self.attn(feat.unsqueeze(1)), dim=1)
        att_feat = (weight * feat.unsqueeze(1)).squeeze(1)
        att_feat = self.dropout(att_feat)
        return self.fc(att_feat)


def _train_eval(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    weight_decay: float = 0.0,
    scheduler_patience: int = 2,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 3,
    save_dir: str = "",
) -> Tuple[Dict[str, float], float, float, nn.Module, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n = len(x_train)
    if n >= 10:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        val_n = max(1, int(round(0.1 * n)))
        tr_idx = perm[val_n:]
        va_idx = perm[:val_n]
    else:
        tr_idx = np.arange(n)
        va_idx = np.arange(n)

    x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
    x_val, y_val = x_train[va_idx], y_train[va_idx]

    train_loader = DataLoader(SeqDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    pos = y_tr.sum(axis=0)
    neg = len(y_tr) - pos
    pos_weight = torch.FloatTensor((neg / (pos + 1e-6)).astype(np.float32)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=scheduler_patience,
        factor=scheduler_factor,
    )

    train_start = time.perf_counter()
    model.train()
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    no_improve = 0
    
    # Initialize history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1_macro': [],
        'val_f1_macro': [],
        'train_f1_micro': [],
        'val_f1_micro': []
    }

    for epoch in range(epochs):
        running_loss = 0.0
        train_logits = []
        train_labels_collected = []
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item())
            
            # Collect for metrics computation
            train_logits.append(logits.detach().cpu().numpy())
            train_labels_collected.append(by.cpu().numpy())

        avg_train_loss = running_loss / max(1, len(train_loader))
        history['train_loss'].append(avg_train_loss)
        
        # Compute training metrics
        if train_logits:
            train_logits_all = np.vstack(train_logits)
            train_labels_all = np.vstack(train_labels_collected)
            train_preds = (1 / (1 + np.exp(-train_logits_all)) > 0.5).astype(int)
            train_metrics = compute_metrics(train_labels_all, train_preds)
            history['train_f1_macro'].append(train_metrics.get('f1_macro_mean', 0))
            history['train_f1_micro'].append(train_metrics.get('f1_micro_mean', 0))

        # Keep training policy consistent with biLstmGlove.py.
        if len(train_loader) > 0:
            scheduler.step(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_logits = []
        val_labels_collected = []
        
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                val_loss += float(criterion(logits, by).item())
                
                # Collect for metrics computation
                val_logits.append(logits.cpu().numpy())
                val_labels_collected.append(by.cpu().numpy())
                
        val_loss = val_loss / max(1, len(val_loader))
        history['val_loss'].append(val_loss)
        
        # Compute validation metrics
        if val_logits:
            val_logits_all = np.vstack(val_logits)
            val_labels_all = np.vstack(val_labels_collected)
            val_preds = (1 / (1 + np.exp(-val_logits_all)) > 0.5).astype(int)
            val_metrics = compute_metrics(val_labels_all, val_preds)
            history['val_f1_macro'].append(val_metrics.get('f1_macro_mean', 0))
            history['val_f1_micro'].append(val_metrics.get('f1_micro_mean', 0))
        
        model.train()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                break

    model.load_state_dict(best_state)
    train_time = time.perf_counter() - train_start

    model.eval()
    infer_start = time.perf_counter()
    all_logits = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            all_logits.append(model(bx).cpu().numpy())
    infer_time = time.perf_counter() - infer_start

    logits = np.vstack(all_logits)
    preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    
    # Save history if save_dir provided
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        history_file = Path(save_dir) / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    return compute_metrics(y_test, preds), train_time, infer_time, model, preds, y_test


def _prepare_seq_data(train_texts: Sequence[str], test_texts: Sequence[str], max_len: int):
    tok_train = [tokenize_text(t) for t in train_texts]
    tok_test = [tokenize_text(t) for t in test_texts]
    vocab = VocabBuilder(min_freq=2).build(tok_train)
    x_train = texts_to_sequences(tok_train, vocab, max_len)
    x_test = texts_to_sequences(tok_test, vocab, max_len)
    return x_train, x_test, vocab


def _load_glove_embeddings(vocab: Dict[str, int], embedding_dim: int, glove_path: str) -> Tuple[np.ndarray, Dict[str, object]]:
    emb = np.random.normal(0, 0.1, (len(vocab), embedding_dim)).astype(np.float32)
    emb[0] = 0.0
    loaded = 0
    used_path = ""

    if glove_path and os.path.exists(glove_path):
        open_fn = gzip.open if glove_path.endswith(".gz") else open
        with open_fn(glove_path, "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split()
                if len(parts) <= embedding_dim:
                    continue
                word = parts[0]
                if word not in vocab:
                    continue
                try:
                    vec = np.asarray(parts[1 : 1 + embedding_dim], dtype=np.float32)
                except ValueError:
                    continue
                if vec.shape[0] != embedding_dim:
                    continue
                emb[vocab[word]] = vec
                loaded += 1
        used_path = glove_path

    return emb, {
        "embedding_dim": int(embedding_dim),
        "glove_path": used_path,
        "glove_loaded_tokens": int(loaded),
        "vocab_size": int(len(vocab)),
    }


def run_lstm_like(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    cfg: RNNConfig,
    bidirectional: bool,
    use_smote: bool,
    seed: int,
    save_dir: str = "",
) -> Tuple[Dict[str, float], float, float]:
    x_train, x_test, vocab = _prepare_seq_data(train_texts, test_texts, cfg.max_len)
    y_train = train_labels
    embeddings, glove_info = _load_glove_embeddings(vocab, cfg.embedding_dim, cfg.glove_path)

    smote_stats = {"applied": 0, "method": "disabled"}
    if use_smote:
        x_train, y_train, smote_stats = apply_smote_multilabel(
            x_train.astype(np.float32),
            y_train,
            seed=seed,
            postprocess="int",
            clip_min=0,
            clip_max=len(vocab) - 1,
        )

    model = LSTMClassifier(
        vocab_size=len(vocab),
        emb_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        n_labels=train_labels.shape[1],
        bidirectional=bidirectional,
        dropout=cfg.dropout,
        embeddings=embeddings,
        embedding_trainable=cfg.glove_trainable,
    )
    metrics, train_time, infer_time, model, preds, y_test_returned = _train_eval(
        model,
        x_train,
        y_train,
        x_test,
        test_labels,
        cfg.batch_size,
        cfg.epochs,
        cfg.lr,
        seed,
        cfg.weight_decay,
        cfg.scheduler_patience,
        cfg.scheduler_factor,
        cfg.early_stopping_patience,
        save_dir,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Save predictions and labels for confusion matrix calculation
        np.save(os.path.join(save_dir, "predictions.npy"), preds)
        np.save(os.path.join(save_dir, "labels.npy"), y_test_returned)
        
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab": vocab,
                "model_name": "bilstm" if bidirectional else "lstm",
                "config": {
                    "max_len": cfg.max_len,
                    "embedding_dim": cfg.embedding_dim,
                    "hidden_dim": cfg.hidden_dim,
                    "dropout": cfg.dropout,
                    "bidirectional": bidirectional,
                    "weight_decay": cfg.weight_decay,
                    "scheduler_patience": cfg.scheduler_patience,
                    "scheduler_factor": cfg.scheduler_factor,
                    "early_stopping_patience": cfg.early_stopping_patience,
                    "glove_trainable": cfg.glove_trainable,
                    "glove_info": glove_info,
                },
            },
            os.path.join(save_dir, "model.pt"),
        )
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "bilstm" if bidirectional else "lstm",
                    "use_smote": bool(use_smote),
                    "smote_stats": smote_stats,
                    "train_size_before": int(len(train_labels)),
                    "train_size_after": int(len(y_train)),
                    "glove_info": glove_info,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return metrics, train_time, infer_time


def run_lstm_like_from_sequences(
    x_train: np.ndarray,
    train_labels: np.ndarray,
    x_test: np.ndarray,
    test_labels: np.ndarray,
    vocab: Dict[str, int],
    cfg: RNNConfig,
    bidirectional: bool,
    seed: int,
    save_dir: str = "",
    smote_stats: Dict[str, object] | None = None,
) -> Tuple[Dict[str, float], float, float]:
    embeddings, glove_info = _load_glove_embeddings(vocab, cfg.embedding_dim, cfg.glove_path)

    model = LSTMClassifier(
        vocab_size=len(vocab),
        emb_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        n_labels=train_labels.shape[1],
        bidirectional=bidirectional,
        dropout=cfg.dropout,
        embeddings=embeddings,
        embedding_trainable=cfg.glove_trainable,
    )
    metrics, train_time, infer_time, model, preds, y_test_returned = _train_eval(
        model,
        x_train,
        train_labels,
        x_test,
        test_labels,
        cfg.batch_size,
        cfg.epochs,
        cfg.lr,
        seed,
        cfg.weight_decay,
        cfg.scheduler_patience,
        cfg.scheduler_factor,
        cfg.early_stopping_patience,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Save predictions and labels for confusion matrix calculation
        np.save(os.path.join(save_dir, "predictions.npy"), preds)
        np.save(os.path.join(save_dir, "labels.npy"), y_test_returned)
        
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab": vocab,
                "model_name": "bilstm" if bidirectional else "lstm",
                "config": {
                    "max_len": cfg.max_len,
                    "embedding_dim": cfg.embedding_dim,
                    "hidden_dim": cfg.hidden_dim,
                    "dropout": cfg.dropout,
                    "bidirectional": bidirectional,
                    "weight_decay": cfg.weight_decay,
                    "scheduler_patience": cfg.scheduler_patience,
                    "scheduler_factor": cfg.scheduler_factor,
                    "early_stopping_patience": cfg.early_stopping_patience,
                    "glove_trainable": cfg.glove_trainable,
                    "glove_info": glove_info,
                },
            },
            os.path.join(save_dir, "model.pt"),
        )
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "bilstm" if bidirectional else "lstm",
                    "use_smote": bool(smote_stats and smote_stats.get("applied", 0)),
                    "smote_stats": smote_stats if smote_stats is not None else {"applied": 0, "method": "disabled"},
                    "train_size_before": int(len(train_labels)),
                    "train_size_after": int(len(train_labels)),
                    "glove_info": glove_info,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return metrics, train_time, infer_time


def run_cnn_attention(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    cfg: CNNConfig,
    use_smote: bool,
    seed: int,
    save_dir: str = "",
) -> Tuple[Dict[str, float], float, float]:
    x_train, x_test, vocab = _prepare_seq_data(train_texts, test_texts, cfg.max_len)
    y_train = train_labels
    embeddings, glove_info = _load_glove_embeddings(vocab, cfg.embedding_dim, cfg.glove_path)

    smote_stats = {"applied": 0, "method": "disabled"}
    if use_smote:
        x_train, y_train, smote_stats = apply_smote_multilabel(
            x_train.astype(np.float32),
            y_train,
            seed=seed,
            postprocess="int",
            clip_min=0,
            clip_max=len(vocab) - 1,
        )

    model = CNNAttentionClassifier(
        vocab_size=len(vocab),
        emb_dim=cfg.embedding_dim,
        n_filters=cfg.num_filters,
        filter_sizes=cfg.filter_sizes,
        n_labels=train_labels.shape[1],
        dropout=cfg.dropout,
        embeddings=embeddings,
        embedding_trainable=cfg.glove_trainable,
    )
    metrics, train_time, infer_time, model, preds, y_test_returned = _train_eval(
        model,
        x_train,
        y_train,
        x_test,
        test_labels,
        cfg.batch_size,
        cfg.epochs,
        cfg.lr,
        seed,
        cfg.weight_decay,
        cfg.scheduler_patience,
        cfg.scheduler_factor,
        cfg.early_stopping_patience,
        save_dir,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Save predictions and labels for confusion matrix calculation
        np.save(os.path.join(save_dir, "predictions.npy"), preds)
        np.save(os.path.join(save_dir, "labels.npy"), y_test_returned)
        
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab": vocab,
                "model_name": "cnn_attention",
                "config": {
                    "max_len": cfg.max_len,
                    "embedding_dim": cfg.embedding_dim,
                    "num_filters": cfg.num_filters,
                    "filter_sizes": list(cfg.filter_sizes),
                    "dropout": cfg.dropout,
                    "weight_decay": cfg.weight_decay,
                    "scheduler_patience": cfg.scheduler_patience,
                    "scheduler_factor": cfg.scheduler_factor,
                    "early_stopping_patience": cfg.early_stopping_patience,
                    "glove_trainable": cfg.glove_trainable,
                    "glove_info": glove_info,
                },
            },
            os.path.join(save_dir, "model.pt"),
        )
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "cnn_attention",
                    "use_smote": bool(use_smote),
                    "smote_stats": smote_stats,
                    "train_size_before": int(len(train_labels)),
                    "train_size_after": int(len(y_train)),
                    "glove_info": glove_info,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return metrics, train_time, infer_time
