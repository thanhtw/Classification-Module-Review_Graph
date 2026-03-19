import random
from collections import Counter
from typing import Dict, List, Sequence

try:
    import jieba  # type: ignore
except ModuleNotFoundError:
    jieba = None

import numpy as np
import pandas as pd
import torch

from .config import LABEL_COLUMNS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_and_clean_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8")
    required = ["text"] + LABEL_COLUMNS
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    data["text"] = (
        data["text"]
        .fillna("")
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.strip()
    )
    data = data[data["text"] != ""].copy()
    for col in LABEL_COLUMNS:
        data[col] = data[col].fillna(0).astype(int)
    return data.reset_index(drop=True)


def tokenize_text(text: str) -> List[str]:
    if jieba is not None:
        return list(jieba.cut(text))
    parts = text.split()
    return parts if parts else list(text)


def preprocess_for_tfidf(text: str) -> str:
    return " ".join(tokenize_text(text))


class VocabBuilder:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq

    def build(self, tokenized_texts: Sequence[Sequence[str]]) -> Dict[str, int]:
        freq = Counter()
        for toks in tokenized_texts:
            freq.update(toks)

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for token, count in freq.items():
            if count >= self.min_freq:
                vocab[token] = len(vocab)
        return vocab


def texts_to_sequences(tokenized_texts: Sequence[Sequence[str]], vocab: Dict[str, int], max_len: int) -> np.ndarray:
    unk = vocab["<UNK>"]
    result = []
    for toks in tokenized_texts:
        ids = [vocab.get(t, unk) for t in toks]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [0] * (max_len - len(ids))
        result.append(ids)
    return np.asarray(result)
