import os
from dataclasses import dataclass
from pathlib import Path

LABEL_COLUMNS = ["relevance", "concreteness", "constructive"]
AVAILABLE_MODELS = [
    "bert",
    "roberta",
    "linear_svm",
    "naive_bayes",
    "logistic_regression",
    "lstm",
    "bilstm",
    "llm_zero_shot",
    "llm_few_shot",
]


@dataclass
class CommonConfig:
    seed: int = 42
    test_size: float = 0.2
    use_smote: bool = True
    output_dir: str = "results/modular_models"


@dataclass
class RNNConfig:
    max_len: int = 200
    embedding_dim: int = 300
    hidden_dim: int = 128
    dropout: float = 0.3
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 3
    glove_path: str = ""
    glove_trainable: bool = True


@dataclass
class TransformerConfig:
    model_name: str
    max_len: int = 128
    batch_size: int = 16
    epochs: int = 30
    lr: float = 2e-5
    weight_decay: float = 1e-3


@dataclass
class LLMConfig:
    model_name: str = "llama-3.1-8b-instant"  # Faster, smaller model with good JSON support
    max_new_tokens: int = 128  # Increased for safety
    temperature: float = 0.0
    few_shot_k: int = 100


def load_env_file(env_path: str | Path = ".env") -> None:
    """Load KEY=VALUE pairs from .env into environment variables.

    Existing environment variables are not overwritten.
    """
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_env_int(name: str, default: int) -> int:
    """Read integer env var with fallback."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_env_float(name: str, default: float) -> float:
    """Read float env var with fallback."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
