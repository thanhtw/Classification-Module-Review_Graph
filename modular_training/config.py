from dataclasses import dataclass

LABEL_COLUMNS = ["relevance", "concreteness", "constructive"]
AVAILABLE_MODELS = [
    "bert",
    "roberta",
    "linear_svm",
    "naive_bayes",
    "logistic_regression",
    "cnn_attention",
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
class CNNConfig:
    max_len: int = 200
    embedding_dim: int = 300
    num_filters: int = 128
    filter_sizes: tuple = (3, 4, 5)
    dropout: float = 0.3
    batch_size: int = 32
    epochs: int = 5
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
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 1e-3


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    max_new_tokens: int = 64
    temperature: float = 0.0
    few_shot_k: int = 3
