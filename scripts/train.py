from __future__ import annotations

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm.auto import tqdm

from src.analysis.analysis_utils import (
    export_constructiveness_evaluation_reliability,
    export_train_smote_analysis,
)
from src.training.config import (
    AVAILABLE_MODELS,
    CommonConfig,
    LABEL_COLUMNS,
    LLMConfig,
    RNNConfig,
    TransformerConfig,
    get_env_float,
    get_env_int,
    get_env_str,
    load_env_file,
)
from src.data.preprocessor import load_and_clean_data, set_seed
try:
    from src.models.models_llm import run_llm_zero_few_shot
except ModuleNotFoundError as e:
    run_llm_zero_few_shot = None
    # Optionally print a warning, but do not break ML-only workflows
    if 'openai' in str(e):
        print("[WARN] openai module not found: LLM functionality will be unavailable. ML/SMOTE tests are unaffected.")
    else:
        raise
from src.models.models_nn import run_lstm_like
from src.models.models_ml import run_linear_svm, run_naive_bayes, run_logistic_regression
from src.models.models_transformers import run_transformer
from src.utils.reporting import (
    export_combined_final_test_predictions,
    export_fold_result,
    export_results,
    export_test_predictions,
)


def parse_args() -> argparse.Namespace:
    load_env_file(project_root / ".env")

    parser = argparse.ArgumentParser(
        description="Train/test modular models (BERT, RoBERTa, LSTM/BiLSTM, LinearSVM, NaiveBayes, LogisticRegression, optional LLM) on cleaned_3label_data.csv"
    )
    parser.add_argument("--data_path", type=str, default="data/cleaned_3label_data.csv")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear_svm", "logistic_regression", "naive_bayes", "lstm", "bilstm", "bert", "roberta"],
        choices=AVAILABLE_MODELS,
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Used only when --n_folds <= 1")
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of folds within the 80% development split; use 1 to skip inner CV",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/modular_multimodel")
    parser.add_argument(
        "--append_results",
        action="store_true",
        help="Merge this run into existing results so model families can be run in separate sessions",
    )
    parser.add_argument("--no_smote", action="store_true", help="Disable SMOTE on training split")
    parser.add_argument(
        "--smote_target_ratio",
        type=float,
        default=1.0,
        help="Target minority/majority ratio in (0, 1] for label-preserving training-fold resampling",
    )

    parser.add_argument("--rnn_epochs", type=int, default=get_env_int("TRAIN_RNN_EPOCHS", 30))
    parser.add_argument("--bert_epochs", type=int, default=get_env_int("TRAIN_BERT_EPOCHS", 30))
    parser.add_argument("--roberta_epochs", type=int, default=get_env_int("TRAIN_ROBERTA_EPOCHS", 30))
    parser.add_argument("--ml_epochs", type=int, default=get_env_int("TRAIN_ML_EPOCHS", 30))
    parser.add_argument("--rnn_lr", type=float, default=get_env_float("TRAIN_RNN_LR", 1e-3))
    parser.add_argument("--bert_lr", type=float, default=get_env_float("TRAIN_BERT_LR", 2e-5))
    parser.add_argument("--roberta_lr", type=float, default=get_env_float("TRAIN_ROBERTA_LR", 2e-5))
    parser.add_argument("--glove_path", type=str, default="", help="Path to pretrained word vectors text file")
    parser.add_argument("--freeze_glove", action="store_true", help="Freeze embedding layer initialized by pretrained vectors")

    parser.add_argument("--bert_model_name", type=str, default="bert-base-chinese", help="Hugging Face model id for BERT")
    parser.add_argument(
        "--roberta_model_name",
        type=str,
        default="hfl/chinese-roberta-wwm-ext",
        help="Hugging Face model id for RoBERTa",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=get_env_str("HF_MODEL_CACHE_DIR", ".cache/huggingface/hub"),
        help="Persistent Hugging Face cache; model files download only when missing",
    )

    parser.add_argument(
        "--llm_model_name",
        type=str,
        default=get_env_str("OPENAI_LLM_MODEL_NAME", "gpt-5.6-luna"),
        help="OpenAI model name for llm_zero_shot/llm_few_shot",
    )
    parser.add_argument("--llm_few_shot_k", type=int, default=10, help="Number of few-shot examples to include in each prompt")
    parser.add_argument(
        "--llm_max_new_tokens",
        type=int,
        default=get_env_int("OPENAI_LLM_MAX_TOKENS", 512),
        help="Initial completion-token budget; token-limit failures retry once with a larger budget",
    )
    parser.add_argument("--llm_temperature", type=float, default=0.0, help="Sampling temperature for LLM decoding")
    parser.add_argument(
        "--llm_reasoning_effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default=get_env_str("OPENAI_LLM_REASONING_EFFORT", "none"),
        help="Reasoning effort for GPT-5.6; classification defaults to none",
    )
    return parser.parse_args()


def _make_folds(
    n_samples: int,
    n_folds: int,
    test_size: float,
    seed: int,
    labels: Optional[np.ndarray] = None,
) -> List[Dict[str, np.ndarray]]:
    idx_local = np.arange(n_samples)
    fold_list: List[Dict[str, np.ndarray]] = []
    strat_targets = None
    if labels is not None:
        # Stratify by multilabel combination (e.g., "101") to reduce fold imbalance.
        strat_targets = np.array(["".join(row.astype(str).tolist()) for row in labels])

    if n_folds >= 2:
        if strat_targets is not None:
            try:
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                for tr_idx, te_idx in skf.split(idx_local, strat_targets):
                    fold_list.append({"train_idx": tr_idx, "test_idx": te_idx})
                return fold_list
            except ValueError:
                # Fallback when some combinations are too rare for n_splits.
                pass

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(idx_local):
            fold_list.append({"train_idx": tr_idx, "test_idx": te_idx})
    else:
        holdout_stratify = strat_targets if strat_targets is not None else None
        try:
            tr_idx, te_idx = train_test_split(
                idx_local,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=holdout_stratify,
            )
        except ValueError:
            # Fallback when stratification is not feasible due to rare classes.
            tr_idx, te_idx = train_test_split(
                idx_local,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
            )
        fold_list.append({"train_idx": tr_idx, "test_idx": te_idx})
    return fold_list


def _export_fold_splits(df, folds: List[Dict[str, np.ndarray]], output_dir: str) -> None:
    """Persist the exact train/test rows used by every model in each fold."""
    split_root = Path(output_dir) / "splits"
    for fold_id, fold_data in enumerate(folds, start=1):
        fold_dir = split_root / f"fold_{fold_id:02d}"
        train_dir = fold_dir / "train"
        test_dir = fold_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        train_df = df.iloc[fold_data["train_idx"]].copy()
        test_df = df.iloc[fold_data["test_idx"]].copy()
        train_df.insert(0, "source_index", train_df.index.astype(int))
        test_df.insert(0, "source_index", test_df.index.astype(int))
        train_df.to_csv(train_dir / "data.csv", index=False, encoding="utf-8")
        test_df.to_csv(test_dir / "data.csv", index=False, encoding="utf-8")

        split_metadata = {
            "fold": fold_id,
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "train_file": "train/data.csv",
            "test_file": "test/data.csv",
            "shared_by_all_models": True,
        }
        (fold_dir / "metadata.json").write_text(
            json.dumps(split_metadata, indent=2), encoding="utf-8"
        )


def _is_usable_huggingface_snapshot(snapshot_path: str) -> bool:
    """Return whether a snapshot has config, tokenizer assets, and PyTorch weights."""
    snapshot = Path(snapshot_path)
    has_config = (snapshot / "config.json").is_file()
    has_tokenizer = any(
        (snapshot / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "spiece.model")
    )
    has_weights = any(snapshot.glob("*.safetensors")) or any(
        snapshot.glob("pytorch_model*.bin")
    )
    return has_config and has_tokenizer and has_weights


def _resolve_huggingface_checkpoint(model_id: str, cache_dir: str) -> str:
    """Resolve a complete local snapshot, downloading only required PyTorch files."""
    from huggingface_hub import snapshot_download

    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_path),
            local_files_only=True,
        )
        if not _is_usable_huggingface_snapshot(snapshot_path):
            raise FileNotFoundError("cached snapshot is incomplete")
        print(f"Using cached Hugging Face checkpoint: {model_id} -> {snapshot_path}")
    except (OSError, ValueError):
        print(f"Checkpoint not cached; downloading PyTorch files from Hugging Face: {model_id}")
        common_files = [
            "config.json",
            "tokenizer*",
            "vocab*",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
            "*.model",
        ]
        # Prefer safetensors. Unlike an unrestricted snapshot download, this
        # avoids fetching duplicate PyTorch .bin weights and TF/Flax artifacts.
        snapshot_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_path),
            local_files_only=False,
            allow_patterns=[*common_files, "*.safetensors", "*.safetensors.index.json"],
        )
        if not _is_usable_huggingface_snapshot(snapshot_path):
            # Older checkpoints may only publish pytorch_model.bin (possibly sharded).
            snapshot_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_path),
                local_files_only=False,
                allow_patterns=[*common_files, "pytorch_model*.bin", "pytorch_model.bin.index.json"],
            )
        if not _is_usable_huggingface_snapshot(snapshot_path):
            raise RuntimeError(
                f"Downloaded checkpoint '{model_id}' is incomplete: expected config, "
                "tokenizer files, and PyTorch weights."
            )
        print(f"Checkpoint cached for future folds and runs: {snapshot_path}")
    return str(Path(snapshot_path).resolve())


def main() -> None:
    args = parse_args()
    if not 0.0 < args.smote_target_ratio <= 1.0:
        raise ValueError("--smote_target_ratio must be in the interval (0, 1]")
    default_zh_vec_path = os.path.join("embeddings", "cc.zh.300.vec.gz")
    if not args.glove_path and os.path.exists(default_zh_vec_path):
        args.glove_path = default_zh_vec_path
        print(f"Using local Chinese pretrained vectors: {args.glove_path}")

    requested_models = list(args.models)
    supported_in_runner = {
        "bert",
        "roberta",
        "linear_svm",
        "naive_bayes",
        "logistic_regression",
        "lstm",
        "bilstm",
        "llm_zero_shot",
        "llm_few_shot",
    }
    models_to_run = [m for m in requested_models if m in supported_in_runner]
    if not models_to_run:
        raise ValueError(
            "No model supported by this runner was selected. "
            "Use --models with any of: bert roberta linear_svm naive_bayes logistic_regression lstm bilstm llm_zero_shot llm_few_shot"
        )
    if any(m in {"llm_zero_shot", "llm_few_shot"} for m in models_to_run) and run_llm_zero_few_shot is None:
        raise ModuleNotFoundError(
            "LLM models were requested, but the OpenAI dependency is unavailable. "
            "Install the `openai` package before running llm_zero_shot or llm_few_shot."
        )

    common = CommonConfig(seed=args.seed, test_size=args.test_size, use_smote=(not args.no_smote), output_dir=args.output_dir)
    set_seed(common.seed)
    os.makedirs(common.output_dir, exist_ok=True)

    df = load_and_clean_data(args.data_path)
    texts = df["text"].tolist()
    labels = df[LABEL_COLUMNS].values.astype(int)

    rnn_cfg = RNNConfig(
        epochs=args.rnn_epochs,
        lr=args.rnn_lr,
        embedding_dim=300,
        glove_path=args.glove_path,
        glove_trainable=(not args.freeze_glove),
    )
    bert_cfg = TransformerConfig(model_name=args.bert_model_name, epochs=args.bert_epochs, lr=args.bert_lr)
    roberta_cfg = TransformerConfig(model_name=args.roberta_model_name, epochs=args.roberta_epochs, lr=args.roberta_lr)
    if "bert" in models_to_run:
        bert_cfg.local_model_path = _resolve_huggingface_checkpoint(
            bert_cfg.model_name, args.hf_cache_dir
        )
    if "roberta" in models_to_run:
        roberta_cfg.local_model_path = _resolve_huggingface_checkpoint(
            roberta_cfg.model_name, args.hf_cache_dir
        )
    llm_cfg = LLMConfig(
        model_name=args.llm_model_name,
        max_new_tokens=args.llm_max_new_tokens,
        temperature=args.llm_temperature,
        few_shot_k=args.llm_few_shot_k,
        reasoning_effort=args.llm_reasoning_effort,
    )

    # Reserve one untouched test set first. Cross-validation is performed only
    # inside the remaining development/training portion.
    holdout = _make_folds(len(texts), 1, common.test_size, common.seed, labels=labels)[0]
    development_idx = holdout["train_idx"]
    final_test_idx = holdout["test_idx"]
    development_labels = labels[development_idx]
    inner_folds = _make_folds(
        len(development_idx),
        args.n_folds,
        common.test_size,
        common.seed,
        labels=development_labels,
    )
    folds = [
        {
            "train_idx": development_idx[fold["train_idx"]],
            "test_idx": development_idx[fold["test_idx"]],
        }
        for fold in inner_folds
    ]

    analysis_dir = os.path.join(common.output_dir, "global_train_data_analysis")
    export_train_smote_analysis(
        train_texts=[texts[i] for i in development_idx],
        train_labels=labels[development_idx],
        output_dir=analysis_dir,
        seed=common.seed,
        use_smote=bool(common.use_smote),
        target_ratio=args.smote_target_ratio,
        full_labels=labels,
    )
    _export_fold_splits(df, folds, common.output_dir)

    protocol_split_dir = Path(common.output_dir) / "splits" / "final_holdout"
    (protocol_split_dir / "train_80").mkdir(parents=True, exist_ok=True)
    (protocol_split_dir / "test_20").mkdir(parents=True, exist_ok=True)
    df.iloc[development_idx].to_csv(protocol_split_dir / "train_80" / "data.csv", index=True, index_label="source_index")
    df.iloc[final_test_idx].to_csv(protocol_split_dir / "test_20" / "data.csv", index=True, index_label="source_index")
    (protocol_split_dir / "metadata.json").write_text(
        json.dumps(
            {
                "train_samples": int(len(development_idx)),
                "test_samples": int(len(final_test_idx)),
                "test_size": float(common.test_size),
                "inner_cv_folds": int(args.n_folds),
                "test_set_used_during_cross_validation": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Total samples: {len(texts)}")
    print(f"Protocol split: {len(development_idx)} development/train | {len(final_test_idx)} final test")
    print(f"Validation mode: {'cross-validation within development split' if args.n_folds >= 2 else 'single development holdout'}")
    if args.n_folds >= 2:
        print(f"Folds: {args.n_folds}")
    else:
        print(f"Holdout test_size: {common.test_size}")
    print(f"Training-fold resampling enabled: {bool(common.use_smote)}")
    print(f"Resampling target minority/majority ratio: {args.smote_target_ratio:.2f}")
    if any(m in {"llm_zero_shot", "llm_few_shot"} for m in models_to_run):
        print(f"LLM backend model: {llm_cfg.model_name}")

    rows: List[Dict[str, float]] = []
    process_records: List[Dict[str, object]] = []
    artifacts_root = os.path.join(common.output_dir, "model_artifacts")
    os.makedirs(artifacts_root, exist_ok=True)

    smote_allowed_models = {"linear_svm", "naive_bayes", "logistic_regression", "bert", "roberta"}
    total_runs = len(models_to_run) * len(folds)
    overall_pbar = tqdm(total=total_runs, desc="Evaluation progress", unit="fold")

    try:
        for raw_name in models_to_run:
            model_name = raw_name

            for fold_id, fold_data in enumerate(folds, start=1):
                overall_pbar.set_postfix(model=raw_name, fold=f"{fold_id}/{len(folds)}")
                train_idx = fold_data["train_idx"]
                test_idx = fold_data["test_idx"]

                train_texts = [texts[i] for i in train_idx]
                test_texts = [texts[i] for i in test_idx]
                y_train = labels[train_idx]
                y_test = labels[test_idx]

                print("\n" + "=" * 60)
                print(f"Running: {raw_name} | Fold {fold_id}/{len(folds)}")
                print("=" * 60)

                # Keep a model's seed identical whether it is run alone, as a
                # staged family, or in the original all-model command.
                canonical_order = [
                    "linear_svm", "logistic_regression", "naive_bayes",
                    "lstm", "bilstm", "bert", "roberta",
                    "llm_zero_shot", "llm_few_shot",
                ]
                seed = common.seed + (canonical_order.index(raw_name) + 1) * 1000 + fold_id
                use_smote_for_model = bool(common.use_smote and model_name in smote_allowed_models)
                model_artifact_dir = os.path.join(artifacts_root, model_name, f"fold_{fold_id}")
                model_temp_dir = os.path.join(common.output_dir, "temp", model_name, f"fold_{fold_id}")
                os.makedirs(model_artifact_dir, exist_ok=True)
                os.makedirs(model_temp_dir, exist_ok=True)

                model_start = time.time()

                if model_name == "bert":
                    metrics, train_t, infer_t = run_transformer(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    cfg=bert_cfg,
                    seed=seed,
                    use_smote=use_smote_for_model,
                    output_dir=model_temp_dir,
                    save_dir=model_artifact_dir,
                    smote_target_ratio=args.smote_target_ratio,
                    )
                elif model_name == "roberta":
                    metrics, train_t, infer_t = run_transformer(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    cfg=roberta_cfg,
                    seed=seed,
                    use_smote=use_smote_for_model,
                    output_dir=model_temp_dir,
                    save_dir=model_artifact_dir,
                    smote_target_ratio=args.smote_target_ratio,
                    )
                elif model_name == "linear_svm":
                    metrics, train_t, infer_t = run_linear_svm(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    use_smote=use_smote_for_model,
                    seed=seed,
                    epochs=args.ml_epochs,
                    save_dir=model_artifact_dir,
                    smote_target_ratio=args.smote_target_ratio,
                    )
                elif model_name == "naive_bayes":
                    metrics, train_t, infer_t = run_naive_bayes(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    use_smote=use_smote_for_model,
                    seed=seed,
                    epochs=args.ml_epochs,
                    save_dir=model_artifact_dir,
                    smote_target_ratio=args.smote_target_ratio,
                    )
                elif model_name == "logistic_regression":
                    metrics, train_t, infer_t = run_logistic_regression(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    use_smote=use_smote_for_model,
                    seed=seed,
                    epochs=args.ml_epochs,
                    save_dir=model_artifact_dir,
                    smote_target_ratio=args.smote_target_ratio,
                    )
                elif model_name == "lstm":
                    metrics, train_t, infer_t = run_lstm_like(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    cfg=rnn_cfg,
                    bidirectional=False,
                    use_smote=use_smote_for_model,
                    seed=seed,
                    save_dir=model_artifact_dir,
                    )
                elif model_name == "bilstm":
                    metrics, train_t, infer_t = run_lstm_like(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    cfg=rnn_cfg,
                    bidirectional=True,
                    use_smote=use_smote_for_model,
                    seed=seed,
                    save_dir=model_artifact_dir,
                    )
                elif model_name in {"llm_zero_shot", "llm_few_shot"}:
                    metrics, train_t, infer_t = run_llm_zero_few_shot(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    cfg=llm_cfg,
                    mode="few_shot" if model_name == "llm_few_shot" else "zero_shot",
                    seed=seed,
                    save_dir=model_artifact_dir,
                    )
                else:
                    raise ValueError(f"Unsupported model in current pipeline scope: {raw_name}")

                # Every model implementation persists predictions.npy and labels.npy.
                # Also write a readable row-level test result immediately, before
                # proceeding to the next model/fold.
                saved_predictions = np.load(os.path.join(model_artifact_dir, "predictions.npy"))
                saved_labels = np.load(os.path.join(model_artifact_dir, "labels.npy"))
                test_results_file = export_test_predictions(
                    save_dir=model_artifact_dir,
                    model=model_name,
                    fold=fold_id,
                    source_indices=test_idx,
                    texts=test_texts,
                    y_true=saved_labels,
                    y_pred=saved_predictions,
                )

                model_end = time.time()

                row = {
                    "model": raw_name,
                    "fold": int(fold_id),
                    "train_time_sec": float(train_t),
                    "infer_time_sec": float(infer_t),
                    "smote_train_only": int(use_smote_for_model),
                    "smote_target_ratio": float(args.smote_target_ratio) if use_smote_for_model else 0.0,
                    "artifact_dir": model_artifact_dir,
                    "temp_dir": model_temp_dir,
                    "execution_type": "prediction_only" if model_name in {"llm_zero_shot", "llm_few_shot"} else "train_and_test",
                    "training_required": int(model_name not in {"llm_zero_shot", "llm_few_shot"}),
                    "test_results_file": test_results_file,
                }
                row.update(metrics)
                rows.append(row)
                # Save every completed model/fold immediately so interrupted
                # long-running experiments retain all finished results.
                export_fold_result(row, common.output_dir)

                process_records.append(
                    {
                        "fold": int(fold_id),
                        "model": raw_name,
                        "normalized_model": model_name,
                        "seed": int(seed),
                        "artifact_dir": model_artifact_dir,
                        "temp_dir": model_temp_dir,
                        "analysis_dir": analysis_dir,
                        "smote_train_only": bool(use_smote_for_model),
                        "smote_target_ratio": float(args.smote_target_ratio) if use_smote_for_model else 0.0,
                        "started_at_unix": float(model_start),
                        "ended_at_unix": float(model_end),
                        "train_time_sec": float(train_t),
                        "infer_time_sec": float(infer_t),
                        "llm_model_name": llm_cfg.model_name if model_name in {"llm_zero_shot", "llm_few_shot"} else "",
                        "execution_type": "prediction_only" if model_name in {"llm_zero_shot", "llm_few_shot"} else "train_and_test",
                        "prediction_scope": "inner cross-validation validation fold only",
                        "test_results_file": test_results_file,
                        "metrics": {k: float(v) for k, v in metrics.items()},
                    }
                )

                print(f"subset_accuracy={metrics['subset_accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")
                overall_pbar.update(1)
    finally:
        overall_pbar.close()

    export_results(rows, common.output_dir, append=args.append_results)

    # Retrain each model once on the complete 80% development set, then perform
    # the only evaluation against the untouched 20% final test set.
    final_rows: List[Dict[str, float]] = []
    final_reliability_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    final_output_dir = os.path.join(common.output_dir, "final_test")
    final_train_texts = [texts[i] for i in development_idx]
    final_test_texts = [texts[i] for i in final_test_idx]
    final_y_train = labels[development_idx]
    final_y_test = labels[final_test_idx]
    for raw_name in models_to_run:
        model_name = raw_name
        seed = common.seed + (canonical_order.index(raw_name) + 1) * 1000
        use_smote_for_model = bool(common.use_smote and model_name in smote_allowed_models)
        model_artifact_dir = os.path.join(final_output_dir, "model_artifacts", model_name)
        model_temp_dir = os.path.join(final_output_dir, "temp", model_name)
        os.makedirs(model_artifact_dir, exist_ok=True)
        os.makedirs(model_temp_dir, exist_ok=True)
        print(f"\nRunning final 20% holdout evaluation: {raw_name}", flush=True)

        if model_name in {"bert", "roberta"}:
            metrics, train_t, infer_t = run_transformer(
                train_texts=final_train_texts,
                train_labels=final_y_train,
                test_texts=final_test_texts,
                test_labels=final_y_test,
                cfg=bert_cfg if model_name == "bert" else roberta_cfg,
                seed=seed,
                use_smote=use_smote_for_model,
                output_dir=model_temp_dir,
                save_dir=model_artifact_dir,
                smote_target_ratio=args.smote_target_ratio,
            )
        elif model_name in {"linear_svm", "naive_bayes", "logistic_regression"}:
            runner = {
                "linear_svm": run_linear_svm,
                "naive_bayes": run_naive_bayes,
                "logistic_regression": run_logistic_regression,
            }[model_name]
            metrics, train_t, infer_t = runner(
                train_texts=final_train_texts,
                train_labels=final_y_train,
                test_texts=final_test_texts,
                test_labels=final_y_test,
                use_smote=use_smote_for_model,
                seed=seed,
                epochs=args.ml_epochs,
                save_dir=model_artifact_dir,
                smote_target_ratio=args.smote_target_ratio,
            )
        elif model_name in {"lstm", "bilstm"}:
            metrics, train_t, infer_t = run_lstm_like(
                train_texts=final_train_texts,
                train_labels=final_y_train,
                test_texts=final_test_texts,
                test_labels=final_y_test,
                cfg=rnn_cfg,
                bidirectional=model_name == "bilstm",
                use_smote=use_smote_for_model,
                seed=seed,
                save_dir=model_artifact_dir,
            )
        elif model_name in {"llm_zero_shot", "llm_few_shot"}:
            metrics, train_t, infer_t = run_llm_zero_few_shot(
                train_texts=final_train_texts,
                train_labels=final_y_train,
                test_texts=final_test_texts,
                test_labels=final_y_test,
                cfg=llm_cfg,
                mode="few_shot" if model_name == "llm_few_shot" else "zero_shot",
                seed=seed,
                save_dir=model_artifact_dir,
            )
        else:
            raise ValueError(f"Unsupported model in final evaluation: {raw_name}")

        final_predictions = np.load(os.path.join(model_artifact_dir, "predictions.npy"))
        final_labels = np.load(os.path.join(model_artifact_dir, "labels.npy"))
        final_reliability_data[model_name] = (final_labels, final_predictions)
        final_results_file = export_test_predictions(
            save_dir=model_artifact_dir,
            model=model_name,
            fold=0,
            source_indices=final_test_idx,
            texts=final_test_texts,
            y_true=final_labels,
            y_pred=final_predictions,
        )
        final_row = {
            "model": raw_name,
            "fold": 0,
            "evaluation_scope": "final_20_percent_holdout",
            "train_time_sec": float(train_t),
            "infer_time_sec": float(infer_t),
            "smote_train_only": int(use_smote_for_model),
            "smote_target_ratio": float(args.smote_target_ratio) if use_smote_for_model else 0.0,
            "artifact_dir": model_artifact_dir,
            "temp_dir": model_temp_dir,
            "test_results_file": final_results_file,
        }
        final_row.update(metrics)
        final_rows.append(final_row)
        print(f"Final test f1_macro={metrics['f1_macro']:.4f}", flush=True)

    export_results(final_rows, final_output_dir, append=args.append_results)
    combined_final_test_files = export_combined_final_test_predictions(
        output_dir=final_output_dir,
        source_indices=final_test_idx,
        texts=final_test_texts,
        y_true=final_y_test,
        model_results=final_reliability_data,
    )
    export_constructiveness_evaluation_reliability(
        final_reliability_data,
        os.path.join(final_output_dir, "reliability_analysis"),
        seed=common.seed,
    )

    manifest = {
        "run": {
            "data_path": args.data_path,
            "output_dir": common.output_dir,
            "seed": int(common.seed),
            "test_size": float(common.test_size),
            "n_folds": int(args.n_folds),
            "protocol": "80/20 final holdout with cross-validation on the 80% development split",
            "final_test_prediction_files": combined_final_test_files,
            "use_smote_on_training_split": bool(common.use_smote),
            "smote_target_ratio": float(args.smote_target_ratio),
            "models": list(models_to_run),
            "label_columns": list(LABEL_COLUMNS),
            "llm": {
                "model_name": llm_cfg.model_name,
                "few_shot_k": int(llm_cfg.few_shot_k),
                "max_new_tokens": int(llm_cfg.max_new_tokens),
                "temperature": float(llm_cfg.temperature),
                "reasoning_effort": llm_cfg.reasoning_effort,
            },
            "transformers": {
                "bert_model_name": bert_cfg.model_name,
                "bert_local_model_path": bert_cfg.local_model_path,
                "roberta_model_name": roberta_cfg.model_name,
                "roberta_local_model_path": roberta_cfg.local_model_path,
                "huggingface_cache_dir": str(Path(args.hf_cache_dir).resolve()),
            },
        },
        "dataset": {
            "total_samples": int(len(texts)),
            "development_samples": int(len(development_idx)),
            "final_test_samples": int(len(final_test_idx)),
            "folds": int(args.n_folds if args.n_folds >= 2 else 1),
        },
        "records": process_records,
    }

    process_json_path = os.path.join(common.output_dir, "training_process.json")
    if args.append_results and os.path.exists(process_json_path):
        try:
            with open(process_json_path, "r", encoding="utf-8") as f:
                previous_manifest = json.load(f)
            previous_records = previous_manifest.get("records", [])
            record_map = {
                (str(record.get("model")), int(record.get("fold", 0))): record
                for record in previous_records
            }
            for record in process_records:
                record_map[(str(record.get("model")), int(record.get("fold", 0)))] = record
            manifest["records"] = list(record_map.values())
            previous_models = previous_manifest.get("run", {}).get("models", [])
            manifest["run"]["models"] = list(dict.fromkeys([*previous_models, *models_to_run]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"Warning: could not merge previous training manifest: {exc}")

    with open(process_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    process_jsonl_path = os.path.join(common.output_dir, "training_process.jsonl")
    with open(process_jsonl_path, "w", encoding="utf-8") as f:
        for rec in manifest["records"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Training process manifest saved: {process_json_path}")
    print(f"Training process JSONL saved: {process_jsonl_path}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    main()
