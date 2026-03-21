import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm.auto import tqdm

from src.analysis.analysis_utils import export_train_smote_analysis
from src.training.config import (
    AVAILABLE_MODELS,
    CommonConfig,
    LABEL_COLUMNS,
    LLMConfig,
    RNNConfig,
    TransformerConfig,
    get_env_float,
    get_env_int,
    load_env_file,
)
from src.data.preprocessor import load_and_clean_data, set_seed
from src.models.models_llm import run_llm_zero_few_shot
from src.models.models_nn import run_lstm_like
from src.models.models_ml import run_linear_svm, run_naive_bayes, run_logistic_regression
from src.models.models_transformers import run_transformer
from src.utils.reporting import export_results


def parse_args() -> argparse.Namespace:
    load_env_file(project_root / ".env")

    parser = argparse.ArgumentParser(
        description="Train/test modular models (BERT, RoBERTa, LSTM/BiLSTM, LinearSVM, NaiveBayes, LogisticRegression, optional LLM) on cleaned_3label_data.csv"
    )
    parser.add_argument("--data_path", type=str, default="data/cleaned_3label_data.csv")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert", "roberta", "linear_svm", "naive_bayes", "logistic_regression", "lstm", "bilstm"],
        choices=AVAILABLE_MODELS,
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Used only when --n_folds <= 1")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds for cross validation; use 1 for holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/modular_multimodel")
    parser.add_argument("--no_smote", action="store_true", help="Disable SMOTE on training split")

    parser.add_argument("--rnn_epochs", type=int, default=get_env_int("TRAIN_RNN_EPOCHS", 30))
    parser.add_argument("--bert_epochs", type=int, default=get_env_int("TRAIN_BERT_EPOCHS", 30))
    parser.add_argument("--roberta_epochs", type=int, default=get_env_int("TRAIN_ROBERTA_EPOCHS", 30))
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
        "--llm_model_name",
        type=str,
        default="llama-3.1-8b-instant",
        choices=["llama-3.1-8b-instant"],
        help="Groq API model name for llm_zero_shot/llm_few_shot",
    )
    parser.add_argument("--llm_few_shot_k", type=int, default=100, help="Number of few-shot examples to include in each prompt")
    parser.add_argument("--llm_max_new_tokens", type=int, default=64, help="Maximum generated tokens for LLM responses")
    parser.add_argument("--llm_temperature", type=float, default=0.0, help="Sampling temperature for LLM decoding")
    return parser.parse_args()


def _make_folds(
    n_samples: int,
    n_folds: int,
    test_size: float,
    seed: int,
    labels: np.ndarray | None = None,
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


def main() -> None:
    args = parse_args()
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
    llm_cfg = LLMConfig(
        model_name=args.llm_model_name,
        max_new_tokens=args.llm_max_new_tokens,
        temperature=args.llm_temperature,
        few_shot_k=args.llm_few_shot_k,
    )

    analysis_dir = os.path.join(common.output_dir, "global_train_data_analysis")
    export_train_smote_analysis(
        train_texts=texts,
        train_labels=df[LABEL_COLUMNS].values.astype(int),
        output_dir=analysis_dir,
        seed=common.seed,
        use_smote=bool(common.use_smote),
    )

    folds = _make_folds(len(texts), args.n_folds, common.test_size, common.seed, labels=labels)

    print(f"Total samples: {len(texts)}")
    print(f"Validation mode: {'cross-validation' if args.n_folds >= 2 else 'holdout'}")
    if args.n_folds >= 2:
        print(f"Folds: {args.n_folds}")
    else:
        print(f"Holdout test_size: {common.test_size}")
    print(f"SMOTE on train split (ML only): {bool(common.use_smote)}")
    if any(m in {"llm_zero_shot", "llm_few_shot"} for m in models_to_run):
        print(f"LLM backend model: {llm_cfg.model_name}")

    rows: List[Dict[str, float]] = []
    process_records: List[Dict[str, object]] = []
    artifacts_root = os.path.join(common.output_dir, "model_artifacts")
    os.makedirs(artifacts_root, exist_ok=True)

    smote_allowed_models = {"linear_svm", "naive_bayes", "logistic_regression"}
    total_runs = len(models_to_run) * len(folds)
    overall_pbar = tqdm(total=total_runs, desc="Training progress", unit="fold")

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

                seed = common.seed + (models_to_run.index(raw_name) + 1) * 1000 + fold_id
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
                    )
                elif model_name == "linear_svm":
                    metrics, train_t, infer_t = run_linear_svm(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    use_smote=use_smote_for_model,
                    seed=seed,
                    save_dir=model_artifact_dir,
                    )
                elif model_name == "naive_bayes":
                    metrics, train_t, infer_t = run_naive_bayes(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    use_smote=use_smote_for_model,
                    seed=seed,
                    save_dir=model_artifact_dir,
                    )
                elif model_name == "logistic_regression":
                    metrics, train_t, infer_t = run_logistic_regression(
                    train_texts=train_texts,
                    train_labels=y_train,
                    test_texts=test_texts,
                    test_labels=y_test,
                    use_smote=use_smote_for_model,
                    seed=seed,
                    save_dir=model_artifact_dir,
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

                model_end = time.time()

                row = {
                    "model": raw_name,
                    "fold": int(fold_id),
                    "train_time_sec": float(train_t),
                    "infer_time_sec": float(infer_t),
                    "smote_train_only": int(use_smote_for_model),
                    "artifact_dir": model_artifact_dir,
                    "temp_dir": model_temp_dir,
                }
                row.update(metrics)
                rows.append(row)

                process_records.append(
                    {
                        "fold": int(fold_id),
                        "model": raw_name,
                        "normalized_model": model_name,
                        "seed": int(seed),
                        "artifact_dir": model_artifact_dir,
                        "temp_dir": model_temp_dir,
                        "analysis_dir": analysis_dir,
                        "started_at_unix": float(model_start),
                        "ended_at_unix": float(model_end),
                        "train_time_sec": float(train_t),
                        "infer_time_sec": float(infer_t),
                        "llm_model_name": llm_cfg.model_name if model_name in {"llm_zero_shot", "llm_few_shot"} else "",
                        "metrics": {k: float(v) for k, v in metrics.items()},
                    }
                )

                print(f"subset_accuracy={metrics['subset_accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")
                overall_pbar.update(1)
    finally:
        overall_pbar.close()

    export_results(rows, common.output_dir)

    manifest = {
        "run": {
            "data_path": args.data_path,
            "output_dir": common.output_dir,
            "seed": int(common.seed),
            "test_size": float(common.test_size),
            "n_folds": int(args.n_folds),
            "use_smote_on_training_split": bool(common.use_smote),
            "models": list(models_to_run),
            "label_columns": list(LABEL_COLUMNS),
            "llm": {
                "model_name": llm_cfg.model_name,
                "few_shot_k": int(llm_cfg.few_shot_k),
                "max_new_tokens": int(llm_cfg.max_new_tokens),
                "temperature": float(llm_cfg.temperature),
            },
            "transformers": {
                "bert_model_name": bert_cfg.model_name,
                "roberta_model_name": roberta_cfg.model_name,
            },
        },
        "dataset": {
            "total_samples": int(len(texts)),
            "folds": int(args.n_folds if args.n_folds >= 2 else 1),
        },
        "records": process_records,
    }

    process_json_path = os.path.join(common.output_dir, "training_process.json")
    with open(process_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    process_jsonl_path = os.path.join(common.output_dir, "training_process.jsonl")
    with open(process_jsonl_path, "w", encoding="utf-8") as f:
        for rec in process_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Training process manifest saved: {process_json_path}")
    print(f"Training process JSONL saved: {process_jsonl_path}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    main()
