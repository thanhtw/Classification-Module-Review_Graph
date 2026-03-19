import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split

from modular_training.analysis_utils import export_train_smote_analysis
from modular_training.config import AVAILABLE_MODELS, CNNConfig, CommonConfig, LABEL_COLUMNS, RNNConfig, TransformerConfig
from modular_training.data_utils import load_and_clean_data, set_seed
from modular_training.models_nn import run_cnn_attention, run_lstm_like
from modular_training.models_svm import run_svm
from modular_training.models_transformers import run_transformer
from modular_training.report_utils import export_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/test BERT, RoBERTa, SVM, CNN_Attention, LSTM, BiLSTM on cleaned_3label_data.csv with 10-fold CV by default"
    )
    parser.add_argument("--data_path", type=str, default="data/cleaned_3label_data.csv")
    parser.add_argument("--models", nargs="+", default=["bert", "roberta", "svm", "cnn_attention", "lstm", "bilstm"], choices=AVAILABLE_MODELS)
    parser.add_argument("--test_size", type=float, default=0.2, help="Used only when --n_folds <= 1")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds for cross validation; use 1 for holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/modular_multimodel")
    parser.add_argument("--no_smote", action="store_true", help="Disable SMOTE on training split")

    parser.add_argument("--rnn_epochs", type=int, default=10)
    parser.add_argument("--cnn_epochs", type=int, default=5)
    parser.add_argument("--bert_epochs", type=int, default=5)
    parser.add_argument("--roberta_epochs", type=int, default=5)
    parser.add_argument("--glove_path", type=str, default="", help="Path to pretrained GloVe/word vectors text file")
    parser.add_argument("--freeze_glove", action="store_true", help="Freeze embedding layer initialized by GloVe vectors")
    return parser.parse_args()


def _normalize_model_name(name: str) -> str:
    n = name.lower()
    if n == "lsmt":
        return "lstm"
    return n


def main() -> None:
    args = parse_args()

    common = CommonConfig(seed=args.seed, test_size=args.test_size, use_smote=(not args.no_smote), output_dir=args.output_dir)
    set_seed(common.seed)
    os.makedirs(common.output_dir, exist_ok=True)

    df = load_and_clean_data(args.data_path)
    texts = df["text"].tolist()
    labels = df[LABEL_COLUMNS].values.astype(int)

    idx = np.arange(len(texts))
    folds: List[Dict[str, np.ndarray]] = []
    if args.n_folds >= 2:
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=common.seed)
        for tr_idx, te_idx in kf.split(idx):
            folds.append({"train_idx": tr_idx, "test_idx": te_idx})
    else:
        train_idx, test_idx = train_test_split(idx, test_size=common.test_size, random_state=common.seed, shuffle=True)
        folds.append({"train_idx": train_idx, "test_idx": test_idx})

    print(f"Total samples: {len(texts)}")
    print(f"Validation mode: {'cross-validation' if args.n_folds >= 2 else 'holdout'}")
    if args.n_folds >= 2:
        print(f"Folds: {args.n_folds}")
    else:
        print(f"Holdout test_size: {common.test_size}")
    print(f"SMOTE applied: {common.use_smote} (train split only)")

    rows: List[Dict[str, float]] = []
    process_records: List[Dict[str, object]] = []
    artifacts_root = os.path.join(common.output_dir, "model_artifacts")
    os.makedirs(artifacts_root, exist_ok=True)

    rnn_cfg = RNNConfig(
        epochs=args.rnn_epochs,
        glove_path=args.glove_path,
        glove_trainable=(not args.freeze_glove),
    )
    cnn_cfg = CNNConfig(
        epochs=args.cnn_epochs,
        glove_path=args.glove_path,
        glove_trainable=(not args.freeze_glove),
    )
    bert_cfg = TransformerConfig(model_name="bert-base-chinese", epochs=args.bert_epochs)
    roberta_cfg = TransformerConfig(model_name="hfl/chinese-roberta-wwm-ext", epochs=args.roberta_epochs)

    for fold_id, fold_data in enumerate(folds, start=1):
        train_idx = fold_data["train_idx"]
        test_idx = fold_data["test_idx"]
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        print("\n" + "#" * 60)
        print(f"Fold {fold_id}/{len(folds)} | Train: {len(train_texts)} | Test: {len(test_texts)}")
        print("#" * 60)

        analysis_dir = os.path.join(common.output_dir, "folds", f"fold_{fold_id}", "train_data_analysis")
        export_train_smote_analysis(
            train_texts=train_texts,
            train_labels=y_train,
            output_dir=analysis_dir,
            seed=common.seed + fold_id,
            use_smote=common.use_smote,
        )
        print(f"Fold {fold_id} train data analysis exported to: {analysis_dir}")

        for raw_name in args.models:
            model_name = _normalize_model_name(raw_name)
            print("\n" + "=" * 60)
            print(f"Running: {raw_name} | Fold {fold_id}/{len(folds)}")
            print("=" * 60)

            seed = common.seed + fold_id * 100 + len(rows) + 1
            model_artifact_dir = os.path.join(artifacts_root, model_name, f"fold_{fold_id}")
            model_temp_dir = os.path.join(common.output_dir, "temp", model_name, f"fold_{fold_id}")
            os.makedirs(model_artifact_dir, exist_ok=True)
            os.makedirs(model_temp_dir, exist_ok=True)

            model_start = time.time()

            if model_name == "svm":
                metrics, train_t, infer_t = run_svm(
                    train_texts,
                    y_train,
                    test_texts,
                    y_test,
                    common.use_smote,
                    seed,
                    save_dir=model_artifact_dir,
                )
            elif model_name == "cnn_attention":
                metrics, train_t, infer_t = run_cnn_attention(
                    train_texts,
                    y_train,
                    test_texts,
                    y_test,
                    cnn_cfg,
                    common.use_smote,
                    seed,
                    save_dir=model_artifact_dir,
                )
            elif model_name == "lstm":
                metrics, train_t, infer_t = run_lstm_like(
                    train_texts,
                    y_train,
                    test_texts,
                    y_test,
                    rnn_cfg,
                    False,
                    common.use_smote,
                    seed,
                    save_dir=model_artifact_dir,
                )
            elif model_name == "bilstm":
                metrics, train_t, infer_t = run_lstm_like(
                    train_texts,
                    y_train,
                    test_texts,
                    y_test,
                    rnn_cfg,
                    True,
                    common.use_smote,
                    seed,
                    save_dir=model_artifact_dir,
                )
            elif model_name == "bert":
                metrics, train_t, infer_t = run_transformer(
                    train_texts,
                    y_train,
                    test_texts,
                    y_test,
                    bert_cfg,
                    seed,
                    common.use_smote,
                    model_temp_dir,
                    save_dir=model_artifact_dir,
                )
            elif model_name == "roberta":
                metrics, train_t, infer_t = run_transformer(
                    train_texts,
                    y_train,
                    test_texts,
                    y_test,
                    roberta_cfg,
                    seed,
                    common.use_smote,
                    model_temp_dir,
                    save_dir=model_artifact_dir,
                )
            else:
                raise ValueError(f"Unsupported model: {raw_name}")

            model_end = time.time()

            row = {
                "model": raw_name,
                "fold": int(fold_id),
                "train_time_sec": float(train_t),
                "infer_time_sec": float(infer_t),
                "smote_train_only": int(common.use_smote),
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
                    "metrics": {k: float(v) for k, v in metrics.items()},
                }
            )

            print(f"subset_accuracy={metrics['subset_accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

    export_results(rows, common.output_dir)

    manifest = {
        "run": {
            "data_path": args.data_path,
            "output_dir": common.output_dir,
            "seed": int(common.seed),
            "test_size": float(common.test_size),
            "n_folds": int(args.n_folds),
            "use_smote_train_only": bool(common.use_smote),
            "models": list(args.models),
            "label_columns": list(LABEL_COLUMNS),
        },
        "dataset": {
            "total_samples": int(len(texts)),
            "folds": int(len(folds)),
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
