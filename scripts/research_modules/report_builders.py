"""Module for generating research paper reports (dataset, models, training)"""

import sys
import json
from pathlib import Path
from datetime import datetime

from src.data.preprocessor import load_and_clean_data
from src.training.config import LABEL_COLUMNS


def generate_dataset_report(data_path="data/cleaned_3label_data.csv", output_dir="results/research_comparison"):
    """Generate comprehensive dataset statistics for research paper"""
    
    print("\n" + "=" * 80)
    print("DATASET ANALYSIS FOR RESEARCH PAPER")
    print("=" * 80)
    
    df = load_and_clean_data(data_path)
    
    # Basic statistics
    n_samples = len(df)
    n_labels = len(LABEL_COLUMNS)
    
    # Text statistics
    text_lengths = df['text'].str.len()
    token_counts = df['text'].str.split().apply(len)
    
    # Label statistics
    label_stats = {}
    label_distribution = {}
    
    for label in LABEL_COLUMNS:
        label_stats[label] = {
            'positive': int((df[label] == 1).sum()),
            'negative': int((df[label] == 0).sum()),
            'positive_ratio': float((df[label] == 1).mean()),
        }
        label_distribution[label] = int((df[label] == 1).sum())
    
    # Multi-label combinations
    df['label_combo'] = df[LABEL_COLUMNS].apply(lambda x: ''.join(x.astype(str)), axis=1)
    label_combos = df['label_combo'].value_counts()
    
    dataset_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_basic": {
            "total_samples": n_samples,
            "number_of_labels": n_labels,
            "label_names": LABEL_COLUMNS,
            "data_source": data_path,
        },
        "text_statistics": {
            "avg_length_chars": float(text_lengths.mean()),
            "std_length_chars": float(text_lengths.std()),
            "min_length_chars": int(text_lengths.min()),
            "max_length_chars": int(text_lengths.max()),
            "avg_tokens": float(token_counts.mean()),
            "std_tokens": float(token_counts.std()),
            "min_tokens": int(token_counts.min()),
            "max_tokens": int(token_counts.max()),
        },
        "label_statistics": label_stats,
        "label_combinations": {
            "total_unique_combinations": len(label_combos),
            "combination_distribution": label_combos.to_dict(),
        },
    }
    
    # Print summary
    print(f"\n📊 DATASET: {Path(data_path).name}")
    print(f"   Total Samples: {n_samples}")
    print(f"   Labels: {', '.join(LABEL_COLUMNS)}")
    print(f"\n📝 TEXT STATISTICS:")
    print(f"   Avg Length: {text_lengths.mean():.1f} chars ({token_counts.mean():.1f} tokens)")
    print(f"   Range: {text_lengths.min()}-{text_lengths.max()} chars")
    print(f"\n🏷️  LABEL DISTRIBUTION:")
    for label, stats in label_stats.items():
        print(f"   {label}: {stats['positive']}/{n_samples} positive ({stats['positive_ratio']:.1%})")
    print(f"\n🔀 MULTI-LABEL COMBINATIONS: {len(label_combos)} unique")
    print(f"   Most common: {label_combos.index[0]} ({label_combos.values[0]} samples)")
    
    # Save detailed report
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_json = output_dir / "dataset_report.json"
    with open(dataset_json, 'w') as f:
        json.dump(dataset_report, f, indent=2)
    print(f"\n✓ Dataset report saved to: {dataset_json}")
    
    # Generate LaTeX table
    latex_path = output_dir / "dataset_statistics.tex"
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|r|r|r|}\n")
        f.write("\\hline\n")
        f.write("Label & Positive & Negative & Balance Ratio \\\\\n")
        f.write("\\hline\n")
        for label, stats in label_stats.items():
            ratio = f"{stats['positive_ratio']:.1%}"
            f.write(f"{label} & {stats['positive']} & {stats['negative']} & {ratio} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Label Distribution in Training Dataset}\n")
        f.write("\\end{table}\n")
    print(f"✓ LaTeX table saved to: {latex_path}")
    
    return dataset_report


def generate_model_configurations(output_dir="results/research_comparison"):
    """Generate model configuration details for research paper"""
    
    print("\n" + "=" * 80)
    print("MODEL CONFIGURATIONS FOR RESEARCH PAPER")
    print("=" * 80)
    
    model_configs = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "linear_svm": {
                "category": "Traditional ML",
                "description": "Linear Support Vector Machine",
                "embeddings": "Word2Vec (300-dim)",
                "hyperparameters": {"C": 1.0, "kernel": "linear", "max_iter": 1000},
                "parameters": "~300 × output_dim",
            },
            "logistic_regression": {
                "category": "Traditional ML",
                "description": "Logistic Regression Classifier",
                "embeddings": "Word2Vec (300-dim)",
                "hyperparameters": {"C": 1.0, "max_iter": 1000},
                "parameters": "~300 × output_dim",
            },
            "naive_bayes": {
                "category": "Traditional ML",
                "description": "Gaussian Naive Bayes",
                "embeddings": "Word2Vec (300-dim)",
                "hyperparameters": {"var_smoothing": 1e-9},
                "parameters": "~300 × output_dim",
            },
            "cnn_attention": {
                "category": "Deep Learning",
                "description": "CNN with Attention Mechanism",
                "architecture": "Conv1D → MaxPool → Attention → Dense",
                "embeddings": "GloVe (300-dim)",
                "hyperparameters": {"filters": 128, "kernel_size": 5, "attention_dim": 64, "epochs": 50},
                "parameters": "~450K",
            },
            "lstm": {
                "category": "Deep Learning",
                "description": "LSTM with Dropout",
                "architecture": "Embedding → LSTM → Dense",
                "embeddings": "GloVe (300-dim)",
                "hyperparameters": {"hidden_dim": 128, "dropout": 0.5, "epochs": 30},
                "parameters": "~200K",
            },
            "bilstm": {
                "category": "Deep Learning",
                "description": "Bidirectional LSTM with Dropout",
                "architecture": "Embedding → BiLSTM → Dense",
                "embeddings": "GloVe (300-dim)",
                "hyperparameters": {"hidden_dim": 128, "dropout": 0.5, "epochs": 30},
                "parameters": "~400K",
            },
            "bert": {
                "category": "Transformer",
                "description": "BERT (Multilingual, Base)",
                "model_id": "bert-base-multilingual-cased",
                "hyperparameters": {"max_length": 128, "learning_rate": 2e-5, "epochs": 3},
                "parameters": "~110M",
                "vocab_size": 119547,
            },
            "roberta": {
                "category": "Transformer",
                "description": "RoBERTa (Base)",
                "model_id": "roberta-base",
                "hyperparameters": {"max_length": 128, "learning_rate": 2e-5, "epochs": 3},
                "parameters": "~125M",
                "vocab_size": 50265,
            },
            "llm_zero_shot": {
                "category": "LLM (Groq API)",
                "description": "Llama 3.1 8B - Zero-shot Classification",
                "model_id": "llama-3.1-8b-instant",
                "inference_type": "Groq Cloud",
                "hyperparameters": {"temperature": 0.0, "max_tokens": 128, "prompt_format": "JSON"},
                "parameters": "8B",
            },
            "llm_few_shot": {
                "category": "LLM (Groq API)",
                "description": "Llama 3.1 8B - Few-shot Classification (k=100)",
                "model_id": "llama-3.1-8b-instant",
                "inference_type": "Groq Cloud",
                "hyperparameters": {"temperature": 0.0, "max_tokens": 128, "few_shot_k": 100, "prompt_format": "JSON"},
                "parameters": "8B",
            },
        }
    }
    
    # Print summary
    for model_key, config in model_configs["models"].items():
        print(f"\n🤖 {config['description']}")
        print(f"   Category: {config['category']}")
        if "embeddings" in config:
            print(f"   Embeddings: {config['embeddings']}")
        if "model_id" in config:
            print(f"   Model: {config['model_id']}")
        print(f"   Parameters: {config.get('parameters', 'N/A')}")
    
    # Save report
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_json = output_dir / "model_configurations.json"
    with open(models_json, 'w') as f:
        json.dump(model_configs, f, indent=2)
    print(f"\n✓ Model configurations saved to: {models_json}")
    
    return model_configs


def generate_training_process_report(output_dir="results/research_comparison"):
    """Generate training process details for research paper"""
    
    print("\n" + "=" * 80)
    print("TRAINING PROCESS FOR RESEARCH PAPER")
    print("=" * 80)
    
    training_report = {
        "timestamp": datetime.now().isoformat(),
        "cross_validation": {
            "strategy": "Stratified K-Fold",
            "n_folds": 10,
            "stratification": "Stratified by all 3 labels simultaneously",
            "random_seed": 42,
        },
        "data_preprocessing": {
            "steps": [
                "Text normalization (lowercase, punctuation)",
                "Tokenization",
                "Stop word removal",
                "Lemmatization/Stemming",
            ],
            "smote_applied": True,
            "smote_config": {
                "strategy": "Oversampling minority classes",
                "k_neighbors": 5,
                "random_state": 42,
            },
            "train_test_split": "90% train / 10% test per fold",
        },
        "training_config": {
            "ml_models": {
                "batch_training": True,
                "hyperparameter_tuning": "Grid Search",
                "cross_validation_inner": "5-fold",
            },
            "deep_learning": {
                "batch_size": 32,
                "epochs": 30,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "early_stopping": True,
                "patience": 3,
                "validation_split": 0.1,
            },
            "transformers": {
                "batch_size": 8,
                "epochs": 3,
                "optimizer": "AdamW",
                "learning_rate": 2e-5,
                "warmup_steps": 100,
                "max_grad_norm": 1.0,
            },
            "llm": {
                "inference_type": "API-based (Groq)",
                "batch_size": 1,
                "temperature": 0.0,
                "max_tokens": 128,
                "prompt_engineering": "Structured JSON format with task-specific instructions",
            },
        },
        "metrics": {
            "evaluation_metrics": [
                "Accuracy (Micro & Macro)",
                "Precision (Micro & Macro)",
                "Recall (Micro & Macro)",
                "F1-Score (Micro & Macro)",
                "Hamming Loss",
                "Subset Accuracy",
            ],
            "multilabel_handling": "Multilabel metrics for 3 independent binary labels",
            "statistical_significance": "Reported as mean ± std across 10 folds",
        },
        "computational_resources": {
            "gpu": "NVIDIA RTX 4080 SUPER (15.69 GB VRAM)",
            "cpu": "Intel Xeon",
            "python_version": "3.11",
            "framework_versions": {
                "torch": "2.x",
                "transformers": "4.x",
                "scikit-learn": "1.x",
                "pandas": "2.x",
                "numpy": "1.x",
            },
        },
    }
    
    # Print summary
    print(f"\n✓ Cross-Validation: {training_report['cross_validation']['n_folds']}-Fold Stratified")
    print(f"✓ Data Preprocessing: {len(training_report['data_preprocessing']['steps'])} steps + SMOTE")
    print(f"✓ Metrics: {len(training_report['metrics']['evaluation_metrics'])} metrics (Micro & Macro)")
    print(f"✓ GPU: {training_report['computational_resources']['gpu']}")
    
    # Save report
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_json = output_dir / "training_process_report.json"
    with open(training_json, 'w') as f:
        json.dump(training_report, f, indent=2)
    print(f"\n✓ Training process report saved to: {training_json}")
    
    # Generate LaTeX section
    latex_path = output_dir / "training_process.tex"
    with open(latex_path, 'w') as f:
        f.write("\\section{Training Process}\n\n")
        f.write("\\subsection{Cross-Validation Strategy}\n")
        f.write(f"We employed {training_report['cross_validation']['n_folds']}-fold stratified cross-validation ")
        f.write("to ensure balanced label distribution across folds. Each fold maintains the same multilabel ")
        f.write("distribution as the original dataset.\n\n")
        
        f.write("\\subsection{Data Preprocessing}\n")
        f.write("Our preprocessing pipeline includes:\n")
        f.write("\\begin{enumerate}\n")
        for step in training_report['data_preprocessing']['steps']:
            f.write(f"\\item {step}\n")
        f.write("\\end{enumerate}\n")
        f.write("To address class imbalance, we applied SMOTE (Synthetic Minority Over-sampling Technique) ")
        f.write("with k=5 neighbors on the training data of each fold.\n\n")
        
        f.write("\\subsection{Training Configuration}\n")
        f.write("\\textbf{Deep Learning Models:} Batch size 32, Adam optimizer (lr=0.001), early stopping with patience=3\\\\\\n")
        f.write("\\textbf{Transformers:} Batch size 8, AdamW optimizer (lr=2e-5), 3 epochs\\\\\\n")
        f.write("\\textbf{LLM Models:} Groq API (llama-3.1-8b-instant), temperature=0.0, max tokens=128\\n\n")
        
        f.write("\\subsection{Evaluation Metrics}\n")
        f.write("We report both macro and micro-averaged metrics to account for multilabel classification:\n")
        f.write("\\begin{itemize}\n")
        for metric in training_report['metrics']['evaluation_metrics']:
            f.write(f"\\item {metric}\n")
        f.write("\\end{itemize}\n")
        f.write("All metrics are reported as mean $\\pm$ standard deviation across 10 folds.\n")
    
    print(f"✓ LaTeX section saved to: {latex_path}")
    
    return training_report


def generate_research_paper_appendix(output_dir="results/research_comparison"):
    """Generate complete appendix for research paper with all details"""
    
    print("\n" + "=" * 80)
    print("GENERATING COMPLETE RESEARCH PAPER APPENDIX")
    print("=" * 80)
    
    # Generate all reports
    dataset_report = generate_dataset_report(output_dir=output_dir)
    model_configs = generate_model_configurations(output_dir=output_dir)
    training_report = generate_training_process_report(output_dir=output_dir)
    
    # Combine into master appendix
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    appendix = {
        "title": "Complete Research Paper Appendix",
        "timestamp": datetime.now().isoformat(),
        "sections": {
            "dataset": dataset_report,
            "models": model_configs,
            "training": training_report,
        }
    }
    
    appendix_json = output_dir / "research_paper_appendix.json"
    with open(appendix_json, 'w') as f:
        json.dump(appendix, f, indent=2)
    
    print(f"\n✓ Complete appendix saved to: {appendix_json}")
    print("\n" + "=" * 80)
    print("GENERATED FILES FOR RESEARCH PAPER:")
    print("=" * 80)
    print(f"1. {output_dir}/dataset_report.json - Dataset statistics")
    print(f"2. {output_dir}/dataset_statistics.tex - LaTeX table for dataset")
    print(f"3. {output_dir}/model_configurations.json - Model details")
    print(f"4. {output_dir}/training_process_report.json - Training details")
    print(f"5. {output_dir}/training_process.tex - LaTeX section for training")
    print(f"6. {output_dir}/research_paper_appendix.json - Complete appendix")
    print("=" * 80)
    
    return appendix
