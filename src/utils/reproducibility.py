"""
Reproducibility tracking: Record all parameters, environment, and hardware info
for complete reproducibility of experiments.
"""

import json
import os
import sys
import platform
from typing import Dict, Any
import subprocess
import torch


def get_environment_info() -> Dict[str, Any]:
    """Collect comprehensive environment information."""
    
    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    except:
        gpu_name = "Unknown"
    
    try:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except:
        gpu_count = 0
    
    # Get Python packages
    packages = {}
    important_packages = ["torch", "transformers", "sklearn", "numpy", "pandas", "scipy"]
    
    for pkg_name in important_packages:
        try:
            pkg = __import__(pkg_name)
            packages[pkg_name] = getattr(pkg, "__version__", "unknown")
        except:
            packages[pkg_name] = "not installed"
    
    env_info = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "platform_processor": platform.processor(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": int(gpu_count),
        "gpu_name": gpu_name,
        "packages": packages,
    }
    
    return env_info


def get_model_checkpoint_info(model_name: str, config_dict: Dict) -> Dict[str, str]:
    """Extract model checkpoint information from config."""
    
    checkpoint_info = {}
    
    # Map model names to checkpoint keys and default values
    model_checkpoint_map = {
        "bert": ("bert_model_name", "bert-base-chinese"),
        "roberta": ("roberta_model_name", "hfl/chinese-roberta-wwm-ext"),
        "llm_zero_shot": ("llm_model_name", "Qwen/Qwen2-7B-Instruct"),
        "llm_few_shot": ("llm_model_name", "Qwen/Qwen2-7B-Instruct"),
    }
    
    if model_name in model_checkpoint_map:
        config_key, default_val = model_checkpoint_map[model_name]
        checkpoint = config_dict.get(config_key, default_val)
        checkpoint_info["checkpoint"] = checkpoint
        checkpoint_info["source"] = "HuggingFace"
    elif model_name in ["svm", "decision_tree"]:
        checkpoint_info["checkpoint"] = f"sklearn.{model_name}"
        checkpoint_info["source"] = "scikit-learn"
    elif model_name in ["lstm", "bilstm", "cnn_attention"]:
        checkpoint_info["checkpoint"] = "custom (trained from scratch)"
        checkpoint_info["source"] = "custom"
    
    return checkpoint_info


def create_reproducibility_manifest(
    model_name: str,
    fold: int,
    seed: int,
    training_config: Dict[str, Any],
    training_time_sec: float,
    inference_time_sec: float,
    artifacts_dir: str,
) -> Dict[str, Any]:
    """
    Create comprehensive reproducibility manifest.
    """
    
    manifest = {
        "reproducibility_info": {
            "timestamp": str(__import__("datetime").datetime.now()),
            "hostname": platform.node(),
        },
        "model": {
            "name": model_name,
            "seed": int(seed),
            "fold": int(fold),
            **get_model_checkpoint_info(model_name, training_config),
        },
        "environment": get_environment_info(),
        "training": {
            "seed": int(seed),
            "training_time_sec": float(training_time_sec),
            "inference_time_sec": float(inference_time_sec),
        },
        "hyperparameters": training_config,
        "artifacts": {
            "saved_to": artifacts_dir,
        },
    }
    
    return manifest


def save_reproducibility_manifest(
    manifest: Dict[str, Any],
    output_path: str,
):
    """Save reproducibility manifest to JSON."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def print_reproducibility_info(manifest: Dict[str, Any]):
    """Pretty print reproducibility information."""
    
    print(f"\n{'='*80}")
    print("REPRODUCIBILITY INFORMATION")
    print(f"{'='*80}")
    print(f"Model: {manifest['model']['name']}")
    print(f"Seed: {manifest['model']['seed']}")
    print(f"Fold: {manifest['model']['fold']}")
    if "checkpoint" in manifest['model']:
        print(f"Checkpoint: {manifest['model']['checkpoint']}")
    
    print(f"\nEnvironment:")
    env = manifest["environment"]
    print(f"  Python: {env['python_version']}")
    print(f"  Platform: {env['platform']}")
    print(f"  GPU: {env['gpu_name']} ({env['gpu_count']} available)")
    
    print(f"\nPackages:")
    for pkg, version in env["packages"].items():
        print(f"  {pkg}: {version}")
    
    print(f"\nTraining:")
    training = manifest["training"]
    print(f"  Training Time: {training['training_time_sec']:.2f}s")
    print(f"  Inference Time: {training['inference_time_sec']:.2f}s")
    
    print(f"\nHyperparameters:")
    for key, value in manifest["hyperparameters"].items():
        if isinstance(value, (int, float, str, bool)):
            print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    print("Reproducibility tracking module ready to use in analysis pipeline.")
