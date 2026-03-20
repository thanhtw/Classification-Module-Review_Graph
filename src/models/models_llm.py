import json
import os
import random
import re
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
from groq import Groq

from src.training.config import LABEL_COLUMNS, LLMConfig
from src.utils.metrics import compute_metrics


def _safe_int01(value: object) -> int:
    try:
        return 1 if int(value) >= 1 else 0
    except Exception:
        return 0


def _extract_json_block(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else ""


def _parse_prediction(raw_text: str) -> Tuple[np.ndarray, bool]:
    json_blob = _extract_json_block(raw_text)
    if not json_blob:
        return np.array([0, 0, 0], dtype=int), False

    try:
        data = json.loads(json_blob)
    except Exception:
        return np.array([0, 0, 0], dtype=int), False

    pred = np.array([_safe_int01(data.get(k, 0)) for k in LABEL_COLUMNS], dtype=int)
    return pred, True


def _make_example_line(text: str, label_vec: np.ndarray) -> str:
    label_json = {
        "relevance": int(label_vec[0]),
        "concreteness": int(label_vec[1]),
        "constructive": int(label_vec[2]),
    }
    return f"Text: {text}\nAnswer: {json.dumps(label_json, ensure_ascii=False)}"


def _build_prompt(
    text: str,
    mode: str,
    few_shot_examples: List[Tuple[str, np.ndarray]],
) -> str:
    instruction = (
        "You are a strict multi-label classifier. "
        "Predict 3 binary labels for the input text. "
        "CRITICAL: Return ONLY the JSON object on a single line, with no additional text, thinking, or explanation. "
        "Output format: {\"relevance\": 0 or 1, \"concreteness\": 0 or 1, \"constructive\": 0 or 1}"
    )

    lines = [instruction]
    if mode == "few_shot" and few_shot_examples:
        lines.append("\nExamples:")
        for ex_text, ex_label in few_shot_examples:
            lines.append(_make_example_line(ex_text, ex_label))

    lines.append(f"\nText: {text}")
    lines.append("Output:")
    return "\n".join(lines)


def _sample_few_shot_examples(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    k: int,
    seed: int,
) -> List[Tuple[str, np.ndarray]]:
    if k <= 0 or len(train_texts) == 0:
        return []
    idx = list(range(len(train_texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    idx = idx[: min(k, len(idx))]
    return [(train_texts[i], train_labels[i]) for i in idx]


def run_llm_zero_few_shot(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    cfg: LLMConfig,
    mode: str,
    seed: int,
    save_dir: str = "",
) -> Tuple[Dict[str, float], float, float]:
    if mode not in {"zero_shot", "few_shot"}:
        raise ValueError(f"Unsupported mode: {mode}")

    # Initialize Groq client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    client = Groq(api_key=api_key)

    setup_start = time.perf_counter()
    few_shot_examples = _sample_few_shot_examples(train_texts, train_labels, cfg.few_shot_k, seed)
    setup_time = time.perf_counter() - setup_start

    # Prepare inference
    pred_rows: List[np.ndarray] = []
    parse_failures = 0

    infer_start = time.perf_counter()
    for text in test_texts:
        prompt = _build_prompt(
            text=text,
            mode=mode,
            few_shot_examples=few_shot_examples,
        )

        # Call Groq API
        try:
            completion = client.chat.completions.create(
                model=cfg.model_name,  # Use model from config
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens,
            )
            generated = completion.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            generated = ""

        pred, ok = _parse_prediction(generated)
        if not ok:
            parse_failures += 1
        pred_rows.append(pred)

    infer_time = time.perf_counter() - infer_start

    y_pred = np.stack(pred_rows) if pred_rows else np.zeros_like(test_labels)
    metrics = compute_metrics(test_labels, y_pred)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": f"Groq API ({cfg.model_name})",
                    "mode": mode,
                    "few_shot_k": int(cfg.few_shot_k),
                    "max_new_tokens": int(cfg.max_new_tokens),
                    "temperature": float(cfg.temperature),
                    "train_size": int(len(train_texts)),
                    "test_size": int(len(test_texts)),
                    "parse_failures": int(parse_failures),
                    "parse_failure_rate": float(parse_failures / max(1, len(test_texts))),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return metrics, setup_time, infer_time
