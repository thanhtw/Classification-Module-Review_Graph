import json
import os
import random
import re
import time
import csv
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.training.config import LABEL_COLUMNS, LLMConfig
from src.utils.metrics import compute_metrics


def _safe_int01(value: object) -> int:
    try:
        return 1 if int(value) >= 1 else 0
    except Exception:
        return 0


def _extract_last_valid_json_dict(text: str) -> Dict[str, object] | None:
    """Return the last valid JSON object found in a response.

    LLMs sometimes echo the prompt format/example JSON before the final answer,
    so a greedy regex can swallow multiple objects and break json.loads().
    """
    decoder = json.JSONDecoder()
    last_dict = None
    preferred_dict = None

    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        last_dict = obj
        if all(label in obj for label in LABEL_COLUMNS):
            preferred_dict = obj

    return preferred_dict or last_dict


def _parse_prediction(raw_text: str) -> Tuple[np.ndarray, bool, str]:
    data = _extract_last_valid_json_dict(raw_text)
    if data is None:
        return np.array([0, 0, 0], dtype=int), False, "no_valid_json_object_found"
    pred = np.array([_safe_int01(data.get(k, 0)) for k in LABEL_COLUMNS], dtype=int)
    if not all(label in data for label in LABEL_COLUMNS):
        return pred, False, "missing_expected_label_keys"
    return pred, True, ""


def _extract_response_text(response: object) -> str:
    """Best-effort extraction of plain text from an OpenAI Responses API object."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text_value = getattr(content, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                chunks.append(text_value)
            elif isinstance(content, dict):
                maybe_text = content.get("text")
                if isinstance(maybe_text, str) and maybe_text.strip():
                    chunks.append(maybe_text)
    return "\n".join(chunks)


def _label_response_format() -> Dict[str, object]:
    """Strict JSON schema for the three multilabel outputs."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "multilabel_prediction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "relevance": {"type": "integer", "enum": [0, 1]},
                    "concreteness": {"type": "integer", "enum": [0, 1]},
                    "constructive": {"type": "integer", "enum": [0, 1]},
                },
                "required": ["relevance", "concreteness", "constructive"],
                "additionalProperties": False,
            },
        },
    }


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
        "Return exactly one JSON object with keys relevance, concreteness, constructive. "
        "Each value must be either 0 or 1."
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

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    client = OpenAI(api_key=api_key)

    setup_start = time.perf_counter()
    few_shot_examples = _sample_few_shot_examples(train_texts, train_labels, cfg.few_shot_k, seed)
    setup_time = time.perf_counter() - setup_start

    # Prepare inference
    pred_rows: List[np.ndarray] = []
    parse_failures = 0
    api_failures = 0
    parse_failure_records: List[Dict[str, object]] = []
    prediction_records: List[Dict[str, object]] = []

    infer_start = time.perf_counter()
    for idx, text in enumerate(test_texts):
        prompt = _build_prompt(
            text=text,
            mode=mode,
            few_shot_examples=few_shot_examples,
        )

        # Call OpenAI Chat Completions API with strict structured output.
        api_error = ""
        try:
            completion = client.chat.completions.create(
                model=cfg.model_name,  # Use model from config
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_format=_label_response_format(),
                max_completion_tokens=cfg.max_new_tokens,
                store=False,
            )
            generated = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"OpenAI API error: {e}")
            api_failures += 1
            api_error = str(e)
            generated = ""

        pred, ok, parse_reason = _parse_prediction(generated)
        if not ok:
            parse_failures += 1
            parse_failure_records.append(
                {
                    "index": int(idx),
                    "mode": mode,
                    "reason": parse_reason or "unknown_parse_failure",
                    "api_error": api_error,
                    "raw_response": generated,
                    "text_preview": str(text)[:500],
                }
            )
        pred_rows.append(pred)
        prediction_records.append(
            {
                "index": int(idx),
                "text": str(text),
                "mode": mode,
                "raw_response": generated,
                "parse_ok": int(ok),
                "parse_reason": parse_reason,
                "api_error": api_error,
                "true_relevance": int(test_labels[idx][0]),
                "true_concreteness": int(test_labels[idx][1]),
                "true_constructive": int(test_labels[idx][2]),
                "pred_relevance": int(pred[0]),
                "pred_concreteness": int(pred[1]),
                "pred_constructive": int(pred[2]),
            }
        )

    infer_time = time.perf_counter() - infer_start

    y_pred = np.stack(pred_rows) if pred_rows else np.zeros_like(test_labels)
    metrics = compute_metrics(test_labels, y_pred)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save predictions and labels for confusion matrix calculation
        np.save(os.path.join(save_dir, "predictions.npy"), y_pred)
        np.save(os.path.join(save_dir, "labels.npy"), test_labels)

        prediction_csv_path = os.path.join(save_dir, "prediction_results_with_ground_truth.csv")
        with open(prediction_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "text",
                    "mode",
                    "raw_response",
                    "parse_ok",
                    "parse_reason",
                    "api_error",
                    "true_relevance",
                    "true_concreteness",
                    "true_constructive",
                    "pred_relevance",
                    "pred_concreteness",
                    "pred_constructive",
                ],
            )
            writer.writeheader()
            writer.writerows(prediction_records)

        if parse_failure_records:
            parse_failure_path = os.path.join(save_dir, "parse_failures.jsonl")
            with open(parse_failure_path, "w", encoding="utf-8") as f:
                for record in parse_failure_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": f"OpenAI API ({cfg.model_name})",
                    "mode": mode,
                    "few_shot_k": int(cfg.few_shot_k),
                    "max_new_tokens": int(cfg.max_new_tokens),
                    "temperature": float(cfg.temperature),
                    "train_size": int(len(train_texts)),
                    "test_size": int(len(test_texts)),
                    "api_failures": int(api_failures),
                    "parse_failures": int(parse_failures),
                    "parse_failure_rate": float(parse_failures / max(1, len(test_texts))),
                    "prediction_results_file": "prediction_results_with_ground_truth.csv",
                    "parse_failures_file": "parse_failures.jsonl" if parse_failure_records else "",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return metrics, setup_time, infer_time
