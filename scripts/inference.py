import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 初始化模型和 tokenizer
def load_model(model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()  # 切換到評估模式
    return model, tokenizer

# def predict_with_threshold(logits, threshold=0.5):
#     probs = torch.softmax(logits, dim=-1)
#     confidence, predictions = torch.max(probs, dim=-1)
#     adjusted_predictions = (probs[:, 1] > threshold).long()  # 對「有意義」類別應用閾值
#     return adjusted_predictions, confidence

def predict_with_threshold(logits, threshold=0.8):
    probs = torch.sigmoid(logits)
    
    # 將概率轉換為二分類標籤，基於自定義閾值
    predictions = (probs > threshold).long()
    
    # 返回概率以及調整後的預測結果
    return predictions, probs

# 推論 function
def predict_from_json_with_threshold(model, tokenizer, device, json_input: dict, threshold=0.5):
    texts = json_input.get("texts", [])
    if not texts:
        raise ValueError("JSON input must contain 'texts' key with a list of sentences.")

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predictions, confidences = predict_with_threshold(logits, threshold)

    results = []
    for text, pred, conf in zip(texts, predictions, confidences):
        results.append({
            "text": text,
            "label": int(pred.item()),
            "confidence": float(conf.item())
        })

    return results