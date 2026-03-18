from inference import load_model, predict_from_json_with_threshold
import torch

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model_path = './fine-tuned-Robert-chinese'  # 替換為你的模型路徑

model_path = "../RoBERTa_10fold/roberta-smote-10fold-chinese_iteration_3"
model, tokenizer = load_model(model_path, device)


# 構建測試輸入
json_input = {
    "texts": [
        "跟我家一樣整齊",
        "good",
        "大致上沒有問題",
        "是",
        "整齊",
        " ",
        "排版有符合作業要求",
        "忘記打註解了",
        "整齊但tab沒有照規定",
        "連印出Hello World! 都沒辦法",
        "印不出來喔，System.out.println的S要大寫",
        "HelloWorld 下面要加一下註解 /****@param args (這裡加變數說明)*/",
        "整齊，google checks檢查沒問題",
        "   "
    ]
}
# 推論函數：包含閾值調整
def predict_with_threshold(model, tokenizer, device, json_input, threshold=0.8):
    texts = json_input.get("texts", [])
    if not texts:
        raise ValueError("JSON input must contain 'texts' key with a list of sentences.")

    # 文本轉模型輸入
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=250,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        print(outputs)

    # 取得預測結果和信心分數（logits轉為機率）
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # 設置信心閾值
    confidence_threshold = 0.9  # 可以根據需要調整這個閾值

    # 根據閾值判斷預測結果
    predictions = torch.argmax(probs, dim=-1)
    confidence_scores = torch.max(probs, dim=-1).values
    
    for text, prediction, confidence in zip(texts, predictions, confidence_scores):
        if confidence >= confidence_threshold:
            print(f"文本: {text}, 預測: {prediction.item()}, 信心分數: {confidence.item()}")
        else:
            print(f"文本: {text}, 預測: 未確定, 信心分數: {confidence.item()}")

    results = []
    return results

# 推論
threshold = 0.8  # 設定閾值
results = predict_with_threshold(model, tokenizer, device, json_input, threshold)

# 輸出結果
print("推論結果：")
for result in results:
    print(
        f"文本: {result['text']}, 預測: {result['label']}, "
        f"信心分數: {result['confidence']:.2f}"
    )