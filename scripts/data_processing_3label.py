import pandas as pd
import os

# 設定正確的絕對路徑
file_path = '../data/originData_3labeled.csv'

# 檢查文件是否存在
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件不存在於: {file_path}，請確認路徑與文件名！")

# 讀取並清理數據
try:
    data = pd.read_csv(
        file_path,
        encoding='utf-8',
        header=None,
        names=['text', 'relevance', 'concreteness', 'constructive'],
        quoting=1,  # 處理含換行的字段
        skip_blank_lines=True,
        engine='python'
    )
    
    # 替換換行符為空格
    data['text'] = data['text'].str.replace('\n', ' ', regex=True)
    
    # 清理前後空白
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # 保存清理後的數據
    output_path = '/home/selab/Desktop/Minnn/bertFineTune/data/cleaned_3label_data.csv'
    data.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("數據清理完成！保存於:", output_path)

except Exception as e:
    print("處理失敗，錯誤訊息:", str(e))
