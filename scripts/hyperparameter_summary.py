"""
碩士論文 - 多標籤文本分類模型超參數設定總結
本文件整理了所有實驗模型的詳細超參數配置
"""

import pandas as pd

def get_model_hyperparameters():
    """
    返回所有模型的超參數設定表格
    適用於碩士論文撰寫
    """
    
    hyperparameters = {
        'BERT': {
            '模型類型': 'bert-base-chinese',
            '預訓練模型': 'Google BERT中文基礎版',
            '最大序列長度': 500,
            '學習率': 2e-5,
            '批次大小': 4,
            '訓練輪數': 6,
            '權重衰減': 0.001,
            '優化器': 'AdamW',
            '學習率調度器': '線性衰減',
            '梯度裁剪': 1.0,
            '損失函數': 'BCEWithLogitsLoss + 類別權重',
            'GPU記憶體': '~8-10GB',
            'dropout': '0.1 (預設)',
            '注意力頭數': 12,
            '隱藏層數': 12,
            '隱藏維度': 768
        },
        
        'RoBERTa': {
            '模型類型': 'hfl/chinese-roberta-wwm-ext',
            '預訓練模型': 'HFL中文RoBERTa全詞遮罩擴展版',
            '最大序列長度': 500,
            '學習率': 2e-5,
            '批次大小': 4,
            '訓練輪數': 6,
            '權重衰減': 0.001,
            '優化器': 'AdamW',
            '學習率調度器': '線性衰減',
            '梯度裁剪': 1.0,
            '損失函數': 'BCEWithLogitsLoss + 類別權重',
            'GPU記憶體': '~8-10GB',
            'dropout': '0.1 (預設)',
            '注意力頭數': 12,
            '隱藏層數': 12,
            '隱藏維度': 768
        },
        
        'CNN+Attention': {
            '詞嵌入維度': 100,
            '卷積核數量': 128,
            '卷積核大小': '[3, 4, 5]',
            '最大序列長度': 200,
            '學習率': 0.001,
            '批次大小': 32,
            '訓練輪數': 15,
            '權重衰減': 1e-4,
            '優化器': 'Adam',
            '學習率調度器': 'ReduceLROnPlateau',
            '調度器耐心值': 3,
            '學習率衰減因子': 0.5,
            'dropout': 0.3,
            '梯度裁剪': 1.0,
            '損失函數': 'BCEWithLogitsLoss + 類別權重',
            '池化方式': '全局最大池化',
            '注意力機制': '單層線性注意力',
            'GPU記憶體': '~2-3GB'
        },
        
        'BiLSTM+GloVe': {
            '詞嵌入維度': 100,
            'LSTM隱藏維度': 128,
            '最大序列長度': 200,
            'LSTM層數': 1,
            '雙向LSTM': True,
            '學習率': 0.001,
            '批次大小': 32,
            '訓練輪數': 10,
            '權重衰減': 1e-4,
            '優化器': 'Adam',
            '學習率調度器': 'ReduceLROnPlateau',
            '調度器耐心值': 2,
            '學習率衰減因子': 0.5,
            'dropout': 0.3,
            'LSTM dropout': 0.3,
            '梯度裁剪': 1.0,
            '損失函數': 'BCEWithLogitsLoss + 類別權重',
            '注意力機制': '加權平均注意力',
            'GPU記憶體': '~1-2GB'
        },
        
        'SVM+TF-IDF': {
            '向量化方法': 'TF-IDF',
            '最大特徵數': 5000,
            'n-gram範圍': '(1, 2)',
            '最小文檔頻率': 2,
            '最大文檔頻率': 0.95,
            'SVM核函數': 'RBF',
            'SVM正則化參數C': 1.0,
            'SVM gamma': 'scale',
            '類別權重': 'balanced',
            '多標籤策略': 'MultiOutputClassifier',
            '並行處理': 'n_jobs=-1',
            '機率預測': True,
            '記憶體使用': '~500MB-1GB'
            ,
            '訓練設備': 'CPU',
            '分詞工具': 'jieba'
        }
    }
    
    return hyperparameters

def generate_latex_table():
    """
    生成適合論文的LaTeX表格格式
    """
    hyperparams = get_model_hyperparameters()
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{多標籤文本分類模型超參數設定}")
    print("\\label{tab:hyperparameters}")
    print("\\begin{tabular}{|l|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{超參數} & \\textbf{BERT} & \\textbf{RoBERTa} & \\textbf{CNN+Attention} & \\textbf{BiLSTM+GloVe} & \\textbf{SVM+TF-IDF} \\\\")
    print("\\hline")
    
    # 核心參數
    core_params = [
        ('學習率', ['學習率', '學習率', '學習率', '學習率', 'SVM正則化參數C']),
        ('批次大小', ['批次大小', '批次大小', '批次大小', '批次大小', 'N/A']),
        ('訓練輪數', ['訓練輪數', '訓練輪數', '訓練輪數', '訓練輪數', 'N/A']),
        ('最大序列長度', ['最大序列長度', '最大序列長度', '最大序列長度', '最大序列長度', 'N/A']),
        ('優化器', ['優化器', '優化器', '優化器', '優化器', 'N/A'])
    ]
    
    models = ['BERT', 'RoBERTa', 'CNN+Attention', 'BiLSTM+GloVe', 'SVM+TF-IDF']
    
    for param_name, param_keys in core_params:
        row = f"{param_name}"
        for i, model in enumerate(models):
            if param_keys[i] == 'N/A':
                row += " & N/A"
            else:
                value = hyperparams[model].get(param_keys[i], 'N/A')
                row += f" & {value}"
        row += " \\\\"
        print(row)
        print("\\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")

def generate_detailed_comparison():
    """
    生成詳細的模型比較表
    """
    comparison_data = {
        '模型特徵': {
            'BERT': {
                '模型架構': 'Transformer編碼器',
                '參數量': '~110M',
                '預訓練語料': '中文維基百科等',
                '主要優勢': '強大的上下文理解能力',
                '計算複雜度': '高',
                '記憶體需求': '高 (~8-10GB GPU)'
            },
            'RoBERTa': {
                '模型架構': 'Transformer編碼器',
                '參數量': '~110M',
                '預訓練語料': '中文維基百科+全詞遮罩',
                '主要優勢': '改進的預訓練策略',
                '計算複雜度': '高',
                '記憶體需求': '高 (~8-10GB GPU)'
            },
            'CNN+Attention': {
                '模型架構': 'CNN + 注意力機制',
                '參數量': '~1-5M',
                '預訓練語料': '隨機初始化詞向量',
                '主要優勢': '高效的局部特徵提取',
                '計算複雜度': '中',
                '記憶體需求': '中 (~2-3GB GPU)'
            },
            'BiLSTM+GloVe': {
                '模型架構': '雙向LSTM + 注意力',
                '參數量': '~1-3M',
                '預訓練語料': '模擬GloVe詞向量',
                '主要優勢': '序列建模能力',
                '計算複雜度': '中',
                '記憶體需求': '低 (~1-2GB GPU)'
            },
            'SVM+TF-IDF': {
                '模型架構': '支持向量機',
                '參數量': '取決於支持向量數',
                '預訓練語料': 'N/A',
                '主要優勢': '訓練穩定，可解釋性強',
                '計算複雜度': '低',
                '記憶體需求': '很低 (~500MB-1GB CPU)'
            }
        }
    }
    
    return comparison_data

def save_hyperparameters_csv():
    """
    將超參數保存為CSV文件，方便論文製表
    """
    hyperparams = get_model_hyperparameters()
    
    # 轉換為DataFrame
    df_data = []
    for model_name, params in hyperparams.items():
        for param_name, param_value in params.items():
            df_data.append({
                '模型': model_name,
                '超參數': param_name,
                '數值': param_value
            })
    
    df = pd.DataFrame(df_data)
    df.to_csv('/home/selab/Desktop/Minnn/bertFineTune/model_hyperparameters.csv', 
              index=False, encoding='utf-8-sig')
    
    print("超參數表格已保存至: model_hyperparameters.csv")
    
    # 生成樞紐表格式
    pivot_df = df.pivot(index='超參數', columns='模型', values='數值')
    pivot_df.to_csv('/home/selab/Desktop/Minnn/bertFineTune/model_hyperparameters_pivot.csv', 
                   encoding='utf-8-sig')
    
    print("樞紐表格式已保存至: model_hyperparameters_pivot.csv")
    
    return df, pivot_df

def print_summary():
    """
    打印論文用的簡潔總結
    """
    print("="*80)
    print("多標籤文本分類模型超參數設定總結")
    print("="*80)
    
    summary = """
    本研究採用五種不同的機器學習模型進行多標籤中文文本分類：
    
    1. BERT (Bidirectional Encoder Representations from Transformers)
       - 使用bert-base-chinese預訓練模型
       - 學習率: 2e-5, 批次大小: 4, 訓練6個epoch
       
    2. RoBERTa (Robustly Optimized BERT Pretraining Approach)  
       - 使用hfl/chinese-roberta-wwm-ext預訓練模型
       - 學習率: 2e-5, 批次大小: 4, 訓練6個epoch
       
    3. CNN with Attention
       - 多尺度卷積核[3,4,5], 每種128個過濾器
       - 學習率: 0.001, 批次大小: 32, 訓練15個epoch
       
    4. BiLSTM with GloVe
       - 雙向LSTM隱藏維度128, 詞嵌入維度100
       - 學習率: 0.001, 批次大小: 32, 訓練10個epoch
       
    5. SVM with TF-IDF
       - RBF核函數, C=1.0, 最大特徵數5000
       - TF-IDF向量化, n-gram範圍(1,2)
    
    所有深度學習模型均採用SMOTE重新採樣處理類別不平衡問題，
    並使用10折交叉驗證評估模型性能。
    """
    
    print(summary)

if __name__ == "__main__":
    # 生成所有格式的輸出
    print_summary()
    print("\n" + "="*50)
    print("LaTeX表格格式:")
    print("="*50)
    generate_latex_table()
    
    print("\n" + "="*50) 
    print("CSV文件生成:")
    print("="*50)
    df, pivot_df = save_hyperparameters_csv()
    
    print(f"\n樞紐表預覽:")
    print(pivot_df.head(10))
