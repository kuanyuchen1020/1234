"""
匯入必要的套件：
- argparse：用來解析命令列參數（這裡雖然匯入但未使用）
- pandas：用於資料處理與表格運算
- numpy：用於數值運算與隨機種子設定
- sklearn：包含模型訓練與特徵處理模組
- scipy.stats：用於統計分析（此處用來計算皮爾森相關係數）
- random：設定隨機種子以確保結果可重現
"""

import argparse, pandas as pd, numpy as np
from sklearn.model_selection import KFold                # 用於 K 折交叉驗證
from sklearn.feature_extraction.text import TfidfVectorizer  # 將文字轉換為 TF-IDF 特徵向量
from sklearn.linear_model import Ridge                   # 使用 Ridge Regression（L2 正則化線性模型）
from sklearn.pipeline import make_pipeline               # 將多個步驟（特徵轉換＋模型）組成流水線
from sklearn.metrics import mean_squared_error           # 用於計算均方誤差（MSE）
from scipy.stats import pearsonr                         # 用於計算皮爾森相關係數
import random

# ---------------------- 設定隨機種子函式 ----------------------
def set_seed(seed=42):
    """
    設定隨機種子，確保每次程式執行時結果一致。
    參數：
        seed (int): 隨機種子數值，預設為 42
    """
    np.random.seed(seed)     # 設定 NumPy 的隨機種子
    random.seed(seed)        # 設定 Python 內建 random 的隨機種子


# ---------------------- 計算皮爾森相關係數 ----------------------
def pearson(y_true, y_pred):
    """
    計算真實值 (y_true) 與預測值 (y_pred) 之間的皮爾森相關係數。
    若資料中出現常數（無變化）則返回 0.0，以避免數學錯誤。

    參數：
        y_true (array-like): 真實標籤值
        y_pred (array-like): 模型預測值

    回傳：
        float: 皮爾森相關係數（衡量線性相關程度，介於 -1 到 1）
    """
    # 若真實值或預測值的唯一值少於 2，代表資料無變化（常數），無法計算相關係數
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    # pearsonr 會回傳一個物件，包含 statistic（相關係數）與 p-value
    return pearsonr(y_true, y_pred).statistic
