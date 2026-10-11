import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split

# 在脚本中确保 NLTK 数据已下载（首次运行会自动下载）
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk import word_tokenize

RANDOM_SEED = 42

def tokenize(text):
    """统一分词函数，所有 Task 共用"""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return word_tokenize(text)

def load_nyt(path="data/nyt.csv"):
    """加载 NYT 数据集，返回含 text 和 label 两列的 DataFrame"""
    df = pd.read_csv(path)
    # 打印列名以便确认，首次运行后可注释掉
    print("NYT columns:", df.columns.tolist())
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    return df

def load_ag(path="data/ag.csv"):
    """加载 AG News 数据集，仅取文本列"""
    df = pd.read_csv(path)
    print("AG columns:", df.columns.tolist())
    if "text" not in df.columns:
        text_col = df.columns[0]
    else:
        text_col = "text"
    df = df[[text_col]].dropna()
    df.columns = ["text"]
    df["text"] = df["text"].astype(str)
    return df

def split_nyt(df, seed=RANDOM_SEED):
    """NYT 数据划分：80% 训练，10% 验证，10% 测试"""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label"] if df["label"].nunique() > 1 else None
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"] if temp_df["label"].nunique() > 1 else None
    )
    return train_df, val_df, test_df

def evaluate(y_true, y_pred):
    """统一评价函数"""
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro")
    }