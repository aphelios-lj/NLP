import numpy as np
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
from src.data_utils import tokenize, evaluate


def load_glove(path):
    """读取 GloVe 100d 词向量文件"""
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    return embeddings


def document_vector(tokens, embeddings, dim=100):
    """对文档内有效词向量取平均"""
    vecs = [embeddings[w] for w in tokens if w in embeddings]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def build_doc_vectors(texts, embeddings, dim=100):
    """批量构建文档向量矩阵"""
    return np.vstack([document_vector(tokenize(t), embeddings, dim) for t in texts])


def _train_and_eval_vectors(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    基于向量特征的通用训练与评价。
    返回 4 个值 (val_metrics, test_metrics, clf, None)，
    与 Task 1 中 run_binary_bow / run_word_freq_bow 的返回值数量保持一致，
    以便 main.py 统一使用 `val, test, _, _ = ...` 解包。
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    val_metrics = evaluate(y_val, clf.predict(X_val))
    test_metrics = evaluate(y_test, clf.predict(X_test))
    return val_metrics, test_metrics, clf, None


def run_glove(train_df, val_df, test_df, glove_path="glove/glove.6B.100d.txt"):
    """GloVe 100d 平均词向量 + Logistic Regression"""
    embeddings = load_glove(glove_path)

    X_train = build_doc_vectors(train_df["text"], embeddings)
    X_val = build_doc_vectors(val_df["text"], embeddings)
    X_test = build_doc_vectors(test_df["text"], embeddings)

    val_metrics, test_metrics, clf, _ = _train_and_eval_vectors(
        X_train, train_df["label"],
        X_val, val_df["label"],
        X_test, test_df["label"]
    )
    return val_metrics, test_metrics, clf, embeddings


def train_word2vec(texts, vector_size=100, window=5, min_count=1,
                   workers=4, epochs=5, seed=42):
    """在给定文本上训练 Word2Vec"""
    sentences = [tokenize(t) for t in texts]
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=seed
    )
    return model


def word2vec_to_dict(model):
    """将 Word2Vec 模型转为词向量字典，便于复用 build_doc_vectors"""
    return {word: model.wv[word] for word in model.wv.index_to_key}


def run_word2vec_on_ag(train_df, val_df, test_df, ag_df):
    """在 AG News 上训练 Word2Vec，再用于 NYT 分类"""
    model = train_word2vec(ag_df["text"].tolist(), vector_size=100)
    embeddings = word2vec_to_dict(model)

    X_train = build_doc_vectors(train_df["text"], embeddings)
    X_val = build_doc_vectors(val_df["text"], embeddings)
    X_test = build_doc_vectors(test_df["text"], embeddings)

    val_metrics, test_metrics, clf, _ = _train_and_eval_vectors(
        X_train, train_df["label"],
        X_val, val_df["label"],
        X_test, test_df["label"]
    )
    return val_metrics, test_metrics, clf, model


def run_word2vec_on_nyt(train_df, val_df, test_df):
    """在 NYT 训练集上训练 Word2Vec（避免数据泄露），再用于 NYT 分类"""
    model = train_word2vec(train_df["text"].tolist(), vector_size=100)
    embeddings = word2vec_to_dict(model)

    X_train = build_doc_vectors(train_df["text"], embeddings)
    X_val = build_doc_vectors(val_df["text"], embeddings)
    X_test = build_doc_vectors(test_df["text"], embeddings)

    val_metrics, test_metrics, clf, _ = _train_and_eval_vectors(
        X_train, train_df["label"],
        X_val, val_df["label"],
        X_test, test_df["label"]
    )
    return val_metrics, test_metrics, clf, model