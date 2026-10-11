from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from src.data_utils import tokenize, evaluate

def _train_and_eval(vectorizer, train_df, val_df, test_df):
    """通用训练与评价流程"""
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_df["label"])

    val_metrics = evaluate(val_df["label"], clf.predict(X_val))
    test_metrics = evaluate(test_df["label"], clf.predict(X_test))
    return val_metrics, test_metrics, clf, vectorizer

def run_binary_bow(train_df, val_df, test_df):
    """Binary Bag of Words"""
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        lowercase=False,
        binary=True,
        token_pattern=None
    )
    return _train_and_eval(vectorizer, train_df, val_df, test_df)

def run_word_freq_bow(train_df, val_df, test_df):
    """Word Frequency Bag of Words"""
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        lowercase=False,
        binary=False,
        token_pattern=None
    )
    return _train_and_eval(vectorizer, train_df, val_df, test_df)