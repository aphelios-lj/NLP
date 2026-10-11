import sys
import time
from src.data_utils import load_nyt, load_ag, split_nyt
from src.task1_bow import run_binary_bow, run_word_freq_bow
from src.task2_word2vec import run_glove, run_word2vec_on_ag, run_word2vec_on_nyt
# from src.task3_bert import run_bert   # 暂时注释


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def print_result(name, val_metrics, test_metrics, elapsed):
    log(f"\n{'='*60}")
    log(f"{name}  [耗时 {elapsed:.1f}s]")
    log(f"  Validation: Accuracy={val_metrics['accuracy']:.4f}, Macro-F1={val_metrics['macro_f1']:.4f}")
    log(f"  Test:       Accuracy={test_metrics['accuracy']:.4f}, Macro-F1={test_metrics['macro_f1']:.4f}")
    log(f"{'='*60}\n")


def timed(name, fn, *args, **kwargs):
    log(f"[开始] {name}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    log(f"[完成] {name}，用时 {elapsed:.1f}s")
    return result, elapsed


def main():
    log("Loading data...")
    nyt_df = load_nyt("data/nyt.csv")
    ag_df = load_ag("data/ag.csv")
    log(f"NYT 样本数: {len(nyt_df)}, AG 样本数: {len(ag_df)}")

    log("Splitting NYT (80/10/10)...")
    train_df, val_df, test_df = split_nyt(nyt_df)
    log(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Task 1
    (val, test, _, _), t = timed(
        "Task 1: Binary Bag of Words",
        run_binary_bow, train_df, val_df, test_df
    )
    print_result("Task 1: Binary Bag of Words", val, test, t)

    (val, test, _, _), t = timed(
        "Task 1: Word Frequency Bag of Words",
        run_word_freq_bow, train_df, val_df, test_df
    )
    print_result("Task 1: Word Frequency Bag of Words", val, test, t)

    # Task 2
    (val, test, _, _), t = timed(
        "Task 2: Pre-trained GloVe (100d)",
        run_glove, train_df, val_df, test_df
    )
    print_result("Task 2: Pre-trained GloVe (100d)", val, test, t)

    (val, test, _, _), t = timed(
        "Task 2: Word2Vec Trained on AG News",
        run_word2vec_on_ag, train_df, val_df, test_df, ag_df
    )
    print_result("Task 2: Word2Vec Trained on AG News", val, test, t)

    (val, test, _, _), t = timed(
        "Task 2: Word2Vec Trained on NYT",
        run_word2vec_on_nyt, train_df, val_df, test_df
    )
    print_result("Task 2: Word2Vec Trained on NYT", val, test, t)

    # Task 3（暂时跳过）
    # log("[开始] Task 3: BERT 微调")
    # val_result, test_result, _, _ = run_bert(train_df, val_df, test_df)
    # print_result(
    #     "Task 3: BERT-base-uncased Fine-tuned",
    #     {"accuracy": val_result["eval_accuracy"], "macro_f1": val_result["eval_macro_f1"]},
    #     {"accuracy": test_result["eval_accuracy"], "macro_f1": test_result["eval_macro_f1"]},
    #     0.0
    # )

    log("Task 1 与 Task 2 全部完成。")


if __name__ == "__main__":
    main()