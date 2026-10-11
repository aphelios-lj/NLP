import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score


class NYTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_label_map(train_df, val_df, test_df):
    labels = sorted(train_df["label"].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    for df in [train_df, val_df, test_df]:
        df["label_id"] = df["label"].map(label2id)

    return label2id, id2label


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


def run_bert(train_df, val_df, test_df):
    label2id, id2label = build_label_map(train_df, val_df, test_df)

    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    train_dataset = NYTDataset(train_df["text"], train_df["label_id"], tokenizer, max_len=64)
    val_dataset = NYTDataset(val_df["text"], val_df["label_id"], tokenizer, max_len=64)
    test_dataset = NYTDataset(test_df["text"], test_df["label_id"], tokenizer, max_len=64)

    training_args = TrainingArguments(
        output_dir="outputs/bert",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir="outputs/bert/logs",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    val_result = trainer.evaluate(val_dataset)
    test_result = trainer.evaluate(test_dataset)

    return val_result, test_result, trainer, tokenizer