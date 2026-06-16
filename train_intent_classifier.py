"""
DistilBERT intent classifier for the Campus Chatbot project.

Fine-tunes distilbert-base-uncased on the synthetic campus FAQ corpus to classify
user queries into intents: find_location, ask_hours, find_event, ask_description.

Tracks training metrics (loss, accuracy) and evaluates with precision/recall/F1.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

TEXT_CORPUS_PATH = "data/knowledge_base/text_corpus.json"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/intent_classifier"
RESULTS_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_corpus(path=TEXT_CORPUS_PATH):
    with open(path, "r") as f:
        corpus = json.load(f)
    texts = [sample["text"] for sample in corpus]
    intents = [sample["intent"] for sample in corpus]
    return texts, intents


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def plot_training_curves(log_history, save_path=os.path.join(RESULTS_DIR, "intent_training_curves.png")):
    train_loss = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    eval_acc = [(e["epoch"], e["eval_accuracy"]) for e in log_history if "eval_accuracy" in e]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if train_loss:
        steps, losses = zip(*train_loss)
        axes[0].plot(steps, losses, marker="o")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")

    if eval_acc:
        epochs, accs = zip(*eval_acc)
        axes[1].plot(epochs, accs, marker="o", color="green")
        axes[1].set_title("Validation Accuracy per Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    texts, intents = load_corpus()

    unique_intents = sorted(set(intents))
    intent_to_idx = {intent: i for i, intent in enumerate(unique_intents)}
    idx_to_intent = {i: intent for intent, i in intent_to_idx.items()}
    labels = [intent_to_idx[intent] for intent in intents]

    print(f"Loaded {len(texts)} samples across {len(unique_intents)} intents: {unique_intents}")

    # Train/val split (stratified)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"Train samples: {len(train_texts)} | Val samples: {len(val_texts)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_intents),
        id2label=idx_to_intent,
        label2id=intent_to_idx,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        num_train_epochs=8,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
    )

    from transformers import EarlyStoppingCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n=== Training DistilBERT intent classifier ===")
    trainer.train()

    print("\n=== Final Evaluation ===")
    eval_results = trainer.evaluate()
    print(eval_results)

    # Detailed per-class report
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    precision, recall, f1, support = precision_recall_fscore_support(
        val_labels, preds, average=None, zero_division=0, labels=list(range(len(unique_intents)))
    )

    per_class_report = []
    for i, intent in enumerate(unique_intents):
        per_class_report.append({
            "intent": intent,
            "precision": round(float(precision[i]), 3),
            "recall": round(float(recall[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(support[i]),
        })
        print(f"  {intent}: precision={precision[i]:.3f}, recall={recall[i]:.3f}, f1={f1[i]:.3f}, support={support[i]}")

    # Save results
    results_summary = {
        "overall": {k: round(float(v), 3) for k, v in eval_results.items() if isinstance(v, (int, float))},
        "per_class": per_class_report,
        "intent_labels": unique_intents,
    }
    with open(os.path.join(RESULTS_DIR, "intent_classifier_results.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved evaluation results to {RESULTS_DIR}/intent_classifier_results.json")

    # Plot training curves
    plot_training_curves(trainer.state.log_history)

    # Save the fine-tuned model
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print(f"Saved fine-tuned model to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()