# src/train.py
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (MODEL_DIR, EPOCHS, BATCH_SIZE,
                   LEARNING_RATE, WEIGHT_DECAY)
from dataset import AGNewsDataset
from model import get_model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}


def train():
    print("Loading datasets...")
    train_dataset = AGNewsDataset(split="train")
    test_dataset  = AGNewsDataset(split="test")

    print("Loading model...")
    model = get_model()

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        logging_dir=str(MODEL_DIR / "logs"),
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Training started...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(str(MODEL_DIR / "final"))
    print("Training complete. Model saved.")


if __name__ == "__main__":
    train()