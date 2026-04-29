# src/model.py
from transformers import BertForSequenceClassification
import torch
from config import MODEL_NAME, NUM_LABELS, DEVICE

def get_model():
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    return model.to(DEVICE)