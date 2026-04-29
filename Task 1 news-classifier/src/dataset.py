# src/dataset.py
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
from config import (MODEL_NAME, MAX_LENGTH, DATA_DIR)

class AGNewsDataset(Dataset):
    def __init__(self, split="train"):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        raw = load_dataset("ag_news")
        self.data = raw[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }