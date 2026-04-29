# src/predict.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from config import MODEL_NAME, MODEL_DIR, MAX_LENGTH, DEVICE, LABEL_MAP

def load_model():
    path = str(MODEL_DIR / "final")
    model = BertForSequenceClassification.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def predict(text: str, model, tokenizer) -> dict:
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    pred_id = torch.argmax(probs).item()
    confidence = probs[0][pred_id].item()

    return {
        "label": LABEL_MAP[pred_id],
        "confidence": round(confidence * 100, 2)
    }