# app/app.py
import sys
sys.path.append("../src")

import streamlit as st
from predict import load_model, predict

st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰"
)

st.title("📰 News Topic Classifier")
st.markdown("Powered by BERT — fine-tuned on AG News")

@st.cache_resource
def get_model():
    return load_model()

model, tokenizer = get_model()

text = st.text_area(
    "Paste a news headline or article:",
    height=150,
    placeholder="e.g. Apple reports record quarterly earnings..."
)

if st.button("Classify"):
    if text.strip():
        with st.spinner("Classifying..."):
            result = predict(text, model, tokenizer)
        st.success(f"Category: **{result['label']}**")
        st.info(f"Confidence: **{result['confidence']}%**")
    else:
        st.warning("Please enter some text first.")