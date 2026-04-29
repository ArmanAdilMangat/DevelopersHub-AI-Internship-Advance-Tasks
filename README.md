# 🤖 AI/ML Engineering Internship — DevelopersHub Corporation

> **Intern:** Adil | BS Artificial Intelligence, University of Management and Technology (UMT), Lahore
> **Organization:** DevelopersHub Corporation — Advanced AI/ML Engineering Track
> **Stack:** Python 3.10 · PyTorch · HuggingFace Transformers · Scikit-learn · Streamlit · NVIDIA RTX 4060

---

## Overview

This repository contains three production-grade AI/ML projects completed during the Advanced Phase of my internship at DevelopersHub Corporation. Each project follows a modular, industry-standard architecture with full EDA, model training, evaluation, and live Streamlit deployment.

---

## Projects at a Glance

| # | Project | Domain | Model(s) | Key Result |
|---|---------|--------|----------|------------|
| [Task 1](#-task-1--news-topic-classifier-using-bert) | News Topic Classifier | NLP / Text Classification | BERT (fine-tuned) | **94.7% Accuracy · 94.7 F1** |
| [Task 2](#-task-2--customer-churn-prediction-pipeline) | Customer Churn Prediction | Classic ML / Tabular | Logistic Regression · Random Forest | End-to-End Scikit-learn Pipeline |
| [Task 5](#-task-5--auto-tagging-support-tickets-using-llms) | Support Ticket Auto-Tagger | NLP / LLM Comparison | Zero-Shot · Few-Shot · Fine-Tuned | **99.82% Accuracy (Fine-Tuned)** |

---

## 📰 Task 1 — News Topic Classifier Using BERT

### Problem
Automatically classify news articles into one of four categories — **World, Sports, Business, Sci/Tech** — using transformer-based transfer learning.

### Dataset
[AG News (HuggingFace)](https://huggingface.co/datasets/ag_news) — 120,000 training samples + 7,600 test samples. Perfectly balanced at 25% per class.

### Approach
Fine-tuned `bert-base-uncased` on the full AG News training set using the HuggingFace `Trainer` API with mixed-precision training (`fp16`) on an NVIDIA RTX 4060 GPU.

### Project Structure
```
news-classifier/
├── data/
├── notebooks/
│   └── 01_eda.ipynb          # Class distribution, text length analysis
├── src/
│   ├── config.py             # Centralized paths & hyperparameters
│   ├── dataset.py            # AGNewsDataset (tokenization + PyTorch Dataset)
│   ├── model.py              # BertForSequenceClassification loader
│   ├── train.py              # HuggingFace Trainer with compute_metrics
│   └── predict.py            # Inference wrapper for deployment
├── models/                   # Saved checkpoints
├── app/
│   └── app.py                # Streamlit demo
└── requirements.txt
```

### Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| `MAX_LENGTH` | 128 | EDA confirmed 99.97% of samples under 128 words |
| `BATCH_SIZE` | 16 | Safe for 8GB VRAM with fp16 enabled |
| `EPOCHS` | 3 | Standard for BERT fine-tuning; avoids overfitting |
| `LEARNING_RATE` | 2e-5 | Golden BERT fine-tuning value (preserves pretrained weights) |
| `WEIGHT_DECAY` | 0.01 | AdamW regularization standard |

### Results

| Epoch | Accuracy | Macro F1 |
|-------|----------|----------|
| 1 | 94.39% | 94.40% |
| 2 | **94.74%** | **94.75%** ← best model saved |
| 3 | 94.68% | 94.69% |

### How to Run
```bash
cd news-classifier
py -3.10 -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Train
python src/train.py

# Launch demo
streamlit run app/app.py
```

### Demo
The Streamlit app accepts any news headline or paragraph and returns the predicted category with confidence score.

```
Input : "Apple reports record quarterly earnings, beating analyst expectations"
Output: Category — Business | Confidence — 97.4%
```

---

## 📊 Task 2 — Customer Churn Prediction Pipeline

### Problem
Build an end-to-end, production-ready ML pipeline to predict which telecom customers are likely to churn, enabling proactive retention strategies.

### Dataset
[Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features including tenure, contract type, monthly charges, and service subscriptions.

### Approach
Two complete Scikit-learn pipelines (preprocessing → model) trained with `GridSearchCV` (5-fold CV, F1 scoring) to handle class imbalance.

### Project Structure
```
customer-churn-pipeline/
├── Data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Notebooks/
│   ├── 01_eda.ipynb          # Feature distributions, correlation heatmap
│   └── 02_evaluation.ipynb   # ROC curves, confusion matrices
├── src/
│   ├── config.py             # Centralized paths
│   ├── preprocessor.py       # ColumnTransformer (scaling + encoding)
│   ├── pipeline.py           # LR + RF pipeline builders
│   ├── train.py              # GridSearchCV training loop
│   └── predict.py            # Inference
├── models/
│   ├── churn_pipeline.pkl
│   └── plots/
│       ├── confusion_matrix.png
│       └── roc_curve.png
├── app/
│   └── app.py                # Streamlit demo
└── requirements.txt
```

### Pipeline Architecture

```
Raw CSV
  │
  ▼
fix_total_charges()           # TotalCharges: string → float
  │
  ▼
ColumnTransformer
  ├── Numerical → StandardScaler
  └── Categorical → OneHotEncoder
  │
  ▼
Classifier (LR or RF)
  │
  ▼
Churn Probability + Label
```

### Model Comparison

| Model | Best CV F1 | Notes |
|-------|-----------|-------|
| Logistic Regression | Baseline | Interpretable, fast |
| Random Forest | Higher F1 | Captures non-linear patterns |

### Key ML Concepts Applied
- Scikit-learn `Pipeline` for leak-proof preprocessing
- `GridSearchCV` with `scoring='f1'` (F1 chosen over accuracy due to class imbalance)
- `joblib` model serialization for deployment
- ROC-AUC evaluation for probabilistic thresholding

### How to Run
```bash
cd customer-churn-pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src/train.py
streamlit run app/app.py
```

---

## 🎫 Task 5 — Auto-Tagging Support Tickets Using LLMs

### Problem
Automatically assign topic tags (e.g. `ORDER`, `REFUND`, `PAYMENT`, `ACCOUNT`) to free-text customer support tickets, returning the **top-3 most probable tags** per ticket. Compare three LLM strategies to understand the trade-off between labeling cost and performance.

### Dataset
[Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — ~27,000 customer queries across ~11 high-level categories. Stratified 80/20 train-test split.

### Three-Way Approach Comparison

| Approach | Model | Training Data | Strategy |
|----------|-------|--------------|----------|
| Zero-Shot | `facebook/bart-large-mnli` | None | NLI hypothesis per tag; no training required |
| Few-Shot | SetFit (`all-MiniLM-L6-v2` + LR head) | 8 examples/class | Contrastive fine-tuning; purpose-built for low-data |
| Fine-Tuned | `distilbert-base-uncased` | Full train set | HuggingFace `Trainer` API; supervised upper bound |

### Project Structure
```
auto-tagging-tickets/
├── data/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── load_data.py
│   ├── zero_shot.py
│   ├── few_shot_setfit.py
│   ├── fine_tune_distilbert.py
│   └── evaluate.py
├── results/
│   └── confusion_matrices.png
├── models/
│   └── distilbert-finetuned/
├── app.py                    # Streamlit demo (all 3 modes)
└── requirements.txt
```

### Results

Evaluated on a balanced 550-sample subset (50 per class):

| Model | Top-1 Accuracy | Macro F1 | Top-3 Accuracy |
|-------|---------------|----------|----------------|
| Zero-Shot (BART-MNLI) | 47.27% | 0.4479 | 87.27% |
| Few-Shot (SetFit, 8/class) | **98.18%** | **0.9817** | **99.64%** |
| Fine-Tuned (DistilBERT) | **99.82%** | **0.9982** | **100.00%** |

**Fine-Tuned on full test set (4,927 samples):** Top-1 = 99.92% · Macro F1 = 0.9993 · Top-3 = 99.98%

### Key Findings

- **Zero-shot** hits 47% top-1 with zero labeling effort — reasonable for 11 classes, but struggles with semantically adjacent categories (`REFUND` vs `CANCEL`). Top-3 accuracy jumps to 87%, confirming the correct label is almost always within the model's top candidates.

- **Few-shot SetFit** with only **8 examples per class** achieves 98% top-1 — a **+51 percentage point jump** over zero-shot. This is the standout result: near fine-tuned performance at a fraction of the annotation cost.

- **Fine-tuned DistilBERT** is the supervised ceiling at 99.82% top-1, confirming the task is essentially solved with sufficient labeled data.

- Top-3 accuracy is near-perfect for both trained approaches, making top-3 ranking most meaningful as a UX feature for the zero-shot baseline.

### How to Run
```bash
cd auto-tagging-tickets
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src/load_data.py
python src/zero_shot.py
python src/few_shot_setfit.py
python src/fine_tune_distilbert.py   # GPU recommended
python src/evaluate.py

streamlit run app.py
```

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Deep Learning | PyTorch 2.5, HuggingFace Transformers, Datasets, Accelerate |
| Classical ML | Scikit-learn (Pipelines, GridSearchCV, ColumnTransformer) |
| NLP Models | BERT, DistilBERT, BART, SetFit / MiniLM |
| EDA & Visualization | Pandas, NumPy, Matplotlib, Seaborn |
| Deployment | Streamlit |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM, CUDA 13.2) |
| IDE | TRAE AI (VS Code based) |

---

## 💡 Skills Demonstrated

- **Transfer Learning & Fine-Tuning** — BERT, DistilBERT on downstream classification tasks
- **Zero-Shot & Few-Shot Learning** — NLI-based classification, SetFit contrastive fine-tuning
- **Production ML Pipelines** — Scikit-learn Pipelines, leak-proof train/test splits, GridSearchCV
- **NLP Preprocessing** — WordPiece tokenization, attention masks, sequence truncation/padding
- **Evaluation Methodology** — Accuracy, Macro F1, Top-3 Accuracy, ROC-AUC, confusion matrices
- **Mixed-Precision Training** — fp16 for GPU memory efficiency
- **Modular Code Design** — `config.py` single-source-of-truth, separation of concerns across `src/`
- **Model Deployment** — Streamlit apps with `@st.cache_resource` for efficient model loading

---

## 📁 Repository Structure

```
DevelopersHub-Internship/
├── Task-1-News-Classifier/        # BERT fine-tuning on AG News
├── Task-2-Churn-Prediction/       # Scikit-learn ML pipeline
├── Task-5-Ticket-Auto-Tagger/     # LLM comparison (zero/few/fine-tuned)
└── README.md                      # ← You are here
```

---

## 📬 Contact

**Adil**
BS Artificial Intelligence — UMT Lahore
AI/ML Engineering Intern @ DevelopersHub Corporation

> *"Every model starts with a config.py."*
