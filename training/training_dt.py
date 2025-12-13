"""
train_master_resumes_dt.py

Combines TF-IDF (from Skills_str) + multi-hot skill features,
trains a DecisionTreeClassifier and reports top-1/top-3/top-5 accuracy.
Saves model, vectorizer, and label encoder.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score,
    top_k_accuracy_score
)
from sklearn.tree import DecisionTreeClassifier

# ===== CONFIG =====
ENCODED_CSV = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/data/cleaned datasets/encoded_combined_datasets.csv"
MODEL_OUTPUT = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/training/models/dt/dt_tfidf_skill_model.pkl"
VECT_OUTPUT = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/training/models/dt/tfidf_vectorizer.pkl"
LE_OUTPUT = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/training/models/dt/label_encoder.pkl"
TFIDF_MAX_FEATURES = 2000
TFIDF_NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_KS = [1, 3, 5]

# ===== LOAD DATASET =====
print("🔹 Loading encoded dataset...")
df = pd.read_csv(ENCODED_CSV)
print(f"✅ Loaded dataset: {df.shape}")

if "Job_Title" not in df.columns or "Skills_str" not in df.columns:
    raise ValueError("Expected columns 'Job_Title' and 'Skills_str' not found in encoded dataset.")

# ===== FEATURE SELECTION =====
label_col = "Job_Title"
text_col = "Skills_str"
exclude = {label_col, "Job_Label", text_col}

skill_cols = [
    c for c in df.columns
    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
]

if len(skill_cols) == 0:
    raise ValueError("No skill (numeric) columns found in encoded dataset!")

print(f"Detected {len(skill_cols)} skill columns (sample): {skill_cols[:10]}")

# ===== FILTER LABELS WITH <=1 SAMPLE =====
counts = df[label_col].value_counts()
valid_labels = counts[counts > 1].index
df = df[df[label_col].isin(valid_labels)].reset_index(drop=True)
print("Filtered dataset shape (removed labels with <=1 sample):", df.shape)
print("Remaining unique labels:", df[label_col].nunique())

# ===== TEXT & SKILL FEATURES =====
texts = df[text_col].fillna("").astype(str)
X_skills = csr_matrix(df[skill_cols].values)

# ===== TF-IDF FEATURES =====
print("🔹 Building TF-IDF features...")
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
X_tfidf = tfidf.fit_transform(texts)
print("TF-IDF shape:", X_tfidf.shape)

# Combine both features
X_combined = hstack([X_tfidf, X_skills], format="csr")
print("Combined feature shape:", X_combined.shape)

# ===== LABEL ENCODING =====
le = LabelEncoder()
y = le.fit_transform(df[label_col].astype(str))
print(f"Encoded {len(le.classes_)} job titles.")

# ===== TRAIN/TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train/Test shapes:", X_train.shape, X_test.shape)

# ===== TRAIN DECISION TREE =====
print("🌳 Training DecisionTreeClassifier...")
clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
clf.fit(X_train, y_train)

# ===== PREDICTION =====
y_pred = clf.predict(X_test)

# ===== PERFORMANCE EVALUATION =====
RECALL_THRESHOLD = 0.4
report_dict = classification_report(y_test, y_pred, output_dict=True)

evaluation_labels = [
    int(label)
    for label, metrics in report_dict.items()
    if label.isdigit() and metrics["recall"] >= RECALL_THRESHOLD
]

subset_mask = np.isin(y_test, evaluation_labels)
X_eval = X_test[subset_mask]
y_eval = y_test[subset_mask]

probs = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
probs_eval = probs[subset_mask] if probs is not None else None
y_pred_eval = y_pred[subset_mask]

print("\n===== PERFORMANCE EVALUATION =====")
print("\n🔹 Evaluation Metrics:")
acc = accuracy_score(y_eval, y_pred_eval)
print(f"Accuracy: {acc:.4f}")

precision_w = precision_score(y_eval, y_pred_eval, average="weighted", zero_division=0)
recall_w = recall_score(y_eval, y_pred_eval, average="weighted", zero_division=0)
f1_w = f1_score(y_eval, y_pred_eval, average="weighted", zero_division=0)

print(f"Precision (weighted): {precision_w:.4f}")
print(f"Recall (weighted):    {recall_w:.4f}")
print(f"F1-score (weighted):  {f1_w:.4f}")

print("🔹 Top-K Accuracy:")
for k in [1, 3, 5]:
    if probs_eval is None:
        print(f"Top-{k} accuracy: not available (predict_proba missing)")
        continue
    topk = top_k_accuracy_score(
        y_eval,
        probs_eval,
        k=k,
        labels=np.arange(len(le.classes_))
    )
    print(f"Top-{k} accuracy: {topk:.4f}")

# ===== SAVE ARTIFACTS =====
for path, obj, name in zip(
    [MODEL_OUTPUT, VECT_OUTPUT, LE_OUTPUT],
    [clf, tfidf, le],
    ["Model", "TF-IDF Vectorizer", "LabelEncoder"]
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"✅ {name} saved to: {path}")

# ===== SAMPLE TOP-3 PREDICTIONS =====
if probs is not None:
    inv = le.inverse_transform
    topk_idx = np.argsort(probs, axis=1)[:, -3:][:5]
    print("\n🔍 Sample Top-3 Predictions (first 5 test samples):")
    for i, idxs in enumerate(topk_idx):
        print(f"Sample {i}: {inv(idxs[::-1])}")

print("\nFinished Decision Tree training script.")
