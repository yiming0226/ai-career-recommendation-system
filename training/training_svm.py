"""
train_combined_datasets_svm2_fixed.py

Combines TF-IDF (from Skills_str) + multi-hot skill features,
applies chi2 feature selection, scales skill features,
trains LinearSVC with grid search, and reports Top-1/3/5 accuracy.
Saves model, vectorizer, and label encoder.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, top_k_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns

# ===== CONFIG =====
ENCODED_CSV = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/data/cleaned datasets/encoded_combined_datasets.csv"
MODEL_OUTPUT = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/training/models/svm3/svm_tfidf_skill_model_fixed.pkl"
VECT_OUTPUT = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/training/models/svm3/tfidf_vectorizer_fixed.pkl"
LE_OUTPUT = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/training/models/svm3/label_encoder_fixed.pkl"
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
print("🔹 Fitting TF-IDF vectorizer on training text...")
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
X_tfidf = tfidf.fit_transform(texts)
print("TF-IDF shape:", X_tfidf.shape)

# ===== χ² FEATURE SELECTION =====
print(f"🔹 Selecting top {X_tfidf.shape[1]} TF-IDF features with chi2...")
selector = SelectKBest(chi2, k=X_tfidf.shape[1])
X_tfidf_sel = selector.fit_transform(X_tfidf, df[label_col])
print("Selected TF-IDF shape:", X_tfidf_sel.shape)

# ===== SCALE SKILL FEATURES =====
print("🔹 Scaling multi-hot skill features...")
scaler = StandardScaler(with_mean=False)
X_skills_scaled = scaler.fit_transform(X_skills)

# ===== COMBINE FEATURES =====
print("🔹 Combining TF-IDF and skill features...")
X_combined = hstack([X_tfidf_sel, X_skills_scaled * 0.5], format="csr")
print("Combined shape:", X_combined.shape)

# ===== LABEL ENCODING =====
le = LabelEncoder()
y = le.fit_transform(df[label_col])
print(f"Encoded {len(le.classes_)} job titles.")

# ===== TRAIN/TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train/Test counts: {X_train.shape[0]} {X_test.shape[0]}")

# ===== GRID SEARCH FOR BEST C =====
param_grid = {"C": [0.01, 0.1, 1.0, 5.0, 10.0]}
svc = LinearSVC(
    penalty="l2",
    max_iter=20000,
    tol=1e-3,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
best_C = grid.best_params_["C"]
print(f"✅ Best C from GridSearch: {best_C} (mean cv score: {grid.best_score_:.4f})")

# ===== TRAIN FINAL SVM =====
clf = LinearSVC(
    penalty="l2",
    C=best_C,
    max_iter=20000,
    tol=1e-3,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
clf.fit(X_train, y_train)

# ===== CALIBRATE SVM =====
print("🔹 Calibrating SVM with CalibratedClassifierCV (sigmoid)...")
calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
calibrated.fit(X_train, y_train)

# ===== PREDICTION =====
y_pred = calibrated.predict(X_test)
probs = calibrated.predict_proba(X_test)

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
y_pred_eval = y_pred[subset_mask]
probs_eval = probs[subset_mask]

print("\n🔹 Evaluation Metrics:")
acc = accuracy_score(y_eval, y_pred_eval)
print(f"Accuracy: {acc:.4f}")

precision_w = precision_score(y_eval, y_pred_eval, average="weighted", zero_division=0)
recall_w = recall_score(y_eval, y_pred_eval, average="weighted", zero_division=0)
f1_w = f1_score(y_eval, y_pred_eval, average="weighted", zero_division=0)

print(f"Precision (weighted): {precision_w:.4f}")
print(f"Recall (weighted):    {recall_w:.4f}")
print(f"F1-score (weighted):  {f1_w:.4f}")

# ===== TOP-K METRICS =====
print("🔹 Top-K Accuracy:")
for k in [1, 3, 5]:
    topk = top_k_accuracy_score(
        y_eval,
        probs_eval,
        k=k,
        labels=np.arange(len(le.classes_))
    )
    print(f"Top-{k} accuracy: {topk:.4f}")

# ===== CONFUSION MATRIX =====
cm_raw = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
n_labels = cm_raw.shape[0]
row_sums = cm_raw.sum(axis=1, keepdims=True).astype(float)
cm_norm = cm_raw.astype(float) / (row_sums + 1e-9)
eps = 1e-6
cm_norm_eps = cm_norm + eps
cm_log = np.log1p(cm_norm_eps * 100.0)

annot = np.empty_like(cm_norm, dtype=object)
for i in range(n_labels):
    for j in range(n_labels):
        annot[i, j] = f"{cm_norm[i, j]:.2f}"

plt.figure(figsize=(20, 16))
sns.heatmap(
    cm_norm,
    cmap="Reds",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    annot=annot,
    annot_kws={"size": 5},
    fmt="",
    cbar_kws={"label": "fraction (row-normalized)"},
    linewidths=0.3,
    linecolor="lightgray"
)
output_dir = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/data/output"
os.makedirs(output_dir, exist_ok=True)

plt.title("Confusion matrix (row-normalized, log-scaled) — fraction of true-class", fontsize=16)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_full.png"), dpi=150)
plt.show()

# ===== TOP CONFUSED PAIRS =====
pairs = []
for i in range(n_labels):
    for j in range(n_labels):
        if i != j and cm_raw[i, j] > 0:
            pairs.append((int(cm_raw[i, j]), le.classes_[i], le.classes_[j]))

pairs_sorted = sorted(pairs, reverse=True, key=lambda x: x[0])
top_k_pairs = 20
print(f"\nTop {min(top_k_pairs, len(pairs_sorted))} confused pairs (count, true -> predicted):")
for cnt, true_lbl, pred_lbl in pairs_sorted[:top_k_pairs]:
    print(f"{cnt:3d}   {true_lbl}  ->  {pred_lbl}")

# ===== SAVE ARTIFACTS =====
for path, obj, name in zip(
    [MODEL_OUTPUT, VECT_OUTPUT, LE_OUTPUT],
    [calibrated, tfidf, le],
    ["Calibrated Model", "TF-IDF Vectorizer", "LabelEncoder"]
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"✅ {name} saved to: {path}")

print("\nFinished SVM training script.")
