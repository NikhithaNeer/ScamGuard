# data/base_model.py
import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix, hstack

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ENG_PATH = os.path.join(DATA_DIR, "engineered_features.csv")
if not os.path.exists(ENG_PATH):
    raise FileNotFoundError(f"Missing file: {ENG_PATH}")

# ---------------- Load data ----------------
eng = pd.read_csv(ENG_PATH)

# Target
if "fraudulent" not in eng.columns:
    raise ValueError("'fraudulent' column not found in engineered_features.csv")
y = eng["fraudulent"].astype(int).values
if np.isnan(y).any():
    raise ValueError("Target y contains NaN. Clean your 'fraudulent' column.")

# Text source for TF-IDF
text_col = "combined_text" if "combined_text" in eng.columns else "description"
if text_col not in eng.columns:
    raise ValueError("Neither 'combined_text' nor 'description' present in engineered_features.csv")
eng[text_col] = eng[text_col].fillna("").astype(str)

# ---------------- Select numeric engineered features ----------------
# Exclude any raw text/helper columns from the numeric engineered block
TEXT_OR_HELPER = {
    "combined_text", "title", "company_profile", "description", "requirements",
    "benefits", "location", "salary_range", "company_name", "url", "contact_email"
}
DROP_COLS = {"fraudulent"} | TEXT_OR_HELPER

eng_cols = [
    c for c in eng.columns
    if c not in DROP_COLS and pd.api.types.is_numeric_dtype(eng[c])
]
if not eng_cols:
    raise ValueError("No numeric engineered columns found after filtering. Check your CSV.")

# Replace +/-inf with NaN then impute NaN with 0.0 (good for counts/flags)
eng[eng_cols] = eng[eng_cols].replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy="constant", fill_value=0.0)
X_eng = imputer.fit_transform(eng[eng_cols].to_numpy(dtype=float))

# ---------------- TF-IDF from same rows ----------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
X_text = tfidf.fit_transform(eng[text_col])

# Row alignment check
if X_text.shape[0] != X_eng.shape[0]:
    raise RuntimeError(f"Row mismatch: X_eng={X_eng.shape[0]}, X_text={X_text.shape[0]}")

# Combine (engineered dense -> sparse for hstack with TF-IDF)
X = hstack([csr_matrix(X_eng), X_text]).tocsr()

# ---------------- Train / Eval ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=2000, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")

models = [("LR", lr), ("RF", rf)]
best_name, best_model, best_acc = None, None, -1.0

for name, clf in models:
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    print(f"\n=== {name} ===")
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_hat, digits=3))
    if acc > best_acc:
        best_name, best_model, best_acc = name, clf, acc

print(f"\nBest model: {best_name} (acc={best_acc:.4f})")

# ---------------- Persist artifacts ----------------
with open(os.path.join(MODELS_DIR, "model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

manifest = {
    "engineered_cols": eng_cols,     # exact order used during training
    "text_col": text_col,
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": [1, 2],
    "tfidf_min_df": 3,
}
with open(os.path.join(MODELS_DIR, "features_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("\nSaved:")
print(" - models/model.pkl")
print(" - models/tfidf_vectorizer.pkl")
print(" - models/features_manifest.json")
print(f"Final X shape: rows={X.shape[0]}, cols={X.shape[1]}")
print(f"Engineered block: {len(eng_cols)} cols | TF-IDF block: {X.shape[1]-len(eng_cols)} cols")