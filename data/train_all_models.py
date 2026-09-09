# data/train_all_models.py
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    precision_recall_fscore_support, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# XGBoost is optional; if not installed you can still train LR/RF
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    warnings.warn("xgboost not installed: will skip XGB training")

# ---------------- Paths ----------------
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ENG_PATH = os.path.join(DATA_DIR, "engineered_features.csv")
if not os.path.exists(ENG_PATH):
    raise FileNotFoundError(f"Missing file: {ENG_PATH}")

# ---------------- Load data ----------------
df = pd.read_csv(ENG_PATH)

# Target
if "fraudulent" not in df.columns:
    raise ValueError("'fraudulent' column not found in engineered_features.csv")
y = df["fraudulent"].astype(int).values
if np.isnan(y).any():
    raise ValueError("Target y contains NaN. Clean your 'fraudulent' column.")

# ---------------- Text column & soup ----------------
# Prefer a pre-built 'combined_text' if it exists; otherwise use description-only
text_col = "combined_text" if "combined_text" in df.columns else "description"
if text_col not in df.columns:
    raise ValueError("Neither 'combined_text' nor 'description' present in engineered_features.csv")
df[text_col] = df[text_col].fillna("").astype(str)

# ---------------- Numeric engineered features ----------------
TEXT_OR_HELPER = {
    "combined_text", "title", "company_profile", "description", "requirements",
    "benefits", "location", "salary_range", "company_name", "url", "contact_email"
}
DROP_COLS = {"fraudulent"} | TEXT_OR_HELPER

eng_cols = [
    c for c in df.columns
    if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])
]
if not eng_cols:
    raise ValueError("No numeric engineered columns found after filtering. Check your CSV.")

# Replace +/-inf with NaN then impute NaN with 0.0 (for counts/flags)
df[eng_cols] = df[eng_cols].replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy="constant", fill_value=0.0)
X_eng = imputer.fit_transform(df[eng_cols].to_numpy(dtype=float))

# ---------------- TF-IDF ----------------
# Fit TF-IDF on the *same* corpus used in the app (text_col)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
X_text = tfidf.fit_transform(df[text_col])

# Row alignment sanity check
if X_text.shape[0] != X_eng.shape[0]:
    raise RuntimeError(f"Row mismatch: X_eng={X_eng.shape[0]}, X_text={X_text.shape[0]}")

# Combine (engineered dense -> sparse for hstack with TF-IDF)
X = hstack([csr_matrix(X_eng), X_text]).tocsr()
n_eng = len(eng_cols)
n_txt = X.shape[1] - n_eng

print(f"\n[Build] rows={X.shape[0]}  cols={X.shape[1]}  (eng={n_eng}, tfidf={n_txt})")

# ---------------- Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------------- Helpers ----------------
def youden_threshold(y_true, proba):
    """Youden’s J statistic: max(tpr - fpr) on ROC to choose a good threshold."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])

def eval_and_print(name, proba, y_true):
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred)
    auc = roc_auc_score(y_true, proba)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    print(classification_report(y_true, pred, digits=4))
    thr = youden_threshold(y_true, proba)
    print(f"Suggested threshold (Youden): {thr:.3f}")
    return {"acc": acc, "auc": auc, "p": p, "r": r, "f1": f1, "thr": thr}

# ---------------- Train: Logistic Regression ----------------
lr = LogisticRegression(max_iter=2000, n_jobs=-1)
lr.fit(X_train, y_train)
lr_proba = lr.predict_proba(X_test)[:, 1]
lr_stats = eval_and_print("LogisticRegression", lr_proba, y_test)

# ---------------- Train: Random Forest ----------------
rf = RandomForestClassifier(
    n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_stats = eval_and_print("RandomForest", rf_proba, y_test)

# ---------------- Train: XGBoost (optional) ----------------
xgb_stats = None
if HAS_XGB:
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = (neg / max(pos, 1))

    xgb = XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9,
        min_child_weight=1.0, reg_alpha=0.0, reg_lambda=1.0,
        tree_method="hist", random_state=42, n_jobs=-1,
        scale_pos_weight=spw, objective="binary:logistic",
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_stats = eval_and_print("XGBoost", xgb_proba, y_test)

# ---------------- Simple soft vote on validation ----------------
vote_names = ["LogReg", "RandomForest"]
vote_prob_parts = [lr_proba, rf_proba]
weights = [0.5, 0.5]  # default equal weights

if HAS_XGB:
    vote_names.append("XGBoost")
    vote_prob_parts.append(xgb_proba)
    weights = [1/3, 1/3, 1/3]

vote_prob = np.average(np.vstack(vote_prob_parts), axis=0, weights=weights)
vote_stats = eval_and_print("SoftVote (equal weights)", vote_prob, y_test)

# ---------------- Persist artifacts ----------------
# Individual models
with open(os.path.join(MODELS_DIR, "logreg.pkl"), "wb") as f:
    pickle.dump(lr, f)

with open(os.path.join(MODELS_DIR, "rf.pkl"), "wb") as f:
    pickle.dump(rf, f)

if HAS_XGB:
    with open(os.path.join(MODELS_DIR, "xgb.pkl"), "wb") as f:
        pickle.dump(xgb, f)

# TF-IDF
with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

# Manifest for reproducibility
manifest = {
    "engineered_cols": eng_cols,
    "text_col": text_col,
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": [1, 2],
    "tfidf_min_df": 3,
    "n_eng_features": n_eng,
    "n_text_features": n_txt
}
with open(os.path.join(MODELS_DIR, "features_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# Voting meta (used by app.py for weighting + threshold)
vote_meta = {
    "models": vote_names,
    "weights": weights,
    "threshold": vote_stats["thr"],  # Youden on soft-vote
    "per_model_thresholds": {
        "LogReg": lr_stats["thr"],
        "RandomForest": rf_stats["thr"],
        **({"XGBoost": xgb_stats["thr"]} if HAS_XGB else {})
    },
    "val_metrics": {
        "LogReg": lr_stats,
        "RandomForest": rf_stats,
        **({"XGBoost": xgb_stats} if HAS_XGB else {}),
        "SoftVote": vote_stats
    }
}
with open(os.path.join(MODELS_DIR, "vote_meta.json"), "w") as f:
    json.dump(vote_meta, f, indent=2)

print("\nSaved:")
print(" - models/logreg.pkl")
print(" - models/rf.pkl")
if HAS_XGB:
    print(" - models/xgb.pkl")
print(" - models/tfidf_vectorizer.pkl")
print(" - models/features_manifest.json")
print(" - models/vote_meta.json")