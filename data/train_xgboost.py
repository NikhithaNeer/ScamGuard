# data/train_xgboost.py
import os, re, pickle, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ENGINEERED_PATH = os.path.join(DATA_DIR, "engineered_features.csv")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

# --------------- Load data ---------------
df = pd.read_csv(ENGINEERED_PATH)

# Safety fills
for c in ["title","company_profile","description","requirements","benefits"]:
    if c not in df.columns:
        df[c] = ""
df = df.fillna({"title":"", "company_profile":"", "description":"", "requirements":"", "benefits":""})

# target
y = df["fraudulent"].astype(int).values

# text soup exactly like app combines
def combine_text(r):
    return " ".join([str(r["title"]), str(r["company_profile"]), str(r["description"]),
                     str(r["requirements"]), str(r["benefits"])]).strip()

text_corpus = df.apply(combine_text, axis=1)

# --------------- Load TF-IDF ---------------
with open(VECTORIZER_PATH, "rb") as f:
    tfidf = pickle.load(f)
X_text = tfidf.transform(text_corpus)

# --------------- Engineered features ---------------
# Use same engineered columns the app expects (everything except target)
engineered_cols = [c for c in df.columns if c != "fraudulent"]
# Remove raw text columns to avoid leakage/duplication (since we use TF-IDF)
for raw in ["title","company_profile","description","requirements","benefits"]:
    if raw in engineered_cols:
        engineered_cols.remove(raw)

X_eng = df[engineered_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

# --------------- Combine features ---------------
# Note: X_text is sparse, X_eng is dense → convert eng to sparse for hstack-like effect
from scipy.sparse import hstack, csr_matrix
X = hstack([csr_matrix(X_eng), X_text]).tocsr()

# --------------- Split ---------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------- Handle imbalance ---------------
# scale_pos_weight ~ (neg/pos)
pos = y_train.sum()
neg = len(y_train) - pos
spw = (neg / max(pos, 1))

# --------------- Train XGBoost ---------------
clf = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.06,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=spw,
    objective="binary:logistic",
    eval_metric="logloss",
)
clf.fit(X_train, y_train)

# --------------- Evaluate ---------------
proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("\n=== XGBoost Results ===")
print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
print(f"Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
print("\nReport:\n", classification_report(y_test, pred, digits=4))

# Optional: find better threshold via Youden’s J on ROC
from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(y_test, proba)
youden = tpr - fpr
best_idx = int(np.argmax(youden))
best_thr = float(thr[best_idx])
print(f"\nSuggested threshold (Youden): {best_thr:.3f}")

# --------------- Save model + meta ---------------
MODEL_OUT = os.path.join(MODELS_DIR, "model_xgb.pkl")
META_OUT = os.path.join(MODELS_DIR, "xgb_meta.pkl")
with open(MODEL_OUT, "wb") as f:
    pickle.dump(clf, f)

meta = {
    "engineered_cols": engineered_cols,
    "n_text_features": X_text.shape[1],
    "threshold": best_thr
}
with open(META_OUT, "wb") as f:
    pickle.dump(meta, f)

print(f"\nSaved: {MODEL_OUT}")
print(f"Saved: {META_OUT}")