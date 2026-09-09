# data/feature_engineering.py
"""
Feature engineering for ScamGuard dataset.

- Adapts to available columns in data/dataset.csv
- Creates robust linguistic features from description (or a coalesced text)
- Adds metadata presence features (has_company_profile, has_requirements, has_benefits)
- Gracefully disables spelling feature if pyspellchecker isn't available
- Saves to data/engineered_features.csv

Run:
    python data/feature_engineering.py
"""

import os
import re
import pandas as pd

# ---------- Optional spelling support (pyspellchecker) ----------
HAS_SPELL = False
try:
    from spellchecker import SpellChecker  # provided by pyspellchecker
    _probe = SpellChecker()
    _ = _probe.known({"test"})
    HAS_SPELL = True
except Exception as e:
    print("⚠️  SpellChecker unavailable or wrong package installed. "
          "Spelling errors feature will be set to 0.\n   Detail:", e)
    SpellChecker = None

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else "."
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

IN_FILE = os.path.join(DATA_DIR, "dataset.csv")
OUT_FILE = os.path.join(DATA_DIR, "engineered_features.csv")

# ---------- Load ----------
df = pd.read_csv(IN_FILE)
print(f"Loaded: {IN_FILE}")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---------- Helpers ----------
def avg_sentence_length(text: str) -> float:
    sents = re.split(r"[.!?]", str(text))
    sents = [s.strip() for s in sents if s.strip()]
    return (sum(len(s.split()) for s in sents) / len(sents)) if sents else 0.0

def count_uppercase_words(text: str) -> int:
    return sum(1 for w in str(text).split() if w.isupper() and len(w) > 1)

if HAS_SPELL:
    _spell = SpellChecker()
    def fast_spelling_errors(text: str, limit: int = 100) -> int:
        words = str(text).split()[:limit]
        # pyspellchecker exposes a dictionary-like interface via 'in spell'
        return sum(1 for w in words if w.lower() not in _spell)
else:
    def fast_spelling_errors(text: str, limit: int = 100) -> int:
        return 0  # disabled gracefully

def coalesce_text(row: pd.Series) -> str:
    """
    Prefer 'description'; otherwise combine available informative fields.
    """
    desc = str(row.get("description", "") or "").strip()
    if desc:
        return desc

    candidates = [
        row.get("title", ""),
        row.get("company_profile", ""),
        row.get("requirements", ""),
        row.get("benefits", ""),
        row.get("location", ""),
        row.get("salary_range", ""),
        row.get("company_name", ""),
    ]
    return " ".join([str(x).strip() for x in candidates if str(x).strip()])

def has_text(series_or_df_col) -> pd.Series:
    """Return 1 if non-empty text, else 0. Accepts missing columns."""
    if series_or_df_col is None:
        return pd.Series([0] * len(df), index=df.index)
    return series_or_df_col.astype(str).apply(lambda s: 1 if s.strip() else 0)

# ---------- Prepare text-like columns (fill safely) ----------
text_like_cols = [
    "title", "company_profile", "description", "requirements", "benefits",
    "location", "salary_range", "company_name"
]
present_text_cols = [c for c in text_like_cols if c in df.columns]
if present_text_cols:
    df[present_text_cols] = df[present_text_cols].fillna("").replace(r"^\s*$", "", regex=True)

# Working text column used for all linguistic features
df["_text_for_feats"] = df.apply(coalesce_text, axis=1)

# ---------- Linguistic features ----------
print("\nCreating linguistic features...")
df["char_count"] = df["_text_for_feats"].apply(len)
df["word_count"] = df["_text_for_feats"].apply(lambda x: len(str(x).split()))
df["avg_word_length"] = df["_text_for_feats"].apply(
    lambda x: (sum(len(w) for w in str(x).split()) / max(1, len(str(x).split())))
)
df["num_exclamations"] = df["_text_for_feats"].apply(lambda x: str(x).count("!"))
df["num_dollar"] = df["_text_for_feats"].apply(lambda x: str(x).count("$"))
df["num_uppercase_words"] = df["_text_for_feats"].apply(count_uppercase_words)
df["avg_sentence_length"] = df["_text_for_feats"].apply(avg_sentence_length)
df["spelling_errors"] = df["_text_for_feats"].apply(fast_spelling_errors)
print("✅ Linguistic features created.")

# ---------- Metadata presence features ----------
print("\nAdding metadata presence features...")
df["has_company_profile"] = has_text(df["company_profile"]) if "company_profile" in df.columns else has_text(None)
df["has_requirements"] = has_text(df["requirements"]) if "requirements" in df.columns else has_text(None)
df["has_benefits"] = has_text(df["benefits"]) if "benefits" in df.columns else has_text(None)
print("✅ Metadata features added.")

# (Optional) Keep a combined_text column for downstream TF-IDF consistency
df["combined_text"] = df["_text_for_feats"]

# ---------- Save ----------
df.drop(columns=["_text_for_feats"], inplace=True)
df.to_csv(OUT_FILE, index=False)
print(f"\n✅ Feature engineering complete! Saved to: {OUT_FILE}")

# ---------- Verification ----------
new_features = [
    "char_count", "word_count", "avg_word_length",
    "num_exclamations", "num_dollar", "num_uppercase_words",
    "avg_sentence_length", "spelling_errors",
    "has_company_profile", "has_requirements", "has_benefits"
]
present_new = [c for c in new_features if c in df.columns]

print("\nSample of engineered features:")
print(df[present_new].head())

print("\nFeature summary statistics:")
print(df[present_new].describe())