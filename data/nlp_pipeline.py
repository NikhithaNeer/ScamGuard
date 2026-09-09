import os, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Prefer engineered features if present
eng_path = os.path.join(DATA_DIR, "engineered_features.csv")
raw_path = os.path.join(DATA_DIR, "dataset.csv")
df = pd.read_csv(eng_path if os.path.exists(eng_path) else raw_path)

# Choose text source
text_col = "combined_text" if "combined_text" in df.columns else "description"
df[text_col] = df[text_col].fillna("").astype(str)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)
X_text = tfidf.fit_transform(df[text_col])

# Save vectorizer
with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

# Optionally save a TF-IDF matrix preview for debugging
tfidf_df = pd.DataFrame.sparse.from_spmatrix(X_text, columns=tfidf.get_feature_names_out())
if "fraudulent" in df.columns:
    tfidf_df["fraudulent"] = df["fraudulent"].values
tfidf_df.to_csv(os.path.join(DATA_DIR, "tfidf_features.csv"), index=False)

print("TF-IDF ready:", X_text.shape, "using", text_col)