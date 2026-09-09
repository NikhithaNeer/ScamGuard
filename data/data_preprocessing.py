import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------- Paths ----------
BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR) if BASE_DIR.endswith("data") else BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "dataset.csv")

# ---------- Load ----------
df = pd.read_csv(DATA_FILE)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3))
print(df.info())

# ---------- Define columns (only keep those that exist) ----------
# Text-like fields you want to preserve for later NLP / display
candidate_text_cols = [
    "title", "company_profile", "description", "requirements", "benefits",
    "location", "salary_range", "company_name", "contact_email", "url"
]
text_cols = [c for c in candidate_text_cols if c in df.columns]

# Categorical fields to label-encode (short vocab, not free text)
candidate_cat_cols = [
    "employment_type", "required_experience", "required_education", "industry"
]
cat_cols = [c for c in candidate_cat_cols if c in df.columns]

# Binary fields (should be 0/1)
candidate_binary_cols = ["telecommuting", "has_company_logo", "has_questions"]
binary_cols = [c for c in candidate_binary_cols if c in df.columns]

# ---------- Clean NA ----------
# Replace whitespace-only with empty strings for text fields
if text_cols:
    df[text_cols] = df[text_cols].replace(r"^\s*$", "", regex=True)
    df[text_cols] = df[text_cols].fillna("")

# Fill NA for categoricals with a marker
if cat_cols:
    df[cat_cols] = df[cat_cols].fillna("Unknown")

# Ensure binaries are ints (fill NA with 0)
if binary_cols:
    for col in binary_cols:
        df[col] = df[col].fillna(0).astype(int)

print("Missing after fill:", int(df.isnull().sum().sum()))

# ---------- Encode categoricals ----------
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ---------- Split ----------
if "fraudulent" not in df.columns:
    raise ValueError("Expected a 'fraudulent' target column in dataset.csv")

X = df.drop(columns=["fraudulent"])
y = df["fraudulent"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------- Save ----------
X_train.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(DATA_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(DATA_DIR, "y_test.csv"), index=False)

print("Preprocessing completed")

# ---------- Quick checks ----------
print("Total missing values after preprocessing:", int(df.isnull().sum().sum()))

if cat_cols:
    print("\nSample of encoded categorical columns:")
    print(df[cat_cols].head())

if binary_cols:
    print("\nBinary column data types:")
    print(df[binary_cols].dtypes)

print("\nFraud ratio in train:", y_train.mean())
print("Fraud ratio in test:", y_test.mean())

saved = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
print("\nSaved files (in /data):", saved)