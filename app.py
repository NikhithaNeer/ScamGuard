# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
import os, re, json, pickle, numpy as np, pandas as pd
from dotenv import load_dotenv
from functools import wraps
from datetime import datetime
from supabase import create_client
from scipy.sparse import hstack, csr_matrix
from data.text_extractor import extract_job_features
from data.db import init_db, insert_log, fetch_logs, delete_log, save_review_answers, fetch_review_answers_for_logs, get_admin_stats

# db helpers
from data.db import (
    init_db,
    insert_log,
    fetch_logs,
    delete_log,
    count_needs_review,
    mark_review_decision
)

EXAMPLE_SCAM_TEXT_1 = """
      Immediate Hiring – Remote Data Entry (No Experience Needed)
      We are urgently hiring Data Entry workers! Earn $800 daily by filling simple forms online.
      No experience required. Just send your full name, date of birth, and a copy of your ID card
      for verification. Payments are made instantly via PayPal or CashApp.
      Limited openings — apply fast!
    """

EXAMPLE_SCAM_TEXT_2 = """
      Customer Service Representative – Work From Home
      We are a new global crypto company. We pay $500 per day for answering simple emails.
      Training provided. To proceed, send your bank account number and routing number 
      so we can verify your payment profile. Your first payment will be released immediately.
    """

EXAMPLE_SCAM_TEXT_3 = """
      Amazon Package Handler (Remote)
      We are partnering with Amazon! Earn $700/day by labeling packages from home.
      All you need is a smartphone. To activate your account, pay a $50 security deposit.
      Refundable after your first week. Message on WhatsApp for fast hiring.
    """
EXAMPLE_SCAM_TEXT_4 = """
      Part-Time Virtual Assistant
      Looking for individuals to help CEOs with scheduling tasks. Salary is $1200/week.
      To begin, confirm your identity by emailing a photo of your driver’s license and social security number.
      Only serious applicants needed.
    """
EXAMPLE_SCAM_TEXT_5 = """
      Urgent Online Typing Job – $1000 Daily
      We provide custom form entry tasks. No skills required. You only need a phone.
      Before assigning work, you must pay a $30 registration fee to unlock your worker ID.
      Payments are instant after ID activation.
    """
EXAMPLE_REAL_TEXT_1 = """
      Title: Software Engineer – Backend (Python)
      Description: We are seeking a Backend Engineer with 3+ years of experience in Python, Flask, or Django.
      Responsibilities include designing scalable REST APIs, writing clean code, and collaborating with cross-functional teams.
      Requirements: Strong knowledge of SQL, version control (Git), CI/CD pipelines.
      Benefits: Health insurance, 401(k), PTO, hybrid workplace options.
    """

EXAMPLE_REAL_TEXT_2 = """
      Customer Support Specialist
      We are expanding our support operations and looking for someone with strong problem-solving skills.
      Responsibilities include responding to customer inquiries, documenting issues, and escalating technical cases to the engineering team.
      Benefits: Health insurance, paid vacation, remote friendly.
    """

EXAMPLE_REAL_TEXT_3 = """
      Data Analyst – Entry Level
      We are seeking a motivated Data Analyst skilled in Excel, SQL, and dashboard reporting tools.
      Responsibilities: Build weekly reports, automate workflows, and maintain data quality.
      Company Profile: We are a mid-sized analytics company serving Fortune 500 clients.
    """

EXAMPLE_REAL_TEXT_4 = """
      Registered Nurse (RN)
      General Hospital is hiring registered nurses for the internal medicine department.
      Requirements: Valid RN license, strong patient care skills, ability to work in a team environment.
      Benefits: Health insurance, retirement plan, paid overtime, shift differentials.
    """

PLAYGROUND_EXAMPLES = [
    {"title": "Remote Data Entry – Scam", "label": "Fake", "text": EXAMPLE_SCAM_TEXT_1},
    {"title": "Crypto Email Agent – Scam", "label": "Fake", "text": EXAMPLE_SCAM_TEXT_2},
    {"title": "Amazon Package Handler Scam", "label": "Fake", "text": EXAMPLE_SCAM_TEXT_3},
    {"title": "Virtual Assistant Scam", "label": "Fake", "text": EXAMPLE_SCAM_TEXT_4},
    {"title": "Typing Job Deposit Scam", "label": "Fake", "text": EXAMPLE_SCAM_TEXT_5},

    {"title": "Backend Python Engineer", "label": "Real", "text": EXAMPLE_REAL_TEXT_1},
    {"title": "Customer Support Specialist", "label": "Real", "text": EXAMPLE_REAL_TEXT_2},
    {"title": "Data Analyst – Entry Level", "label": "Real", "text": EXAMPLE_REAL_TEXT_3},
    {"title": "Registered Nurse (RN)", "label": "Real", "text": EXAMPLE_REAL_TEXT_4},
    ]

from auth import auth as auth_blueprint
import auth as auth_module
from account import account as account_blueprint

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Supabase client initialized successfully")
else:
    supabase = None
    print("✗ Supabase URL or KEY not found in .env")

auth_module.supabase = supabase

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key_here")

app.register_blueprint(auth_blueprint, url_prefix="/auth")
app.register_blueprint(account_blueprint, url_prefix="/account")

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in first!", "error")
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session or int(session.get("permission_level", 0)) < 1:
            flash("Access denied: admin privileges required.", "error")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return wrapper

@app.route("/login")
def home():
    if 'user_id' in session:
        return redirect(url_for("index"))
    return redirect(url_for("auth.login"))

@app.route("/dashboard")
@login_required
def dashboard():
    level = int(session.get("permission_level", 0))
    if level >= 1:
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("user_dashboard"))

@app.route("/user-dashboard")
@login_required
def user_dashboard():
    username = session.get("username")
    user_id = session.get("user_id", "anonymous")
    pending_review = count_needs_review(user_id)

    return render_template(
        "user_dashboard.html",
        username=username,
        pending_review=pending_review
    )

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data")
MODELS_PATH = os.path.join(BASE_DIR, "models")
ENGINEERED_PATH = os.path.join(DATA_PATH, "engineered_features.csv")

def _art(name): 
    return os.path.join(MODELS_PATH, name)


with open(_art("features_manifest.json"), "r") as f:
    manifest = json.load(f)
engineered_cols = manifest["engineered_cols"]
n_text_features = int(manifest.get("n_text_features", 0))

with open(_art("tfidf_vectorizer.pkl"), "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(_art("vote_meta.json"), "r") as f:
    vote_meta = json.load(f)

MODEL_ORDER = [m for m in vote_meta.get("models", [])]
VOTE_WEIGHTS = np.array(vote_meta.get("weights", [1 / max(len(MODEL_ORDER), 1)] * len(MODEL_ORDER)), dtype=float)
DEFAULT_THRESHOLD = float(vote_meta.get("threshold", 0.5))
REVIEW_BAND = float(os.getenv("REVIEW_BAND", "0.06"))


_thr = os.getenv("VOTE_THRESHOLD")
if _thr:
    try: DEFAULT_THRESHOLD = float(_thr)
    except: pass
_w = os.getenv("VOTE_WEIGHTS") 
if _w:
    try:
        arr = np.array([float(x.strip()) for x in _w.split(",")], dtype=float)
        if len(arr) == len(MODEL_ORDER):
            VOTE_WEIGHTS = arr
    except:
        pass

MODEL_FILES = {
    "LogReg": "logreg.pkl",
    "RandomForest": "rf.pkl",
    "XGBoost": "xgb.pkl",
}
models = {}
for name in MODEL_ORDER:
    fname = MODEL_FILES.get(name)
    if not fname: 
        continue
    path = _art(fname)
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
if not models:
    raise RuntimeError("No models loaded. Train models and ensure artifacts exist in /models.")

print(f"[Voting] Loaded models: {list(models.keys())}")
print(f"[Voting] Weights: {VOTE_WEIGHTS.tolist()} | Threshold: {DEFAULT_THRESHOLD}")

engineered_df = pd.read_csv(ENGINEERED_PATH)

# ========================= Helpers =========================
def clean_text_basic(text: str) -> str:
    """
    Simple normalization to match common TF-IDF training:
    - lowercase
    - strip HTML-like tags
    - keep letters/numbers/basic punctuation
    - collapse whitespace
    """
    t = str(text).lower()
    t = re.sub(r"<.*?>", " ", t)                 # remove tags
    t = re.sub(r"[^a-z0-9$%€£.,:;!?()\-\/\s]", " ", t)  # keep common symbols
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_job_posting(text: str):
    """Extract simple sections if user pasted a structured post."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    full_text = "\n".join(lines)
    title = lines[0] if lines else " ".join(text.split()[:5])

    def extract_section(keyword: str, block: str):
        m = re.search(rf"{keyword}\s*[:\-]\s*(.*?)(?=\n[A-Z][A-Za-z ]+:\s*|\Z)",
                      block, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    company_profile = extract_section("Company", full_text)
    description = extract_section("Description", full_text)
    requirements = extract_section("Requirements", full_text)
    benefits = extract_section("Benefits", full_text)
    return title, company_profile, description, requirements, benefits

def make_engineered_row(title, company_profile, description, requirements, benefits):
    """Return a 1D np.array matching the engineered_cols order."""
    desc = str(description or "").strip()
    words = desc.split()
    sentences = [s.strip() for s in re.split(r"[.!?]", desc) if s.strip()]

    row = {c: 0.0 for c in engineered_cols}

    # Derived features (only fill if present in trained set)
    if "char_count" in row: row["char_count"] = len(desc)
    if "word_count" in row: row["word_count"] = len(words)
    if "avg_word_length" in row:
        row["avg_word_length"] = (sum(len(w) for w in words) / len(words)) if words else 0.0
    if "num_exclamations" in row: row["num_exclamations"] = desc.count("!")
    if "num_dollar" in row: row["num_dollar"] = desc.count("$")
    if "num_uppercase_words" in row:
        row["num_uppercase_words"] = sum(1 for w in words if w.isupper() and len(w) > 1)
    if "avg_sentence_length" in row:
        row["avg_sentence_length"] = (sum(len(s.split()) for s in sentences)/len(sentences)) if sentences else 0.0
    if "spelling_errors" in row: row["spelling_errors"] = 0.0
    if "has_company_profile" in row: row["has_company_profile"] = int(bool(str(company_profile).strip()))
    if "has_requirements" in row: row["has_requirements"] = int(bool(str(requirements).strip()))
    if "has_benefits" in row: row["has_benefits"] = int(bool(str(benefits).strip()))

    # Any extra numeric engineered columns—fallback to median if they exist in the file
    for c in engineered_cols:
        if c not in row:
            if c in engineered_df.columns:
                ser = pd.to_numeric(engineered_df[c], errors="coerce")
                med = ser.median()
                row[c] = float(med) if pd.notna(med) else 0.0
            else:
                row[c] = 0.0

    return np.asarray([row[c] for c in engineered_cols], dtype=float)

def predict_with_ensemble(title, company_profile, description, requirements, benefits):
    """
    Build features -> predict with each model -> soft vote -> return:
    (final_label, final_conf_pct, per_model_probs{ name: proba }, final_proba)
    """
    # engineered block (1 x n_eng)
    x_eng = make_engineered_row(title, company_profile, description, requirements, benefits).reshape(1, -1)

    # text block (TF-IDF)
    combined_text = " ".join([
        str(title or ""), str(company_profile or ""), str(description or ""),
        str(requirements or ""), str(benefits or "")
    ]).strip()
    combined_text = clean_text_basic(combined_text)
    x_text = tfidf_vectorizer.transform([combined_text])

    # combine (eng dense -> sparse to match training)
    X = hstack([csr_matrix(x_eng), x_text]).tocsr()

    # per-model probabilities
    per_model = {}
    probs = []
    for name in MODEL_ORDER:
        clf = models.get(name)
        if clf is None:
            continue
        try:
            p = float(clf.predict_proba(X)[0, 1])
        except Exception:
            p = float(clf.predict(X)[0])
        per_model[name] = p
        probs.append(p)

    # weights (ensure valid length)
    if len(probs) == len(VOTE_WEIGHTS):
        weights = VOTE_WEIGHTS
    else:
        weights = np.ones(len(probs), dtype=float) / max(len(probs), 1)

    final_proba = float(np.average(np.asarray(probs), weights=weights)) if probs else 0.5
    label = 1 if final_proba >= DEFAULT_THRESHOLD else 0
    confidence_pct = round((final_proba if label == 1 else 1 - final_proba) * 100, 2)

    return label, confidence_pct, per_model, final_proba

@app.route("/index", methods=["GET", "POST"])
@login_required
def index():
    prediction = None
    confidence = None
    input_text = None
    error_message = None
    extracted = None
    per_model = {}
    final_proba = None

    if request.method == "POST":
        raw_text = request.form.get("job_posting", "").strip()
        input_text = raw_text
        extracted = extract_job_features(raw_text)

        if not raw_text:
            error_message = "Please paste a job posting first."
        else:
            title, company_profile, description, requirements, benefits = parse_job_posting(raw_text)

            label, confidence, per_model, final_proba = predict_with_ensemble(
                title, company_profile, description, requirements, benefits
            )

            # final_proba = P(fake)
            needs_review = False
            if final_proba is not None:
                needs_review = abs(final_proba - DEFAULT_THRESHOLD) <= REVIEW_BAND

            if needs_review:
                ui_text = "🟡 Needs Review"
                stored_label = "Needs Review"
            else:
                ui_text = "⚠️ Fake Job Posting" if int(label) == 1 else "✅ Real Job Posting"
                stored_label = "Fake" if int(label) == 1 else "Real"

            prediction = ui_text

            # ---- save log and get its ID ----
            user_id = session.get("user_id", "anonymous")
            log_id = insert_log(
                user_id=user_id,
                text_excerpt=input_text,
                prediction_label=stored_label,
                confidence=(float(confidence) if confidence is not None else None),
                was_needs_review=(1 if needs_review else 0),
            )

            # ---- redirect to info page ONLY when Needs Review ----
            if needs_review:
                return redirect(url_for("review_info", log_id=log_id))

    # normal render for GET or non-review predictions
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        input_text=input_text,
        error_message=error_message,
        per_model=per_model or {},
        final_proba=final_proba,
        ensemble_threshold=DEFAULT_THRESHOLD,
        review_band=REVIEW_BAND,
        extracted=extracted,
    )

@app.route("/info", methods=["GET", "POST"])
def info():
    return render_template(
        "info.html"
    )

@app.route("/playground", methods=["GET", "POST"])
def playground():
    prediction = None
    confidence = None
    input_text = None
    error_message = None
    extracted = None
    per_model = {}
    final_proba = None  # ensemble P(fake)
    if request.method == "POST":
        raw_text = request.form.get("job_posting", "").strip()
        input_text = raw_text
        extracted = extract_job_features(raw_text)

        if not raw_text:
            error_message = "Please paste a job posting first."
        else:
            # same preprocessing you use in /index
            title, company_profile, description, requirements, benefits = parse_job_posting(raw_text)

            # ensemble prediction (same helper used by /index)
            label, confidence, per_model, final_proba = predict_with_ensemble(
                title, company_profile, description, requirements, benefits
            )

            # Determine “Needs Review” vs Fake/Real
            needs_review = False
            if final_proba is not None:
                # final_proba = P(fake)
                needs_review = abs(final_proba - DEFAULT_THRESHOLD) <= REVIEW_BAND

            if needs_review:
                prediction = "🟡 Needs Review"
            else:
                prediction = "⚠️ Fake Job Posting" if int(label) == 1 else "✅ Real Job Posting"

            # IMPORTANT: do NOT call insert_log() here – this is anonymous playground

    return render_template(
        "playground.html",
        prediction=prediction,
        confidence=confidence,
        input_text=input_text,
        error_message=error_message,
        per_model=per_model or {},
        final_proba=final_proba,
        ensemble_threshold=DEFAULT_THRESHOLD,
        review_band=REVIEW_BAND,
        extracted=extracted,
        examples=PLAYGROUND_EXAMPLES,
    )

def extract_title_and_desc(excerpt: str):
    txt = (excerpt or "").strip()
    if not txt:
        return "(empty)", ""

    lines = txt.splitlines()

    # First line is the title
    title = lines[0].strip()

    # Everything else is description (joined back with newlines)
    if len(lines) > 1:
        desc = "\n".join(lines[1:]).strip()
    else:
        desc = ""

    return title, desc

@app.post("/logs/<int:log_id>/delete")
@login_required
def delete_log_route(log_id: int):
    user_id = session.get("user_id", "anonymous")
    deleted = delete_log(user_id=user_id, log_id=log_id)
    if deleted:
        flash("Log deleted.", "success")
    else:
        flash("Could not delete log (not found or not yours).", "warning")
    return redirect(url_for("logs"))

@app.route("/logs")
@login_required
def logs():
    user_id = session.get("user_id", "anonymous")
    rows = fetch_logs(user_id=user_id, limit=200)

    items = []
    for r in rows:
        title, desc = extract_title_and_desc(r["text_excerpt"])
        label = (r["prediction_label"] or "").strip().title()
        items.append({
            "id": r["id"],
            "title": title,
            "desc": desc,                 # not used in template right now, but ok
            "raw": r["text_excerpt"],     # full text
            "label": label,
            "confidence": r["confidence"],
            "created_at": r["created_at"],
            "was_needs_review": r["was_needs_review"] if "was_needs_review" in r.keys() else 0,
        })

    return render_template("logs.html", items=items)

@app.template_filter("format_datetime")
def format_datetime(value):
    """Convert ISO timestamp to 'MM/DD/YYYY HH:MM' 24-hour format."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%m/%d/%Y %H:%M")
    except Exception:
        return value

@app.route("/anti-scam-tips")
@login_required
def anti_scam_tips():
    return render_template("anti_scam_tips.html")

@app.context_processor
def inject_user_permission():
    """Expose user_permission everywhere (0=user, 1=admin)."""
    return dict(user_permission=int(session.get("permission_level", 0)))

@app.route("/admin_dashboard")
@login_required
def admin_dashboard():
    if int(session.get("permission_level", 0)) < 1:
        return redirect(url_for("index"))
    stats = get_admin_stats()
    users = []
    try:
        if supabase:
            response = supabase.table("login").select("id, Name, Email, Permission_level").execute()
            users = response.data or []
    except Exception as e:
        print(f"Error loading admin dashboard: {e}")

    stats = get_admin_stats() 

    return render_template("admin_dashboard.html",users=users,  stats=stats)

@app.post("/admin/review/<int:log_id>/set-real")
@login_required
def admin_mark_real(log_id):
    if int(session.get("permission_level", 0)) < 1:
        return redirect(url_for("index"))
    mark_review_decision(log_id, "Real")
    flash("Marked as Real.", "success")
    return redirect(url_for("admin_review"))


@app.post("/admin/review/<int:log_id>/set-fake")
@login_required
def admin_mark_fake(log_id):
    if int(session.get("permission_level", 0)) < 1:
        return redirect(url_for("index"))
    mark_review_decision(log_id, "Fake")
    flash("Marked as Fake.", "success")
    return redirect(url_for("admin_review"))

@app.route("/review-info/<int:log_id>", methods=["GET", "POST"])
@login_required
def review_info(log_id: int):
    # load the original prediction_log row so we can show title/text
    from data.db import get_db
    with get_db() as conn:
        cur = conn.execute(
            """
            SELECT id, user_id, text_excerpt, prediction_label, confidence, created_at
            FROM prediction_logs
            WHERE id = ?
            """,
            (log_id,),
        )
        row = cur.fetchone()

    if not row:
        flash("Review item not found.", "warning")
        return redirect(url_for("index"))

    if request.method == "POST":
        user_id = session.get("user_id", "anonymous")

        source = request.form.get("source")
        money_asked = request.form.get("money_asked")
        money_details = request.form.get("money_details")
        personal_info_asked = request.form.get("personal_info_asked")
        personal_info_details = request.form.get("personal_info_details")
        extra_notes = request.form.get("extra_notes")

        save_review_answers(
            log_id=log_id,
            user_id=user_id,
            source=source,
            money_asked=money_asked,
            money_details=money_details,
            personal_info_asked=personal_info_asked,
            personal_info_details=personal_info_details,
            extra_notes=extra_notes,
        )

        flash("Thanks! Your answers were sent to the review team.", "success")
        # go back to full scanner (and keep the session as-is)
        return redirect(url_for("index"))

    # GET – render the page
    title_line = row["text_excerpt"].split("\n", 1)[0]
    return render_template(
        "info.html",
        log_id=row["id"],
        title=title_line,
        raw_text=row["text_excerpt"],
        decision=row["prediction_label"],
        confidence=row["confidence"],
        created_at=row["created_at"],
    )

@app.route("/admin/review")
@login_required
def admin_review():
    if int(session.get("permission_level", 0)) < 1:
        return redirect(url_for("index"))

    from data.db import get_db
    with get_db() as conn:
        cur = conn.execute(
            """
            SELECT id, user_id, text_excerpt, prediction_label,
                   confidence, created_at
            FROM prediction_logs
            WHERE prediction_label = 'Needs Review'
            ORDER BY id DESC
            """
        )
        rows = cur.fetchall()

    items = []
    log_ids = []
    for r in rows:
        txt = r["text_excerpt"] or ""
        title = txt.split("\n", 1)[0].strip()
        items.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "title": title,
            "raw": txt,
            "confidence": r["confidence"],
            "created_at": r["created_at"],
        })
        log_ids.append(r["id"])

    answers_by_log = fetch_review_answers_for_logs(log_ids)

    return render_template(
        "admin_review.html",
        items=items,
        answers_by_log=answers_by_log,
    )

# ========================= Health =========================
@app.route("/health")
def health():
    return {"status": "ok"}

# ========================= Init =========================

@app.route("/")
def landing():
    return render_template("home.html")

init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=9620,debug=False)