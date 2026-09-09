# data/text_extractor.py
"""
Text Extraction Module for ScamGuard
-------------------------------------
This module extracts structured features from raw job descriptions.

Features extracted:
 - position_title
 - remote (Yes/No/Unknown)
 - employment_type
 - salary
 - experience_level
 - benefits
 - requires_personal_info

All functions are rule-based, explainable, and upgrade-friendly.
"""

import re

def extract_position(text: str) -> str:
    if not text.strip():
        return "Unknown"

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return "Unknown"

    first = lines[0]

    for sep in ["–", "-", "|", ":"]:
        if sep in first:
            first = first.split(sep)[0].strip()
            break

    job_words = ["assistant", "specialist", "engineer", "developer",
                 "manager", "entry", "clerk", "data entry", "analyst",
                 "representative", "remote"]
    if any(w in first.lower() for w in job_words):
        return first.title()

    m = re.search(r"(data entry|assistant|customer service|analyst|manager|developer)", 
                  text, re.IGNORECASE)
    if m:
        return m.group(1).title()

    return first.title()

def extract_remote_flag(text: str) -> str:
    t = text.lower()
    remote_keywords = [
        "remote", "work from home", "work-from-home",
        "home based", "online only", "fully remote"
    ]
    onsite_keywords = ["on-site", "onsite", "office-based", "in office"]

    if any(k in t for k in remote_keywords):
        return "Yes"

    if any(k in t for k in onsite_keywords):
        return "No"

    return "Unknown"

SALARY_PATTERN = re.compile(
    r"([$€£]\s?\d+(?:[,\d]{3})*(?:\.\d+)?\s*(?:k|per\s+\w+|/\w+|a\s+\w+|daily|hour|month|year)?)",
    re.IGNORECASE
)

def extract_salary(text: str) -> str:
    m = SALARY_PATTERN.search(text)
    return m.group(1).strip() if m else "Unknown"

def extract_employment_type(text: str) -> str:
    t = text.lower()
    if "full-time" in t or "full time" in t:
        return "Full-time"
    if "part-time" in t or "part time" in t:
        return "Part-time"
    if "contract" in t or "freelance" in t:
        return "Contract"
    if "intern" in t or "internship" in t:
        return "Internship"
    return "Unknown"

def extract_experience(text: str) -> str:
    t = text.lower()

    if "no experience" in t or "no prior experience" in t:
        return "No experience required"

    if "entry level" in t or "entry-level" in t or "junior" in t:
        return "Entry / Junior"

    if re.search(r"\b[3-9]\+?\s+years?\s+of\s+experience\b", t):
        return "Mid / Senior"

    return "Unknown"

def extract_benefits(text: str) -> str:
    t = text.lower()
    benefit_keywords = [
        "benefits include", "we offer", "we provide", "perks",
        "health insurance", "paid time off", "401k", "bonus",
        "medical", "dental", "vision"
    ]

    # Return only the first matching sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for s in sentences:
        if any(k in s.lower() for k in benefit_keywords):
            return s.strip()

    return "Unknown"

def extract_requires_personal_info(text: str) -> str:
    t = text.lower()
    sensitive_terms = [
        "id card", "passport", "social security", "ssn", "bank account",
        "account number", "credit card", "debit card", "date of birth",
        "drivers license", "driver's license"
    ]
    return "Yes" if any(k in t for k in sensitive_terms) else "No"


def extract_job_features(raw_text: str) -> dict:
    text = raw_text or ""

    return {
        "position_title": extract_position(text),
        "remote": extract_remote_flag(text),
        "employment_type": extract_employment_type(text),
        "salary": extract_salary(text),
        "experience_level": extract_experience(text),
        "benefits": extract_benefits(text),
        "requires_personal_info": extract_requires_personal_info(text),
    }