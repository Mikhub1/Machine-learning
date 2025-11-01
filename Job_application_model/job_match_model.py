# job_match_model.py
# ------------------------------------------------------------
# What this model does
# ------------------------------------------------------------
# 1) Zero-shot match score: Uses TF-IDF + cosine similarity to score how well a resume
#    aligns with a job description (no training data needed).
# 2) Skill gap analysis: Extracts skills from both texts and lists skills missing
#    from the resume that appear in the job posting.
# 3) Supervised classifier (optional): If you have labeled data (resume, job_desc, label),
#    it trains a Logistic Regression model on a small feature set:
#    - cosine similarity
#    - Jaccard similarity on extracted skills
#    - basic numeric cues (counts of tools/skills, years)
#
# How to use quickly (no training data required):
# >>> from job_match_model import JobMatcher
# >>> jm = JobMatcher()
# >>> score, analysis = jm.score_pair(resume_text, job_text)
# >>> print(score)         # 0..1 cosine similarity
# >>> print(analysis)      # dict with extracted skills and missing skills
#
# With labeled data (CSV with columns: resume, job_desc, label in {0,1}):
# >>> jm = JobMatcher()
# >>> jm.fit_supervised("training_data.csv")
# >>> prob = jm.predict_proba(resume_text, job_text)   # probability of being a good match
# ------------------------------------------------------------

import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# ---------------------------------------
# A lightweight skills list (extend as needed)
# ---------------------------------------
DEFAULT_SKILLS = [
    # General / office
    "communication", "teamwork", "leadership", "presentation", "excel", "word", "powerpoint",
    # Programming / data
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "r", "matlab", "bash",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "spark",
    # Web / cloud / devops
    "html", "css", "react", "next.js", "node.js", "django", "flask", "docker", "kubernetes",
    "aws", "azure", "gcp", "git", "github", "gitlab", "ci/cd",
    # Data / analytics
    "tableau", "power bi", "airflow", "dbt",
    # Security
    "linux", "networking", "siem", "splunk", "wireshark",
    # Project / agile
    "jira", "confluence", "scrum", "agile",
]

# ---------------------------------------
# Text cleaning & utility functions
# ---------------------------------------
def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\\b(\\d{1,2})\\s*\\+\\s*years?\\b", r"\\1 years", t)
    t = re.sub(r"[^a-z0-9\\.\\-\\+\\# ]+", " ", t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t

def extract_years_of_experience(t: str) -> float:
    # crude: take the max of N years patterns
    years = [int(x) for x in re.findall(r"\\b(\\d{1,2})\\s*years?\\b", t.lower())]
    return float(max(years)) if years else 0.0

def extract_skills(t: str, skills: List[str]) -> List[str]:
    txt = " " + clean_text(t) + " "
    found = []
    for sk in skills:
        pat = r"\\b" + re.escape(sk.lower()) + r"\\b"
        if re.search(pat, txt):
            found.append(sk.lower())
    return sorted(list(set(found)))

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)

# ---------------------------------------
# Feature builder for supervised model
# ---------------------------------------
class PairFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, skills: Optional[List[str]] = None):
        self.skills = skills or DEFAULT_SKILLS
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)

    def fit(self, X: List[Tuple[str, str]], y=None):
        texts = [clean_text(r) + " " + clean_text(j) for (r, j) in X]
        self.vectorizer.fit(texts)
        return self

    def transform(self, X: List[Tuple[str, str]]):
        feats = []
        for resume, job in X:
            r = clean_text(resume)
            j = clean_text(job)

            # TF-IDF cosine
            v = self.vectorizer.transform([r, j])
            cos = float(cosine_similarity(v[0], v[1])[0,0])

            # Skills
            r_sk = extract_skills(r, self.skills)
            j_sk = extract_skills(j, self.skills)
            jac = jaccard(r_sk, j_sk)
            r_cnt, j_cnt = len(r_sk), len(j_sk)

            # Years (very rough)
            r_yrs = extract_years_of_experience(resume)
            j_yrs = extract_years_of_experience(job)

            feats.append([cos, jac, r_cnt, j_cnt, r_yrs, j_yrs])
        return np.array(feats, dtype=float)

# ---------------------------------------
# Main API
# ---------------------------------------
@dataclass
class JobMatchAnalysis:
    resume_skills: List[str]
    job_skills: List[str]
    missing_skills: List[str]
    years_in_resume: float
    years_in_job: float

class JobMatcher:
    def __init__(self, skills: Optional[List[str]] = None):
        self.skills = skills or DEFAULT_SKILLS
        self._tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self._clf: Optional[Pipeline] = None  # set after supervised training

    # ---------- Zero-shot scoring ----------
    def score_pair(self, resume_text: str, job_text: str) -> Tuple[float, Dict]:
        r = clean_text(resume_text)
        j = clean_text(job_text)
        tfidf = self._tfidf.fit_transform([r, j])
        score = float(cosine_similarity(tfidf[0], tfidf[1])[0, 0])

        r_sk = extract_skills(r, self.skills)
        j_sk = extract_skills(j, self.skills)
        missing = sorted(list(set(j_sk) - set(r_sk)))

        analysis = JobMatchAnalysis(
            resume_skills=r_sk,
            job_skills=j_sk,
            missing_skills=missing,
            years_in_resume=extract_years_of_experience(resume_text),
            years_in_job=extract_years_of_experience(job_text),
        )
        return score, analysis.__dict__

    # ---------- Supervised training (optional) ----------
    def fit_supervised(self, csv_path: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        \"\"\"Expects CSV with columns: resume, job_desc, label (0/1).\"\"\"
        df = pd.read_csv(csv_path)
        for col in [\"resume\", \"job_desc\", \"label\"]:
            if col not in df.columns:
                raise ValueError(f\"Missing column '{col}' in {csv_path}\")

        X = list(zip(df[\"resume\"].astype(str).tolist(), df[\"job_desc\"].astype(str).tolist()))
        y = df[\"label\"].astype(int).values

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        pipe = Pipeline([
            (\"feats\", PairFeaturizer(self.skills)),
            (\"clf\", LogisticRegression(max_iter=200))
        ])
        pipe.fit(X_tr, y_tr)
        self._clf = pipe

        probs = pipe.predict_proba(X_te)[:,1]
        preds = (probs >= 0.5).astype(int)
        metrics = {
            \"auc\": float(roc_auc_score(y_te, probs)),
            \"f1\": float(f1_score(y_te, preds)),
            \"accuracy\": float(accuracy_score(y_te, preds)),
        }
        return metrics

    def predict_proba(self, resume_text: str, job_text: str) -> float:
        if self._clf is None:
            raise RuntimeError(\"Model not trained. Call fit_supervised(...) first.\")
        return float(self._clf.predict_proba([(resume_text, job_text)])[:,1][0])

# ---------------------------------------
# Quick demo
# ---------------------------------------
if __name__ == \"__main__\":
    resume = \"\"\"
    Computer Engineering student with 2 years internship experience.
    Strong in Python, SQL, Pandas, scikit-learn, Git, Docker. Built dashboards in Tableau.
    Worked with AWS and basic Kubernetes. Led a small Scrum team.
    \"\"\"
    job = \"\"\"
    We seek a Data Analyst with 3 years experience. Required: Python, SQL, Tableau, Power BI,
    Pandas, Git. Nice-to-have: Docker, AWS, Kubernetes. Excellent communication skills.
    \"\"\"

    jm = JobMatcher()
    score, info = jm.score_pair(resume, job)
    print(\"Zero-shot cosine score:\", round(score, 3))
    print(\"Analysis:\", json.dumps(info, indent=2))

    # If you have labeled data:
    # metrics = jm.fit_supervised(\"training_data.csv\")
    # print(\"Holdout metrics:\", metrics)
    # prob = jm.predict_proba(resume, job)
    # print(\"Supervised match probability:\", round(prob, 3))
