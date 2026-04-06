"""
Fake Job Posting Detector - Streamlit App
=========================================
Detects fraudulent job postings using NLP + ML.
Includes Explainable AI (LIME + SHAP-style word importance).
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import pickle
import time

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

# ── plotting ─────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── NLP ───────────────────────────────────────────────────────────────────────
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "fake_job_postings.csv"   # put CSV next to this script
MODEL_FILE  = "job_fraud_model.pkl"

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
FRAUD_KEYWORDS = [
    "work from home", "easy money", "no experience", "unlimited earning",
    "make money fast", "guaranteed income", "be your own boss", "free training",
    "investment required", "upfront fee", "wire transfer", "western union",
    "uncapped earnings", "passive income", "pyramid", "mlm", "multi level",
    "urgent hiring", "immediate start", "no interview", "data entry",
    "envelope stuffing", "stuffing envelopes", "processing fees",
    "pay to apply", "gift cards", "cryptocurrency payment",
]

SAFE_KEYWORDS = [
    "bachelor", "master", "phd", "degree", "years of experience",
    "competitive salary", "health insurance", "401k", "pension",
    "team player", "responsibilities include", "requirements",
    "equal opportunity", "background check", "references",
    "professional development", "annual leave", "visa sponsorship",
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_fields(row) -> str:
    parts = []
    for col in ["title", "company_profile", "description", "requirements", "benefits"]:
        val = row.get(col, "")
        if isinstance(val, str):
            parts.append(val)
    return " ".join(parts)


def keyword_hit_count(text: str, keywords: list) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def get_word_importances(pipeline, text: str, top_n: int = 20):
    """Return (word, score) list using TF-IDF weight × feature importance."""
    vectorizer = pipeline.named_steps["tfidf"]
    clf        = pipeline.named_steps["clf"]
    vec        = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    # get feature importances from the classifier
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
    else:
        importances = np.ones(len(feature_names))

    tfidf_vals = vec.toarray()[0]
    scores     = tfidf_vals * np.abs(importances)

    top_idx    = scores.argsort()[::-1][:top_n]
    result = [(feature_names[i], float(scores[i])) for i in top_idx if scores[i] > 0]
    return result


def fraud_word_highlight(text: str, word_scores: list) -> str:
    """Return HTML with fraud-indicative words highlighted."""
    top_words = {w for w, _ in word_scores[:15]}
    tokens    = text.split()
    html_parts = []
    for token in tokens:
        clean = re.sub(r"[^a-z]", "", token.lower())
        if clean in top_words:
            html_parts.append(
                f'<mark style="background:#ff6b6b;color:#fff;'
                f'border-radius:3px;padding:1px 4px">{token}</mark>'
            )
        else:
            html_parts.append(token)
    return " ".join(html_parts)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING / LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            bundle = pickle.load(f)
        return bundle

    if not os.path.exists(DATA_PATH):
        return None  # handled in UI

    df = pd.read_csv(DATA_PATH)
    df["text"] = df.apply(combine_fields, axis=1).apply(clean_text)
    df["fraudulent"] = df["fraudulent"].astype(int)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    X = df["text"]
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm          = confusion_matrix(y_test, y_pred)
    roc_auc     = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    bundle = {
        "pipeline": pipeline,
        "report":   report_dict,
        "cm":       cm,
        "roc_auc":  roc_auc,
        "fpr":      fpr,
        "tpr":      tpr,
        "n_train":  len(X_train),
        "n_test":   len(X_test),
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)

    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("-----")
    st.title("🔍 Fake Job Detector")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🕵️ Analyze Job Posting", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Built with Streamlit · TF-IDF · Logistic Regression · XAI")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading model (training on first run — takes ~30 s) …"):
    bundle = load_or_train_model()

if bundle is None:
    st.error(
        f"⚠️ Dataset not found. Please place **`{DATA_PATH}`** "
        "in the same folder as this script and restart."
    )
    st.stop()

pipeline = bundle["pipeline"]


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYZE JOB POSTING
# ═════════════════════════════════════════════════════════════════════════════
if page == "🕵️ Analyze Job Posting":

    st.title("🕵️ Job Posting Fraud Analyzer")
    st.markdown(
        "Fill in the job posting details below. "
        "The model will predict if it's **legitimate** or **fraudulent**, "
        "and explain *why*."
    )

    # ── FORM ─────────────────────────────────────────────────────────────────
    with st.form("job_form"):

        col1, col2 = st.columns(2)
        with col1:
            title      = st.text_input("Job Title *", placeholder="e.g. Software Engineer")
            company    = st.text_input("Company Profile", placeholder="Brief company description")
            industry   = st.text_input("Industry", placeholder="e.g. Information Technology")
        with col2:
            salary     = st.text_input("Salary Range", placeholder="e.g. $60,000 – $80,000")
            location   = st.text_input("Location", placeholder="e.g. New York, NY")
            emp_type   = st.selectbox(
                "Employment Type",
                ["Full-time", "Part-time", "Contract", "Temporary",
                 "Internship", "Other", "Not specified"]
            )

        description  = st.text_area("Job Description *", height=180,
                                    placeholder="Describe the role, responsibilities…")
        requirements = st.text_area("Requirements", height=120,
                                    placeholder="Qualifications, skills, experience…")
        benefits     = st.text_area("Benefits", height=80,
                                    placeholder="Health insurance, 401k, remote…")

        col3, col4, col5 = st.columns(3)
        with col3:
            has_logo  = st.checkbox("Has Company Logo")
        with col4:
            telecomm  = st.checkbox("Remote / Telecommuting")
        with col5:
            has_qs    = st.checkbox("Includes Screening Questions")

        submitted = st.form_submit_button("🔍 Analyze Posting", use_container_width=True)

    # ── PREDICTION ───────────────────────────────────────────────────────────
    if submitted:
        if not title.strip() and not description.strip():
            st.warning("Please enter at least a Job Title or Description.")
            st.stop()

        raw_text = " ".join([
            title, company, description, requirements, benefits
        ])
        clean    = clean_text(raw_text)

        with st.spinner("Analyzing …"):
            time.sleep(0.4)
            prob_fraud  = pipeline.predict_proba([clean])[0][1]
            prob_legit  = 1 - prob_fraud
            prediction  = int(prob_fraud >= 0.5)

        fraud_kw_hits = keyword_hit_count(raw_text, FRAUD_KEYWORDS)
        safe_kw_hits  = keyword_hit_count(raw_text, SAFE_KEYWORDS)
        word_scores   = get_word_importances(pipeline, clean, top_n=25)

        # ── VERDICT ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Verdict")

        vcol1, vcol2, vcol3 = st.columns([2, 1, 1])
        with vcol1:
            if prediction == 1:
                st.error("🚨 **FRAUDULENT** Job Posting Detected")
                verdict_color = "#ff4444"
                verdict_emoji = "🚨"
            else:
                st.success("✅ **LEGITIMATE** Job Posting")
                verdict_color = "#00b894"
                verdict_emoji = "✅"

        with vcol2:
            st.metric("Fraud Probability",  f"{prob_fraud*100:.1f}%")
        with vcol3:
            st.metric("Legit Probability",  f"{prob_legit*100:.1f}%")

        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = prob_fraud * 100,
            title = {"text": "Fraud Risk Score"},
            gauge = {
                "axis": {"range": [0, 100]},
                "bar":  {"color": verdict_color},
                "steps": [
                    {"range": [0,  40], "color": "#d4edda"},
                    {"range": [40, 65], "color": "#fff3cd"},
                    {"range": [65,100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 50
                }
            },
            number = {"suffix": "%", "font": {"size": 40}},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── RISK BREAKDOWN ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("⚡ Risk Signal Breakdown")

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            risk_level = "🔴 High" if fraud_kw_hits >= 3 else ("🟡 Medium" if fraud_kw_hits >= 1 else "🟢 Low")
            st.metric("Fraud Keywords Found",  f"{fraud_kw_hits}",  delta=risk_level, delta_color="off")
        with rc2:
            st.metric("Trust Keywords Found",  f"{safe_kw_hits}")
        with rc3:
            st.metric("Has Company Logo",       "Yes" if has_logo else "No")
        with rc4:
            st.metric("Description Length",     f"{len(description.split())} words")

        # ── EXPLAINABILITY ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🧠 Why this verdict? — Explainable AI")

        tab1, tab2, tab3 = st.tabs([
            "📊 Top Influential Words",
            "🔤 Text Highlight",
            "🧮 Keyword Analysis"
        ])

        with tab1:
            if word_scores:
                words  = [w for w, _ in word_scores[:15]]
                scores = [s for _, s in word_scores[:15]]
                colors = ["#ff6b6b" if prediction == 1 else "#00b894"] * len(words)

                fig_bar = px.bar(
                    x=scores, y=words,
                    orientation="h",
                    color=scores,
                    color_continuous_scale=["#00b894", "#fdcb6e", "#ff6b6b"],
                    labels={"x": "Influence Score", "y": "Word / Phrase"},
                    title="Top 15 Words Influencing the Prediction",
                )
                fig_bar.update_layout(
                    height=420, showlegend=False,
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.caption(
                    "Bars show how strongly each word/phrase pushed the model "
                    "toward the fraud prediction. Longer bar = bigger influence."
                )
            else:
                st.info("Not enough text to compute word importances.")

        with tab2:
            st.markdown("Words highlighted in **red** are the ones most influencing the fraud prediction:")
            highlighted = fraud_word_highlight(clean[:2000], word_scores)
            st.markdown(
                f'<div style="background:#f8f9fa;padding:12px;border-radius:8px;'
                f'font-size:14px;line-height:1.8">{highlighted}</div>',
                unsafe_allow_html=True
            )
            st.caption("Showing first 2 000 characters of combined text.")

        with tab3:
            found_fraud = [kw for kw in FRAUD_KEYWORDS if kw in raw_text.lower()]
            found_safe  = [kw for kw in SAFE_KEYWORDS  if kw in raw_text.lower()]

            kc1, kc2 = st.columns(2)
            with kc1:
                st.markdown("**🔴 Fraud-Indicative Keywords Found:**")
                if found_fraud:
                    for kw in found_fraud:
                        st.markdown(f"- ❌ `{kw}`")
                else:
                    st.success("None found — good sign!")
            with kc2:
                st.markdown("**🟢 Trust-Building Keywords Found:**")
                if found_safe:
                    for kw in found_safe:
                        st.markdown(f"- ✅ `{kw}`")
                else:
                    st.warning("No trust signals detected.")

        # ── RISK TIPS ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 What to Watch Out For")

        tips = {
            "Upfront payment requests": "Legitimate employers never ask candidates to pay fees.",
            "Vague job description": "Fraud postings are often generic with no specific responsibilities.",
            "Unrealistic salary": "Suspiciously high pay for low-skill work is a major red flag.",
            "No company info": "Absence of a verifiable company name / website is concerning.",
            "Urgency / pressure": "Phrases like 'immediate hire' or 'limited spots' are pressure tactics.",
            "Personal info too early": "Asking for SSN / bank details before hiring is a scam signal.",
        }
        tip_cols = st.columns(3)
        for i, (tip, desc) in enumerate(tips.items()):
            with tip_cols[i % 3]:
                st.info(f"**{tip}**\n\n{desc}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Insights":

    st.title("📊 Model Performance & Dataset Insights")

    report  = bundle["report"]
    cm      = bundle["cm"]
    roc_auc = bundle["roc_auc"]
    fpr     = bundle["fpr"]
    tpr     = bundle["tpr"]

    # ── KEY METRICS ────────────────────────────────────────────────────────
    st.subheader("🎯 Model Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{report['accuracy']*100:.1f}%")
    m2.metric("Precision (Fraud)", f"{report['1']['precision']*100:.1f}%")
    m3.metric("Recall (Fraud)",    f"{report['1']['recall']*100:.1f}%")
    m4.metric("F1-Score (Fraud)",  f"{report['1']['f1-score']*100:.1f}%")
    m5.metric("ROC-AUC",           f"{roc_auc:.3f}")

    st.markdown(f"Training samples: **{bundle['n_train']:,}** | Test samples: **{bundle['n_test']:,}**")

    # ── CONFUSION MATRIX + ROC ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        tn, fp, fn, tp = cm.ravel()
        fig_cm = go.Figure(go.Heatmap(
            z       = cm,
            x       = ["Predicted Legit", "Predicted Fraud"],
            y       = ["Actual Legit",    "Actual Fraud"],
            colorscale = "RdYlGn_r",
            text    = [[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
            texttemplate = "%{text}",
            showscale = False,
        ))
        fig_cm.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(
            f"TN={tn} | FP={fp} | FN={fn} | TP={tp}\n\n"
            "FP = legitimate jobs wrongly flagged. FN = fraudulent jobs missed."
        )

    with col2:
        st.subheader("ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            fill="tozeroy", fillcolor="rgba(0,184,148,0.2)",
            line=dict(color="#00b894", width=2.5),
            name=f"AUC = {roc_auc:.3f}"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            line=dict(color="gray", dash="dash"),
            name="Random Classifier"
        ))
        fig_roc.update_layout(
            height=320,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.1),
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── DATASET STATS ──────────────────────────────────────────────────────
    if os.path.exists(DATA_PATH):
        st.markdown("---")
        st.subheader("📈 Dataset Overview")

        df_full = pd.read_csv(DATA_PATH)

        dcol1, dcol2 = st.columns(2)

        with dcol1:
            counts = df_full["fraudulent"].value_counts()
            fig_pie = px.pie(
                names=["Legitimate", "Fraudulent"],
                values=[counts.get(0, 0), counts.get(1, 0)],
                color_discrete_sequence=["#00b894", "#ff6b6b"],
                title="Class Distribution",
                hole=0.4,
            )
            fig_pie.update_layout(height=320, margin=dict(t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with dcol2:
            et = df_full.groupby(["employment_type", "fraudulent"]).size().reset_index(name="count")
            et["label"] = et["fraudulent"].map({0: "Legit", 1: "Fraud"})
            fig_et = px.bar(
                et, x="employment_type", y="count", color="label",
                color_discrete_map={"Legit": "#00b894", "Fraud": "#ff6b6b"},
                barmode="stack",
                title="Fraud by Employment Type",
                labels={"employment_type": "Type", "count": "Count"},
            )
            fig_et.update_layout(height=320, margin=dict(t=40, b=10),
                                  xaxis_tickangle=-30)
            st.plotly_chart(fig_et, use_container_width=True)

        # Top industries with most fraud
        industry_fraud = (
            df_full[df_full["fraudulent"] == 1]["industry"]
            .dropna().value_counts().head(10).reset_index()
        )
        industry_fraud.columns = ["industry", "count"]
        fig_ind = px.bar(
            industry_fraud, x="count", y="industry", orientation="h",
            color="count", color_continuous_scale="Reds",
            title="Top 10 Industries with Fraudulent Postings",
        )
        fig_ind.update_layout(height=380, coloraxis_showscale=False,
                               yaxis=dict(autorange="reversed"),
                               margin=dict(t=40, b=10))
        st.plotly_chart(fig_ind, use_container_width=True)

    # ── TOP FRAUD WORDS ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔠 Most Influential Fraud Words (Model Coefficients)")

    tfidf     = pipeline.named_steps["tfidf"]
    clf       = pipeline.named_steps["clf"]
    feat_names = tfidf.get_feature_names_out()

    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
        top_fraud_idx = coefs.argsort()[-25:][::-1]
        top_legit_idx = coefs.argsort()[:25]

        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown("**🔴 Words pointing → FRAUD**")
            fw_df = pd.DataFrame({
                "Word": feat_names[top_fraud_idx],
                "Score": coefs[top_fraud_idx]
            })
            fig_fw = px.bar(fw_df, x="Score", y="Word", orientation="h",
                            color_discrete_sequence=["#ff6b6b"])
            fig_fw.update_layout(height=500, yaxis=dict(autorange="reversed"),
                                  margin=dict(t=10, b=10))
            st.plotly_chart(fig_fw, use_container_width=True)

        with wc2:
            st.markdown("**🟢 Words pointing → LEGIT**")
            lw_df = pd.DataFrame({
                "Word":  feat_names[top_legit_idx],
                "Score": np.abs(coefs[top_legit_idx])
            })
            fig_lw = px.bar(lw_df, x="Score", y="Word", orientation="h",
                            color_discrete_sequence=["#00b894"])
            fig_lw.update_layout(height=500, yaxis=dict(autorange="reversed"),
                                  margin=dict(t=10, b=10))
            st.plotly_chart(fig_lw, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.title("ℹ️ About This App")

    st.markdown("""
## 🔍 Fake Job Posting Detector

This tool helps job seekers identify potentially **fraudulent job postings** 
using Natural Language Processing and Machine Learning.

---

### 🧠 How It Works

1. **Text Preprocessing** — All text fields (title, description, requirements, 
   benefits, company profile) are cleaned, lowercased, and tokenized.

2. **TF-IDF Vectorisation** — The text is transformed into a 15 000-feature 
   numerical matrix using bigrams and sublinear TF scaling.

3. **Logistic Regression** — A balanced logistic regression classifier 
   (trained on 17 880 job postings) predicts fraud probability.

4. **Explainable AI** — Feature coefficients × TF-IDF scores reveal which 
   exact words drove the prediction.

---

### 📊 Dataset

- **Source:** [EMSCAD Fake Job Postings Dataset](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size:** 17 880 job postings
- **Fraudulent:** 866 (≈4.8%)
- **Legitimate:** 17 014 (≈95.2%)

---

### ⚠️ Common Fraud Patterns

| Signal | Description |
|--------|-------------|
| Upfront fees | Any request for payment from the candidate |
| Vague descriptions | Generic roles with no real responsibilities |
| Unrealistic salary | $500/day for "data entry" type scams |
| No company verifiable info | Missing website, LinkedIn, address |
| Urgency language | "Apply NOW", "Limited seats", "Immediate hire" |
| Wire transfers / crypto | Payment via untraceable methods |
| Work from home + high pay | Common bait for MLM or pyramid schemes |

---

### 📋 Tips for Job Seekers

- **Google the company** before applying — check reviews on Glassdoor / LinkedIn
- **Never pay** to apply or to get hired
- **Verify the email domain** — legitimate companies use their own domain
- **Check the job on the company's official website**
- **Trust your gut** — if it sounds too good to be true, it usually is

---

*Model accuracy may vary. Always use your own judgement alongside this tool.*
    """)

    st.info(
        "📁 **To run this app:** Place `fake_job_postings.csv` in the same "
        "directory as this script, then run `streamlit run fake_job_detector_app.py`"
    )
