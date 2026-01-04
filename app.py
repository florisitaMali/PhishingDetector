import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load your models and resources
# -----------------------------
email_dt = joblib.load("models/email_dt.pkl")
email_rf = joblib.load("models/email_rf.pkl")
email_vectorizer = joblib.load("models/email_vectorizer.pkl")

website_dt = joblib.load("models/website_dt.pkl")
website_rf = joblib.load("models/website_rf.pkl")
website_features = joblib.load("models/website_features.pkl")

# Import agent functions
from agent.decision_logic import agent_decision
from agent.feature_extraction import extract_website_features

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Multi-Modal Phishing Detection Agent",
    layout="centered"
)

st.title("🚨 Multi-Modal Phishing Detection Agent")
st.markdown("""
This intelligent agent detects phishing in **emails** and **websites** and provides a risk-based action:
- **ALLOW ✅** – low-risk
- **WARN ⚠️** – medium-risk
- **BLOCK 🚫** – high-risk
""")

# -----------------------------
# TABS
# -----------------------------
tab_email, tab_website = st.tabs(["📧 Email Phishing", "🌐 Website Phishing"])

# =============================
# EMAIL PHISHING TAB
# =============================
with tab_email:
    st.subheader("📧 Email Phishing Detection")
    email_text = st.text_area("Enter email content:")
    model_choice = st.selectbox("Select Model:", ["Decision Tree", "Random Forest"])

    if st.button("Analyze Email", key="analyze_email"):
        if email_text.strip() != "":
            # Vectorize email
            vectorized_email = email_vectorizer.transform([email_text])

            # Choose model
            model = email_dt if model_choice == "Decision Tree" else email_rf

            # Predict phishing probability
            phishing_prob = model.predict_proba(vectorized_email)[0][1]

            # Get agent decision
            action, risk = agent_decision(phishing_prob)

            # Display results
            st.metric("Phishing Probability", f"{phishing_prob:.2f}")
            st.write("**Risk Level:**", risk)
            st.write("**Agent Action:**", action)
        else:
            st.warning("Please enter the email content.")

# =============================
# WEBSITE PHISHING TAB
# =============================
with tab_website:
    st.subheader("🌐 Website Phishing Detection")
    url = st.text_input("Enter website URL (e.g., https://example.com):")
    model_choice = st.selectbox("Select Model:", ["Decision Tree", "Random Forest"], key="website_model")

    if st.button("Analyze Website", key="analyze_website"):
        if url.strip() != "":
            # Extract features
            feature_dict = extract_website_features(url)
            feature_vector = [feature_dict.get(f, 0) for f in website_features]

            # Choose model
            model = website_dt if model_choice == "Decision Tree" else website_rf

            # Predict phishing probability
            phishing_prob = model.predict_proba([feature_vector])[0][1]

            # Get agent decision
            action, risk = agent_decision(phishing_prob)

            # Display results
            st.metric("Phishing Probability", f"{phishing_prob:.2f}")
            st.write("**Risk Level:**", risk)
            st.write("**Agent Action:**", action)

            # Optional: show extracted features
            if st.checkbox("Show Extracted Features"):
                st.subheader("🔍 Extracted Website Features")
                st.json(feature_dict)
        else:
            st.warning("Please enter a website URL.")
