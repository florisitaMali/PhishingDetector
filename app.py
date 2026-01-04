import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import easyocr  # Deep learning OCR

# -----------------------------
# CACHED LOADING FUNCTIONS
# -----------------------------
@st.cache_resource
def load_models_and_ocr():
    # Load email models
    email_dt_model = joblib.load("models/email_dt.pkl")
    email_rf_model = joblib.load("models/email_rf.pkl")
    email_vec = joblib.load("models/email_vectorizer.pkl")
    
    # Load website models
    website_dt_model = joblib.load("models/website_dt.pkl")
    website_rf_model = joblib.load("models/website_rf.pkl")
    website_feats = joblib.load("models/website_features.pkl")
    
    # Initialize OCR reader (slow, cached)
    ocr_reader = easyocr.Reader(['en'], gpu=True)  # Change gpu=True if you have CUDA GPU
    
    return email_dt_model, email_rf_model, email_vec, website_dt_model, website_rf_model, website_feats, ocr_reader

@st.cache_data
def cached_extract_website_features(url):
    from agent.feature_extraction import extract_website_features
    return extract_website_features(url)

# Load all models and OCR once
email_dt, email_rf, email_vectorizer, website_dt, website_rf, website_features, reader = load_models_and_ocr()

# Import agent decision logic
from agent.decision_logic import agent_decision

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="Multi-Modal Phishing Detection Agent",
    layout="centered"
)

st.title("Multi-Modal Phishing Detection Agent")
st.markdown("""
This intelligent agent detects phishing in **emails**, **websites**, and **email/website screenshots**.
- **ALLOW ✅** – low-risk
- **WARN ⚠️** – medium-risk
- **BLOCK 🚫** – high-risk
""")

# -----------------------------
# TABS
# -----------------------------
tab_email, tab_website, tab_image = st.tabs(
    ["Email Phishing", "Website Phishing", "Image/Screenshot Phishing"]
)

# =============================
# EMAIL PHISHING TAB
# =============================
with tab_email:
    st.subheader("Email Phishing Detection")
    email_text = st.text_area("Enter email content:")
    model_choice = st.selectbox("Select Model:", ["Decision Tree", "Random Forest"])

    if st.button("Analyze Email", key="analyze_email"):
        if email_text.strip():
            with st.spinner("Vectorizing email text..."):
                vectorized_email = email_vectorizer.transform([email_text])

            with st.spinner("Running phishing detection model..."):
                model = email_dt if model_choice == "Decision Tree" else email_rf
                phishing_prob = model.predict_proba(vectorized_email)[0][1]
                action, risk = agent_decision(phishing_prob)

            st.metric("Phishing Probability", f"{phishing_prob:.2f}")
            st.write("**Risk Level:**", risk)
            st.write("**Agent Action:**", action)
        else:
            st.warning("Please enter the email content.")


# =============================
# WEBSITE PHISHING TAB
# =============================
with tab_website:
    st.subheader("Website Phishing Detection")
    url = st.text_input("Enter website URL (e.g., https://example.com):")
    model_choice = st.selectbox("Select Model:", ["Decision Tree", "Random Forest"], key="website_model")

    if st.button("Analyze Website", key="analyze_website"):
        if url.strip():
            with st.spinner("Extracting website features..."):
                feature_dict = cached_extract_website_features(url)
                feature_vector = [feature_dict.get(f, 0) for f in website_features]

            with st.spinner("Running phishing detection model..."):
                model = website_dt if model_choice == "Decision Tree" else website_rf
                phishing_prob = model.predict_proba([feature_vector])[0][1]
                action, risk = agent_decision(phishing_prob)

            st.metric("Phishing Probability", f"{phishing_prob:.2f}")
            st.write("**Risk Level:**", risk)
            st.write("**Agent Action:**", action)

            if st.checkbox("Show Extracted Features"):
                st.subheader("Extracted Website Features")
                st.json(feature_dict)
        else:
            st.warning("Please enter a website URL.")


# =============================
# IMAGE/Screenshot PHISHING TAB
# =============================
with tab_image:
    st.subheader("Image / Screenshot Phishing Detection")
    uploaded_file = st.file_uploader("Upload screenshot of email or website", type=["png", "jpg", "jpeg"])
    model_choice = st.selectbox("Select Model for extracted text:", ["Decision Tree", "Random Forest"], key="image_model")

    if st.button("Analyze Image", key="analyze_image"):
        if uploaded_file is not None:
            with st.spinner("Loading and preparing image..."):
                img = Image.open(uploaded_file).convert("RGB")
                max_dim = 1024
                if max(img.size) > max_dim:
                    scale = max_dim / max(img.size)
                    img = img.resize((int(img.width * scale), int(img.height * scale)))

            with st.spinner("Running OCR (extracting text from image)..."):
                result = reader.readtext(np.array(img))
                extracted_text = " ".join([r[1] for r in result])

            st.subheader("Extracted Text from Image")
            st.text(extracted_text if extracted_text.strip() else "No text found in image.")

            if extracted_text.strip():
                with st.spinner("Vectorizing extracted text..."):
                    vectorized_email = email_vectorizer.transform([extracted_text])

                with st.spinner("Running phishing detection model..."):
                    model = email_dt if model_choice == "Decision Tree" else email_rf
                    phishing_prob = model.predict_proba(vectorized_email)[0][1]
                    action, risk = agent_decision(phishing_prob)

                st.metric("Phishing Probability", f"{phishing_prob:.2f}")
                st.write("**Risk Level:**", risk)
                st.write("**Agent Action:**", action)
            else:
                st.warning("No text could be extracted from the image.")
        else:
            st.warning("Please upload a screenshot.")
