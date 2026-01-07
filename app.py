import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import easyocr
import base64

st.set_page_config(
    page_title="Multi-Modal Phishing Detection Agent",
    layout="centered"
)


if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if "last_image_name" not in st.session_state:
    st.session_state.last_image_name = None

@st.cache_resource
def load_models_and_ocr():
    email_dt_model = joblib.load("models/email_dt.pkl")
    email_rf_model = joblib.load("models/email_rf.pkl")
    email_vec = joblib.load("models/email_vectorizer.pkl")

    website_dt_model = joblib.load("models/website_dt.pkl")
    website_rf_model = joblib.load("models/website_rf.pkl")
    website_feats = joblib.load("models/website_features.pkl")

    ocr_reader = easyocr.Reader(['en'], gpu=True)

    return (
        email_dt_model,
        email_rf_model,
        email_vec,
        website_dt_model,
        website_rf_model,
        website_feats,
        ocr_reader,
    )

@st.cache_data
def cached_extract_website_features(url):
    from agent.feature_extraction import extract_website_features
    return extract_website_features(url)

(
    email_dt,
    email_rf,
    email_vectorizer,
    website_dt,
    website_rf,
    website_features,
    reader,
) = load_models_and_ocr()

from agent.decision_logic import agent_decision

def show_agent_visual(action):
    action = action.upper().strip()

    if "ALLOW" in action:
        gif_path = "assets/allow.gif"
        caption = "Low risk detected"
    elif "WARN" in action:
        gif_path = "assets/warn.gif"
        caption = "Suspicious activity detected"
    elif "BLOCK" in action:
        gif_path = "assets/block.gif"
        caption = "High-risk phishing detected"
    else:
        return

    with open(gif_path, "rb") as f:
        gif_base64 = base64.b64encode(f.read()).decode()

    html = f"""
    <div style="text-align:center">
        <img src="data:image/gif;base64,{gif_base64}" width="260">
        <p style="font-weight:bold; font-size:16px;">{caption}</p>
    </div>
    """
    components.html(html, height=320)

st.title("Multi-Modal Phishing Detection Agent")

st.markdown("""
This intelligent agent detects phishing in **emails**, **websites**, and **screenshots**.
- **ALLOW** – low-risk
- **WARN** – medium-risk
- **BLOCK** – high-risk
""")


tab_email, tab_website, tab_image = st.tabs(
    ["Email Phishing", "Website Phishing", "Image Phishing"]
)

with tab_email:
    st.subheader("Email Phishing Detection")

    email_text = st.text_area("Enter email content:")
    model_choice = st.selectbox("Select Model:", ["Decision Tree", "Random Forest"])

    if st.button("Analyze Email"):
        if email_text.strip():
            vectorized = email_vectorizer.transform([email_text])
            model = email_dt if model_choice == "Decision Tree" else email_rf
            prob = model.predict_proba(vectorized)[0][1]
            action, risk = agent_decision(prob)

            st.metric("Phishing Probability", f"{prob:.2f}")
            st.write("**Risk Level:**", risk)
            st.write("**Agent Action:**", action)
            show_agent_visual(action)
        else:
            st.warning("Please enter email content.")

with tab_website:
    st.subheader("Website Phishing Detection")

    url = st.text_input("Enter website URL:")
    model_choice = st.selectbox(
        "Select Model:", ["Decision Tree", "Random Forest"], key="web_model"
    )

    if st.button("Analyze Website"):
        if url.strip():
            features = cached_extract_website_features(url)
            vector = [features.get(f, 0) for f in website_features]

            model = website_dt if model_choice == "Decision Tree" else website_rf
            prob = model.predict_proba([vector])[0][1]
            action, risk = agent_decision(prob)

            st.metric("Phishing Probability", f"{prob:.2f}")
            st.write("**Risk Level:**", risk)
            st.write("**Agent Action:**", action)
            show_agent_visual(action)

            if st.checkbox("Show extracted features"):
                st.json(features)
        else:
            st.warning("Please enter a URL.")


with tab_image:
    st.subheader("Image / Screenshot Phishing Detection")

    uploaded_file = st.file_uploader(
        "Upload screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"]
    )

    model_choice = st.selectbox(
        "Select Model:", ["Decision Tree", "Random Forest"], key="img_model"
    )

    # OCR only when a NEW image is uploaded
    if uploaded_file and uploaded_file.name != st.session_state.last_image_name:
        img = Image.open(uploaded_file).convert("RGB")
        result = reader.readtext(np.array(img))
        st.session_state.extracted_text = " ".join([r[1] for r in result])
        st.session_state.last_image_name = uploaded_file.name

    # Persist extracted text across model changes
    if st.session_state.extracted_text:
        st.subheader("Extracted Text")
        st.text(st.session_state.extracted_text)

        if st.button("Analyze Image"):
            vectorized = email_vectorizer.transform(
                [st.session_state.extracted_text]
            )
            model = email_dt if model_choice == "Decision Tree" else email_rf
            prob = model.predict_proba(vectorized)[0][1]
            action, risk = agent_decision(prob)

            st.metric("Phishing Probability", f"{prob:.2f}")
            st.write("**Risk Level:**", risk)
            st.write("**Agent Action:**", action)
            show_agent_visual(action)

    elif uploaded_file:
        st.warning("No text detected in image.")
    else:
        st.info("Upload an image to begin.")