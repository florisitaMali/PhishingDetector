import streamlit as st #to build web application
import streamlit.components.v1 as components #to render custom HTML components inside streamlit
import pandas as pd #to create dataframe
import joblib #used to import the models
import numpy as np #used for numeric operations and for handling array
from PIL import Image #used to open and process the images
import easyocr #OCR - Optical Character Recognition - library for extracting text from images
import base64 #used to encode the images (GIFs - assiciated with the action suggested) into base 64 for HTML embendings
 
#configure the streamlit page 
#set the tittle and the layout centered
st.set_page_config(
    page_title="Multi-Modal Phishing Detection Agent",
    layout="centered"
)

#set the session state for the extracted text from the images
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

#save the last image name to prevent the ocr to extract the text from the same image
if "last_image_name" not in st.session_state:
    st.session_state.last_image_name = None

#annotation used to cashed the models into the browser cache
@st.cache_resource
def load_models_and_ocr():
    #load the email models
    email_dt_model = joblib.load("models/email_dt.pkl")
    email_rf_model = joblib.load("models/email_rf.pkl")
    #load the trained TF-IDF vector for email text
    email_vec = joblib.load("models/email_vectorizer.pkl")

    #load the website model
    website_dt_model = joblib.load("models/website_dt.pkl")
    website_rf_model = joblib.load("models/website_rf.pkl")
    #load the website features
    website_feats = joblib.load("models/website_features.pkl")

    #initialize OCR reader on english language and the gpu enabled 
    ocr_reader = easyocr.Reader(['en'], gpu=True)

    #return a tupple of models for both the website and email, the TF-IDF vcector for email, the website features and the ocr reader
    return (
        email_dt_model,
        email_rf_model,
        email_vec,
        website_dt_model,
        website_rf_model,
        website_feats,
        ocr_reader,
    )

# extract the website feature and save them to cache so it is not recalculated
@st.cache_data
def cached_extract_website_features(url):
    #import the function extract_website_features
    from agent.feature_extraction import extract_website_features
    #return the features of the websites based on the url 
    return extract_website_features(url)

#get all models and the resource
email_dt, email_rf, email_vectorizer, website_dt, website_rf, website_features, reader = load_models_and_ocr()

#import the agent decision rules -> decision done based on the propability
from agent.decision_logic import agent_decision

#this function shows the GIF for each action 
def show_agent_visual(action):
    #convert to upercase and remove the spaces in the begining or the end of the action for better comparison
    action = action.upper().strip()

    #For each action select the appropriate gif and caption
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

    #encode gif as Base64 to embed directly to HTML
    with open(gif_path, "rb") as f:
        gif_base64 = base64.b64encode(f.read()).decode()

    #create the custom HTML
    html = f"""
    <div style="text-align:center">
        <img src="data:image/gif;base64,{gif_base64}" width="260">
        <p style="font-weight:bold; font-size:16px;">{caption}</p>
    </div>
    """
    components.html(html, height=320)

# Title 
st.title("Multi-Modal Phishing Detection Agent")

# A short description
st.markdown("""
This intelligent agent detects phishing in **emails**, **websites**, and **screenshots**.
- **ALLOW** – low-risk
- **WARN** – medium-risk
- **BLOCK** – high-risk
""")

#creates each tab for each phishing detection mode
tab_email, tab_website, tab_image = st.tabs(["Email Phishing", "Website Phishing", "Image Phishing"])

# Email Phishing Tab
with tab_email:
    #set the header text of the tab
    st.subheader("Email Phishing Detection")

    #text area input with appropriate prompt
    email_text = st.text_area("Enter email content:")
    #a dropdown for selection of the model for training
    model_choice = st.selectbox("Select Model:", ["Decision Tree", "Random Forest"])

    #if the button Analyze Email is clicked -> trigger email analysis
    if st.button("Analyze Email"):
        #if the text putted on the text area input is not empty, or it does not contain only white spaces then anayze the email
        if email_text.strip():
            #first we vectorize the text 
            vectorized = email_vectorizer.transform([email_text])
            #based on the selection we get the model
            model = email_dt if model_choice == "Decision Tree" else email_rf
            #based on the model we predict the propability using the model.predict_prob() function with input the vectorized email
            #this function returns an array of the form [[prob_for_spam, [prob_for_nonspam]] so we used the indexes [0][1] to get the prob for spam email
            prob = model.predict_proba(vectorized)[0][1]
            #based on the propability the agent make the decision by using the function bellow
            action, risk = agent_decision(prob)

            #Print the result
            st.metric("Phishing Probability", f"{prob:.2f}")
            st.write("**Risk Level:**", risk)
            st.write("**Agent Action:**", action)
            #show the gif
            show_agent_visual(action)
        else:
            #if there is no content show the warning
            st.warning("Please enter email content.")

with tab_website:
    #write the tab title for the Website tab
    st.subheader("Website Phishing Detection")

    #text input for entering the url
    url = st.text_input("Enter website URL:")
    #dropdown to select the model
    model_choice = st.selectbox( "Select Model:", ["Decision Tree", "Random Forest"], key="web_model")

    #if the Analyze Website button is clicked it triggers an event for analyzing the website
    if st.button("Analyze Website"):
        #first we check if the url is entered
        if url.strip():
            #we get the features
            features = cached_extract_website_features(url)
            
            #we create a vector for features
            vector = [features.get(f, 0) for f in website_features]

            #we get the trained model based on the selection
            model = website_dt if model_choice == "Decision Tree" else website_rf
            #get the propability for spam 
            prob = model.predict_proba([vector])[0][1]
            #get the recommended action based on the propability adn the risk
            action, risk = agent_decision(prob)

            #print the result
            st.metric("Phishing Probability", f"{prob:.2f}")
            st.write("**Risk Level:**", risk)
            st.write("**Agent Action:**", action)
            #show the gif
            show_agent_visual(action)
        else:
            #if no url was written show the warning
            st.warning("Please enter a URL.")

# Image tab
with tab_image:
    #set the title for image tab
    st.subheader("Image / Screenshot Phishing Detection")

    #file uploader
    uploaded_file = st.file_uploader("Upload screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])
    #dropdown for model selection
    model_choice = st.selectbox("Select Model:", ["Decision Tree", "Random Forest"], key="img_model")

    #if the an image is uploaded and it is a new image
    if uploaded_file and uploaded_file.name != st.session_state.last_image_name:
        #open the image and convert that to RGB model (Red, Green, Blue)
        img = Image.open(uploaded_file).convert("RGB")

        #extract the text from the image 
        #np.array convert the image into array of pixels
        #result is of the form:
        #r[0] -> coordinates of the text box
        #r[1] -> the text detected
        #r[2] -> confidence score
        result = reader.readtext(np.array(img))
        #save the name and the text of the image to the session state so it is not extracted again when the model is changed
        st.session_state.extracted_text = " ".join([r[1] for r in result])
        st.session_state.last_image_name = uploaded_file.name

    #if the extracted text is not empty then show the image Analyze Image to move on with prediction
    if st.session_state.extracted_text:
        #the subsection title
        st.subheader("Extracted Text")
        #print the text detected
        st.text(st.session_state.extracted_text)

        #if the button Analyze the Image is clicked than trigger the output prediction
        if st.button("Analyze Image"):
            #Same process as the email tab above  
            vectorized = email_vectorizer.transform([st.session_state.extracted_text])
            model = email_dt if model_choice == "Decision Tree" else email_rf
            prob = model.predict_proba(vectorized)[0][1]
            action, risk = agent_decision(prob)

            st.metric("Phishing Probability", f"{prob:.2f}")
            st.write("**Risk Level:**", risk)
            st.write("**Agent Action:**", action)
            show_agent_visual(action)
    #if there is not any text in the image show the warning about the image
    elif uploaded_file:
        st.warning("No text detected in image.")
    #show an info message to upload an image to begin the prediction
    else:
        st.info("Upload an image to begin.")
