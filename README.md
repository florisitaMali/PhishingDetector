# Multi-Modal Phishing Detection Agent
## Project Overview
This project implements a **Multi-Modal Intelligent Phishing Detection Agent** capable of detecting phishing attempts across **emails**, **websites**, and **image-based content (screenshots)**. The system is designed as an intelligent agent that not only classifies inputs as phishing or legitimate, but also **reasons about risk** and selects a rational protective action: **Allow**, **Warn**, or **Block**.


## Group Members and Roles

* **Florisita Mali** – Dataset preparation, model training of model for website dataset
* **Gresja Kulejmani** – Dataset preparation, model training of model for email dataset
* **Ledio Hima** – User interface development (Streamlit),visualization and quantitative evaluation

All team members contributed to system design, testing, and report writing.

## Implemented AI Approach

The system combines two artificial intelligence paradigms:

### 1. Statistical Learning (Supervised Machine Learning)

Two classifiers are trained and evaluated for both website and email phishing detection:

* **Decision Tree Classifier** – Used as a baseline due to its simplicity and interpretability
* **Random Forest Classifier** – Used as the primary model due to improved generalization and reduced overfitting

**Email phishing detection** uses **TF-IDF vectorization** to transform textual email content into numerical features before classification.

**Website phishing detection** uses structured lexical and content-based features extracted directly from URLs and webpages.
*Image phishing detection** uses Optical Character Recognition (OCR)  to extract visible textual content from the image. This step converts visual information into textual data that the agent can reason about.

### 2. Logical Reasoning (Rule-Based System)

The output of the machine learning model is a phishing probability, which is passed to a rule-based decision module:

* Probability > 0.60 → **Block access**
* Probability between >=0.40 → **Warn the user**
* Probability < 0.40 → **Allow access**

This reasoning layer ensures transparent and rational decision-making.

## Installation and Dependencies

### Requirements

* Python 3.9 or later

### Required Python Libraries

Install all dependencies using:

```bash
pip install streamlit pandas numpy scikit-learn joblib pillow easyocr
```

**Note:** EasyOCR may require additional system dependencies for OCR support. GPU acceleration is optional.

---

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/florisitaMali/PhishingDetector.git
cd PhishingDetector
```

2. Ensure the trained models are present in the `models/` directory.

3. Run the Streamlit application:

```bash
streamlit run app.py
```

4. Open the provided local URL in a web browser to access the interface.

---

## Features

* Email phishing detection
* Website phishing detection
* Image/screenshot phishing detection using OCR
* Probabilistic risk assessment
* Rational agent actions (Allow / Warn / Block)
* Interactive user interface

---

## Notes on Reproducibility

* Model training scripts are provided in the `training/` directory
* Trained models are saved using `joblib` for reuse
* All code is implemented in Python following course guidelines

