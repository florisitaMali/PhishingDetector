import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import tldextract
import numpy as np
import joblib
import pandas as pd

# -----------------------------
# Helper functions
# -----------------------------
def fetch_page(url: str) -> str:
    """Fetch HTML content of a webpage, return None if fails."""
    try:
        r = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            return r.text
        else:
            return ""
    except:
        return ""

def normalize_url(url: str) -> str:
    """Ensure URL has a scheme."""
    if not url.startswith(("http://", "https://")):
        return "http://" + url
    return url

# -----------------------------
# Feature extractor
# -----------------------------
def extract_website_features(url: str) -> dict:
    raw_url = url.strip()
    norm_url = normalize_url(raw_url)

    parsed = urlparse(norm_url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    ext = tldextract.extract(norm_url)
    domain = ext.domain
    subdomain = ext.subdomain

    html = fetch_page(norm_url)
    if html is None:
        # URL not reachable
        return None

    soup = BeautifulSoup(html, "html.parser")

    features = {}

    # -----------------------------
    # URL / lexical features
    # -----------------------------
    features["NumDots"] = raw_url.count(".")
    features["SubdomainLevel"] = subdomain.count(".") + (1 if subdomain else 0)
    features["PathLevel"] = path.count("/")
    features["UrlLength"] = len(raw_url)
    features["NumDash"] = raw_url.count("-")
    features["NumDashInHostname"] = hostname.count("-")
    features["NumUnderscore"] = raw_url.count("_")
    features["NumQueryComponents"] = query.count("=")
    features["NumAmpersand"] = query.count("&")
    features["NumNumericChars"] = sum(c.isdigit() for c in raw_url)
    features["NoHttps"] = 0 if raw_url.startswith("https://") else 1

    # -----------------------------
    # Domain structure
    # -----------------------------
    features["IpAddress"] = 1 if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", hostname) else 0
    features["HostnameLength"] = len(hostname)
    features["PathLength"] = len(path)
    features["QueryLength"] = len(query)
    features["DomainInSubdomains"] = 1 if domain and domain in subdomain.lower() else 0
    features["DomainInPaths"] = 1 if domain and domain in path.lower() else 0

    # -----------------------------
    # Random string / typosquatting
    # -----------------------------
    features["RandomString"] = 1 if re.search(
        r"[a-z]+[0-9]+[a-z]*|[a-z]*[0-9]+[a-z]+",
        domain.lower()
    ) else 0

    # -----------------------------
    # Sensitive words / brand
    # -----------------------------
    sensitive = [
        "login","secure","verify","update","account",
        "confirm","bank","paypal","apple","google","gmail"
    ]
    features["NumSensitiveWords"] = sum(w in raw_url.lower() for w in sensitive)
    features["EmbeddedBrandName"] = 1 if re.search(
        r"(gmai|gmail|yah|yahoo|paypa|paypal|bank|apple|googl)",
        domain.lower()
    ) else 0

    # -----------------------------
    # HTML / content features
    # -----------------------------
    links = soup.find_all("a", href=True)
    forms = soup.find_all("form")
    resources = soup.find_all(["img", "script", "link"])

    ext_links = [l for l in links if l["href"].startswith("http") and hostname not in l["href"]]
    ext_resources = [r for r in resources if r.get("src") and hostname not in r.get("src", "")]

    features["PctExtHyperlinks"] = len(ext_links) / len(links) if links else 0
    features["PctExtResourceUrls"] = len(ext_resources) / len(resources) if resources else 0
    features["ExtFavicon"] = 1 if soup.find("link", rel=lambda x: x and "icon" in x.lower()) else 0
    features["InsecureForms"] = 1 if any(f.get("action", "").startswith("http://") for f in forms) else 0
    features["RelativeFormAction"] = 1 if any(f.get("action", "").startswith("/") for f in forms) else 0
    features["ExtFormAction"] = 1 if any(f.get("action", "").startswith("http") and hostname not in f.get("action", "") for f in forms) else 0
    features["AbnormalFormAction"] = 1 if any(f.get("action") in ("", None) for f in forms) else 0
    features["PctNullSelfRedirectHyperlinks"] = sum(l["href"] in ("#", "", "javascript:void(0)") for l in links) / len(links) if links else 0
    features["FrequentDomainNameMismatch"] = 0
    features["FakeLinkInStatusBar"] = 1 if "onmouseover" in html.lower() else 0
    features["RightClickDisabled"] = 1 if "event.button==2" in html.lower() else 0
    features["PopUpWindow"] = 1 if "window.open" in html.lower() else 0
    features["SubmitInfoToEmail"] = 1 if "mailto:" in html.lower() else 0
    features["IframeOrFrame"] = 1 if soup.find("iframe") else 0
    features["MissingTitle"] = 1 if not soup.title else 0
    features["ImagesOnlyInForm"] = 0

    # -----------------------------
    # Risk-transformed (RT)
    # -----------------------------
    features["SubdomainLevelRT"] = 1 if features["SubdomainLevel"] > 2 else 0
    features["UrlLengthRT"] = 1 if features["UrlLength"] > 75 else 0
    features["PctExtResourceUrlsRT"] = 1 if features["PctExtResourceUrls"] > 0.5 else 0
    features["AbnormalExtFormActionR"] = features["AbnormalFormAction"]
    features["ExtMetaScriptLinkRT"] = 1 if features["PctExtResourceUrls"] > 0.3 else 0
    features["PctExtNullSelfRedirectHyperlinksRT"] = 1 if features["PctNullSelfRedirectHyperlinks"] > 0.3 else 0

    return features

# -----------------------------
# Load models and feature names
# -----------------------------
dt_model = joblib.load("models/website_dt.pkl")
rf_model = joblib.load("models/website_rf.pkl")
feature_columns = joblib.load("models/website_features.pkl")

# -----------------------------
# Predict function
# -----------------------------
def predict_website(url: str):
    # Normalize first
    url = normalize_url(url)

    # Try fetching the page
    html = fetch_page(url)
    if not html:  # empty string or None → URL unreachable
        return "This URL does not exist."

    # Extract features
    features_dict = extract_website_features(url)
    if features_dict is None:
        return "This URL does not exist."

    # Convert to DataFrame with proper columns
    X_new = pd.DataFrame([features_dict], columns=feature_columns)

    # Predictions
    pred_dt = dt_model.predict(X_new)[0]
    prob_dt = dt_model.predict_proba(X_new)[0][1]

    pred_rf = rf_model.predict(X_new)[0]
    prob_rf = rf_model.predict_proba(X_new)[0][1]

    return {
        "DecisionTree": {"class": int(pred_dt), "phishing_prob": float(prob_dt)},
        "RandomForest": {"class": int(pred_rf), "phishing_prob": float(prob_rf)}
    }
