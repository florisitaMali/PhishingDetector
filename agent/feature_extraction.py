import re #used for finding patterns in strings
import requests #send and http request and fetch the html content
from urllib.parse import urlparse #used to parse and analyze the url components
from bs4 import BeautifulSoup #used to parse HTML content and extract HTML elements
import tldextract #used to extract top level domain, domain and subdomain
import joblib #used to load the trained models
import pandas as pd #used to create DataFrame

def fetch_page(url: str) -> str:
    #try to fetch the HTML content 
    #if it fails it return empty string in except block
    try:
        # send a GET request, on the specified url and timeout 5 sec
        # User-Agent header is  included to mimic a real browser
        r = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        # it the request is successfull then return the content, otherwise an empty string
        if r.status_code == 200:
            return r.text
        else:
            return ""
    except:
        return ""

def normalize_url(url: str) -> str:
    #normalize the url by adding the http:// if it is missed
    if not url.startswith(("http://", "https://")):
        return "http://" + url
    return url

def extract_website_features(url: str) -> dict:
    #removes the spaces before and after the url
    raw_url = url.strip()
    #normalize the url -> if it does not contain http:// or https:// we add http://
    norm_url = normalize_url(raw_url)

    #url parse split the url into domain, path, query
    parsed = urlparse(norm_url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # tldextract separates the main domain from subdomain
    ext = tldextract.extract(norm_url)
    #main domain
    domain = ext.domain
    #subdomain
    subdomain = ext.subdomain

    #form the function fetch_page we etract the content (int the form of html) of the url. It the url does not 
    html = fetch_page(norm_url)

    #soup is a BeaurifulSoup object which parse the HTML and extract the html tags
    soup = BeautifulSoup(html, "html.parser")

    #feature dictionary
    features = {}

    #these are the url features
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

    #domain features
    features["IpAddress"] = 1 if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", hostname) else 0
    features["HostnameLength"] = len(hostname)
    features["PathLength"] = len(path)
    features["QueryLength"] = len(query)
    features["DomainInSubdomains"] = 1 if domain and domain in subdomain.lower() else 0
    features["DomainInPaths"] = 1 if domain and domain in path.lower() else 0

    #detects random looking strings -> combination of letters and digits
    features["RandomString"] = 1 if re.search(
        r"[a-z]+[0-9]+[a-z]*|[a-z]*[0-9]+[a-z]+",
        domain.lower()
    ) else 0

    #sensitive words -> phishing often mimics banks/brands
    sensitive = [
        "login","secure","verify","update","account",
        "confirm","bank","paypal","apple","google","gmail"
    ]
    features["NumSensitiveWords"] = sum(w in raw_url.lower() for w in sensitive)
    #re.seach check if the pattern is found on the string
    features["EmbeddedBrandName"] = 1 if re.search(
        r"(gmai|gmail|yah|yahoo|paypa|paypal|bank|apple|googl)",
        domain.lower()
    ) else 0

    #from soup find the tag <a> which represents links into html
    links = soup.find_all("a", href=True)
    #from soup get all the forms
    forms = soup.find_all("form")
    #find the resouces like images(<img>), js code (<script>), css (<link>)
    resources = soup.find_all(["img", "script", "link"])

    #external hyperlinks -> phishing websites usually contains links to external domains
    ext_links = [l for l in links if l["href"].startswith("http") and hostname not in l["href"]]
    #external resouces would be suspicious if there are many included
    ext_resources = [r for r in resources if r.get("src") and hostname not in r.get("src", "")]

    #extract the content features
    #fraction of external links over the all links
    features["PctExtHyperlinks"] = len(ext_links) / len(links) if links else 0
    #fraction of external resources over the actual resources
    features["PctExtResourceUrls"] = len(ext_resources) / len(resources) if resources else 0
    #get the favicons
    features["ExtFavicon"] = 1 if soup.find("link", rel=lambda x: x and "icon" in x.lower()) else 0
    #in cases where the form's action attribute has as value an external link it means it is suspicous
    features["InsecureForms"] = 1 if any(f.get("action", "").startswith("http://") for f in forms) else 0
    #relative from the actual directory
    features["RelativeFormAction"] = 1 if any(f.get("action", "").startswith("/") for f in forms) else 0
    #external forms, meaning the value of the action parameter is a link not on the actual domain
    features["ExtFormAction"] = 1 if any(f.get("action", "").startswith("http") and hostname not in f.get("action", "") for f in forms) else 0
    #forms that does not redirect somewhere, ghost forms
    features["AbnormalFormAction"] = 1 if any(f.get("action") in ("", None) for f in forms) else 0
    #calculates the percentage of hyperlinks that do not redirect anywhere meaningful
    features["PctNullSelfRedirectHyperlinks"] = sum(l["href"] in ("#", "", "javascript:void(0)") for l in links) / len(links) if links else 0
    #we cannot check the frequent of domain name mismatch so we put is as 0
    features["FrequentDomainNameMismatch"] = 0
    #fake links shown in the browser status bar using JavaScript onmouseover events
    features["FakeLinkInStatusBar"] = 1 if "onmouseover" in html.lower() else 0
    #check if the right-click functionality is disabled using JavaScript
    features["RightClickDisabled"] = 1 if "event.button==2" in html.lower() else 0
    #check if we have pop-up windows which are common in phishing
    features["PopUpWindow"] = 1 if "window.open" in html.lower() else 0
    #check if there is any email sented
    features["SubmitInfoToEmail"] = 1 if "mailto:" in html.lower() else 0
    #check the presence of the frame and the iframe of the website
    features["IframeOrFrame"] = 1 if soup.find("iframe") else 0
    #check if the tittle miss
    features["MissingTitle"] = 1 if not soup.title else 0
    #it is suposed wether the forms contains only images rather than the input fields
    features["ImagesOnlyInForm"] = 0

    #check if the subdomain level is more than 2 -> often a feature of phishing
    features["SubdomainLevelRT"] = 1 if features["SubdomainLevel"] > 2 else 0
    #check the legth of the url -> long means more propability for phishing
    features["UrlLengthRT"] = 1 if features["UrlLength"] > 75 else 0
    #flag the page as risky if there are more than 50% of its resouce urls external
    features["PctExtResourceUrlsRT"] = 1 if features["PctExtResourceUrls"] > 0.5 else 0
    features["AbnormalExtFormActionR"] = features["AbnormalFormAction"]
    #flags the page as risky if more than 30% of meta, script, or link resources are loaded from external domains
    features["ExtMetaScriptLinkRT"] = 1 if features["PctExtResourceUrls"] > 0.3 else 0
    #flags the page as risky if more than 30% of hyperlinks are null or self-redirecting
    features["PctExtNullSelfRedirectHyperlinksRT"] = 1 if features["PctNullSelfRedirectHyperlinks"] > 0.3 else 0

    return features

#load the trained models
#Load the Decision Tree model
dt_model = joblib.load("models/website_dt.pkl")
#load the Random Forest mode
rf_model = joblib.load("models/website_rf.pkl")
#load the website features that are used during model training
feature_columns = joblib.load("models/website_features.pkl")

#make the prediction about the website
def predict_website(url: str):
    #normalize first
    url = normalize_url(url)

    #extract features
    features_dict = extract_website_features(url)

    #convert to DataFrame with proper columns
    X_new = pd.DataFrame([features_dict], columns=feature_columns)

    #Predictions of the phishing class using the Decision Tree model
    pred_dt = dt_model.predict(X_new)[0]
    #Get the phishing propability 
    prob_dt = dt_model.predict_proba(X_new)[0][1]

    #Predictions of the phishing class using the Random Forest model
    pred_rf = rf_model.predict(X_new)[0]
    #Get the phishing propability 
    prob_rf = rf_model.predict_proba(X_new)[0][1]

    #Return a dictionary where the keys are the models used and the values are
    #value is also a dictionary with the class and the phishing propability as keys and their repective values
    return {
        "DecisionTree": {"class": int(pred_dt), "phishing_prob": float(prob_dt)},
        "RandomForest": {"class": int(pred_rf), "phishing_prob": float(prob_rf)}
    }
