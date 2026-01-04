import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import tldextract

def fetch_page(url):
    """
    Fetch HTML content of the URL. Return empty string if inaccessible.
    """
    try:
        response = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        return response.text if response.status_code == 200 else ""
    except:
        return ""


def extract_website_features(url: str) -> dict:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    ext = tldextract.extract(url)
    domain = ext.domain

    html = fetch_page(url)
    soup = BeautifulSoup(html, "html.parser") if html else BeautifulSoup("", "html.parser")

    features = {}

    # ---------- URL / Lexical Features ----------
    features["UrlLength"] = len(url)
    features["NumDots"] = url.count(".")
    features["SubdomainLevel"] = hostname.count(".")
    features["PathLevel"] = path.count("/")
    features["NumDash"] = url.count("-")
    features["NumDashInHostname"] = hostname.count("-")
    features["NumUnderscore"] = url.count("_")
    features["NumQueryComponents"] = query.count("=")
    features["NumAmpersand"] = query.count("&")
    features["NumNumericChars"] = sum(c.isdigit() for c in url)
    features["NoHttps"] = 0 if parsed.scheme == "https" else 1
    features["RandomString"] = 1 if re.search(r"[a-zA-Z0-9]{10,}", hostname) else 0
    features["IpAddress"] = 1 if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", hostname) else 0
    features["HostnameLength"] = len(hostname)
    features["PathLength"] = len(path)
    features["QueryLength"] = len(query)

    # ---------- Domain / Lexical Patterns ----------
    features["DomainInSubdomains"] = 1 if domain in hostname else 0
    features["DomainInPaths"] = 1 if domain in path else 0

    # ---------- Sensitive words & brand abuse ----------
    sensitive_words = ["login","secure","account","verify","update","bank","paypal","confirm","password"]
    features["NumSensitiveWords"] = sum(word in url.lower() for word in sensitive_words)
    features["EmbeddedBrandName"] = 1 if domain and domain.lower() in path.lower() else 0

    # ---------- HTML Features ----------
    links = soup.find_all("a", href=True)
    ext_links = sum(1 for l in links if l["href"].startswith("http") and hostname not in l["href"])
    features["PctExtHyperlinks"] = ext_links / len(links) if links else 0

    resources = soup.find_all(["img","script","link"])
    ext_resources = sum(1 for r in resources if r.get("src") and hostname not in r.get("src",""))
    features["PctExtResourceUrls"] = ext_resources / len(resources) if resources else 0

    features["ExtFavicon"] = 1 if soup.find("link", rel="shortcut icon") and "http" in soup.find("link", rel="shortcut icon").get("href","") else 0

    forms = soup.find_all("form")
    insecure_forms = sum(1 for f in forms if parsed.scheme != "https")
    external_actions = sum(1 for f in forms if f.get("action","").startswith("http") and hostname not in f.get("action",""))
    relative_actions = sum(1 for f in forms if f.get("action","").startswith("/"))

    features["InsecureForms"] = 1 if insecure_forms > 0 else 0
    features["ExtFormAction"] = 1 if external_actions > 0 else 0
    features["RelativeFormAction"] = 1 if relative_actions > 0 else 0
    features["AbnormalFormAction"] = 1 if external_actions > 0 and relative_actions > 0 else 0

    features["PctNullSelfRedirectHyperlinks"] = 0 # optional: can be computed later
    features["FrequentDomainNameMismatch"] = 0
    features["FakeLinkInStatusBar"] = 0
    features["RightClickDisabled"] = 1 if "event.button==2" in html else 0
    features["PopUpWindow"] = 1 if "window.open" in html else 0
    features["SubmitInfoToEmail"] = 0
    features["IframeOrFrame"] = 1 if soup.find("iframe") else 0
    features["MissingTitle"] = 1 if not soup.title else 0
    features["ImagesOnlyInForm"] = 0

    # ---------- Rendered / Ratio Features (set 0 for now if too complex) ----------
    features["SubdomainLevelRT"] = features["SubdomainLevel"]
    features["UrlLengthRT"] = features["UrlLength"]
    features["PctExtResourceUrlsRT"] = features["PctExtResourceUrls"]
    features["AbnormalExtFormActionR"] = features["AbnormalFormAction"]
    features["ExtMetaScriptLinkRT"] = 0
    features["PctExtNullSelfRedirectHyperlinksRT"] = 0

    return features
