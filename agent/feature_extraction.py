import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import tldextract

# ---------------- Helper functions ----------------

def fetch_page(url: str) -> str:
    try:
        response = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        return response.text if response.status_code == 200 else ""
    except:
        return ""

def extract_visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)

def normalize_for_fetch(url: str) -> str:
    """ONLY used for fetching HTML"""
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url

# ---------------- Main feature extraction ----------------

def extract_website_features(url: str) -> dict:
    raw_url = url.strip()
    parsed_raw = urlparse(raw_url)

    has_scheme = raw_url.startswith(("http://", "https://"))
    is_https = raw_url.startswith("https://")

    # Normalized ONLY for HTML fetching
    fetch_url = normalize_for_fetch(raw_url)
    parsed_fetch = urlparse(fetch_url)

    hostname = parsed_fetch.hostname or ""
    path = parsed_fetch.path or ""
    query = parsed_fetch.query or ""

    ext = tldextract.extract(fetch_url)
    domain = ext.domain

    features = {}

    # ---------- Scheme features ----------
    features["MissingScheme"] = 0 if has_scheme else 1
    features["NoHttps"] = 0 if is_https else 1

    # ---------- URL / Lexical (RAW URL) ----------
    features["UrlLength"] = len(raw_url)
    features["NumDots"] = raw_url.count(".")
    features["NumDash"] = raw_url.count("-")
    features["NumUnderscore"] = raw_url.count("_")
    features["NumNumericChars"] = sum(c.isdigit() for c in raw_url)

    # ---------- Host features ----------
    features["HostnameLength"] = len(hostname)
    features["SubdomainLevel"] = hostname.count(".")
    features["IpAddress"] = 1 if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", hostname) else 0

    # ---------- Phishing lexical patterns ----------
    sensitive_words = [
        "secure","verify","account","update","confirm",
        "bank","paypal","gmail","yahoo"
    ]
    TRUSTED_DOMAINS = {
        "github", "google", "amazon", "microsoft", "wikipedia"
    }

    features["NumSensitiveWords"] = sum(
        word in raw_url.lower() for word in sensitive_words
    ) if domain not in TRUSTED_DOMAINS else 0


    # Typosquatting / brand abuse
    features["LooksLikeBrand"] = 1 if re.search(
        r"(paypal|amazon|google|bank|gmail|yahoo|github)[0-9\-]",
        raw_url.lower()
    ) else 0

    features["RandomString"] = 1 if re.search(
        r"(?:[a-z]{5,}[0-9]{3,}|[0-9]{3,}[a-z]{5,})",
        hostname.lower()
    ) else 0

    features["DomainInPath"] = 1 if domain and domain in path.lower() else 0

    # ---------- HTML Features ----------
    html = fetch_page(fetch_url)
    features["CouldFetchHTML"] = 1 if html else 0

    soup = BeautifulSoup(html, "html.parser") if html else BeautifulSoup("", "html.parser")
    visible_text = extract_visible_text(soup)

    links = soup.find_all("a", href=True)
    ext_links = sum(
        1 for l in links
        if l["href"].startswith("http") and hostname not in l["href"]
    )
    features["PctExtHyperlinks"] = ext_links / len(links) if links else 0

    resources = soup.find_all(["img", "script", "link"])
    ext_resources = sum(
        1 for r in resources
        if r.get("src") and hostname not in r.get("src", "")
    )
    features["PctExtResourceUrls"] = ext_resources / len(resources) if resources else 0

    features["IframeOrFrame"] = 1 if soup.find("iframe") else 0
    features["MissingTitle"] = 1 if not soup.title else 0
    features["PopUpWindow"] = 1 if "window.open" in html else 0
    features["RightClickDisabled"] = 1 if "event.button==2" in html else 0

    features["_visible_text"] = visible_text
    return features
