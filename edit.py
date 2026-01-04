import pandas as pd
from agent.feature_extraction import extract_website_features

# Load existing dataset
df = pd.read_csv(r"Data\dataset_1/raw\Phishing_Legitimate_Websites.csv")

# -----------------------------
# Add synthetic phishing URLs
# -----------------------------
phishing_urls = [
    "gmai1.com",
    "icici6ank.com",
    "bank0findia.com",
    "login-paypal-secure-account.com",
    "verify-account-update-login.xyz",
    "secure-bank-login-confirm.com",
    "paypal-account-verification-login.info",
    "appleid-login-support-secure.com",
    "update-your-account-paypal-login.com"
]

for url in phishing_urls:
    features = extract_website_features(url)
    features['CLASS_LABEL'] = 1  # phishing
    df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)

# -----------------------------
# Add legitimate URLs
# -----------------------------
legitimate_urls = [
    "https://www.google.com",
    "https://www.wikipedia.org",
    "https://github.com/login",
    "https://www.amazon.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.stackoverflow.com"
]

for url in legitimate_urls:
    features = extract_website_features(url)
    features['CLASS_LABEL'] = 0  # legitimate
    df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)

# -----------------------------
# Save updated dataset
# -----------------------------
df.to_csv(r"Data\dataset_1/raw\Phishing_Legitimate_Websites.csv", index=False)

print("✅ Dataset updated with phishing + legitimate URLs.")
