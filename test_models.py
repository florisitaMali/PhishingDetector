import joblib

from agent.decision_logic import agent_decision
from agent.feature_extraction import extract_website_features

# -----------------------------
# Load EMAIL models
# -----------------------------
email_dt = joblib.load("models/email_dt.pkl")
email_rf = joblib.load("models/email_rf.pkl")
email_vectorizer = joblib.load("models/email_vectorizer.pkl")

# -----------------------------
# Load WEBSITE models
# -----------------------------
website_dt = joblib.load("models/website_dt.pkl")
website_rf = joblib.load("models/website_rf.pkl")
website_features = joblib.load("models/website_features.pkl")

print("\n=== EMAIL MODEL TEST ===")

sample_email = """
Dear user,
Your account has been suspended.
Please verify your login immediately.
"""

X_email = email_vectorizer.transform([sample_email])

for name, model in [("Decision Tree", email_dt), ("Random Forest", email_rf)]:
    prob = model.predict_proba(X_email)[0][1]
    action, risk = agent_decision(prob)
    print(f"{name}: Probability={prob:.2f}, Risk={risk}, Action={action}")

print("\n=== WEBSITE MODEL TEST ===")

sample_url = "http://secure-login-paypal.com"

features = extract_website_features(sample_url)
feature_vector = [features.get(f, 0) for f in website_features]

for name, model in [("Decision Tree", website_dt), ("Random Forest", website_rf)]:
    prob = model.predict_proba([feature_vector])[0][1]
    action, risk = agent_decision(prob)
    print(f"{name}: Probability={prob:.2f}, Risk={risk}, Action={action}")

print("\n✅ All models loaded and working correctly.")
