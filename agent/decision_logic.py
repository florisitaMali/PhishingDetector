def agent_decision(phishing_prob: float, url: str = None):
    # ML-based decision
    if phishing_prob >= 0.7:
        return "BLOCK 🚫", "HIGH"
    elif phishing_prob >= 0.4:
        return "WARN ⚠️", "MEDIUM"
    else:
        return "ALLOW ✅", "LOW"
