def agent_decision(phishing_prob: float):
    # make the decision based on the propability
    if phishing_prob >= 0.6:
        return "BLOCK", "HIGH"
    elif phishing_prob >= 0.4:
        return "WARN", "MEDIUM"
    else:
        return "ALLOW", "LOW"
