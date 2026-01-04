def agent_decision(prob):
    if prob > 0.6:
        return "BLOCK ACCESS 🚫", "HIGH"
    elif prob >= 0.4:
        return "WARNING ⚠️", "MEDIUM"
    else:
        return "ALLOW ACCESS ✅", "LOW"
