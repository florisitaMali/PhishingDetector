import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("Data/dataset_1/raw/Phishing_Legitimate_Websites.csv")

# Drop unnecessary columns
X = df.drop(columns=['CLASS_LABEL', 'URL', 'Unnamed: 0'])
y = df['CLASS_LABEL']  # target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train models
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(
    n_estimators=1000,
    random_state=42,
    n_jobs=-1
)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluation function
def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))

evaluate(dt, "Website Decision Tree")
evaluate(rf, "Website Random Forest")

# Save models and features
joblib.dump(dt, "models/website_dt.pkl")
joblib.dump(rf, "models/website_rf.pkl")
joblib.dump(X.columns.tolist(), "models/website_features.pkl")
