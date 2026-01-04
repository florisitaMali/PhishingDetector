import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv(r"C:\Users\malif\OneDrive\Desktop\All\Sem 5\Projects - Sem 5\PhishingDetector\Data\dataset_2\raw\spam_ham_dataset_merged.csv")

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label='spam'))
    print("Recall:", recall_score(y_test, y_pred, pos_label='spam'))
    print("F1 Score:", f1_score(y_test, y_pred, pos_label='spam'))


evaluate(dt, "Email Decision Tree")
evaluate(rf, "Email Random Forest")

joblib.dump(dt, "models/email_dt.pkl")
joblib.dump(rf, "models/email_rf.pkl")
joblib.dump(vectorizer, "models/email_vectorizer.pkl")
