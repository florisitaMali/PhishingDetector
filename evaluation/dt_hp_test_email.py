import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product

df = pd.read_csv(r"Data\dataset_2\raw\spam_ham_dataset_merged.csv")

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split( X_vec, y, test_size=0.2, random_state=42, stratify=y)


def evaluate(model, params):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "Parameters": params,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="spam"),
        "Recall": recall_score(y_test, y_pred, pos_label="spam"),
        "F1 Score": f1_score(y_test, y_pred, pos_label="spam")
    }

results = []

criterion_values = ["gini", "entropy"]
splitter_values = ["best", "random"]
max_depth_values = [None, 10, 20, 30, 40]

for criterion, splitter, depth in product(criterion_values, splitter_values, max_depth_values):
    dt = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=depth,
        random_state=None
    )

    results.append(
        evaluate(
            dt,
            f"criterion={criterion}, splitter={splitter}, max_depth={depth}"
        )
    )

results_df = pd.DataFrame(results)
results_df.to_csv(
    "results/dt_email_hyperparameter_results.csv",
    index=False
)

print(results_df.sort_values("F1 Score", ascending=False))
