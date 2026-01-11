import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product

# Load dataset
df = pd.read_csv("Data/dataset_1/raw/Phishing_Legitimate_Websites.csv")

# Features and target
X = df.drop(columns=["CLASS_LABEL", "Unnamed: 0"])
y = df["CLASS_LABEL"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate(model, params):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "Parameters": params,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1),
        "Recall": recall_score(y_test, y_pred, pos_label=1),
        "F1 Score": f1_score(y_test, y_pred, pos_label=1)
    }

criterion_values = ["gini", "entropy"]
splitter_values = ["best", "random"]
max_depth_values = [None, 10, 20, 30, 40, 50]

results = []

for criterion, splitter, depth in product(
    criterion_values,
    splitter_values,
    max_depth_values
):
    dt = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=depth,
        random_state=42
    )

    results.append(
        evaluate(
            dt,
            f"criterion={criterion}, splitter={splitter}, max_depth={depth}"
        )
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/dt_website_hyperparameter_results.csv",
    index=False
)

print("\nTop 5 configurations by F1 Score:")
print(results_df.sort_values("F1 Score", ascending=False).head())