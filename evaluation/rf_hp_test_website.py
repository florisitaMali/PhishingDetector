import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product

df = pd.read_csv("Data/dataset_1/raw/Phishing_Legitimate_Websites.csv")

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

n_estimators_values = [100, 300, 500]          
criterion_values = ["gini", "entropy"]         
max_depth_values = [None, 10, 20, 30]          
max_features_values = ["sqrt", "log2", None]   

results = []

for n_estimators, criterion, max_depth, max_features in product(
    n_estimators_values,
    criterion_values,
    max_depth_values,
    max_features_values
):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    results.append(
        evaluate(
            rf,
            f"n_estimators={n_estimators}, criterion={criterion}, max_depth={max_depth}, max_features={max_features}"
        )
    )

results_df = pd.DataFrame(results)
results_df.to_csv("results/rf_website_hyperparameter_results.csv", index=False)

print("\nTop 5 Random Forest configurations by F1 Score:")
print(results_df.sort_values("F1 Score", ascending=False).head())