import pandas as pd #used to create dataframe
import joblib #used to store the models into pickle files
from sklearn.model_selection import train_test_split #used to separate the dataset into train and test
from sklearn.tree import DecisionTreeClassifier #import the Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier #import the Random Forest Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # performace measurements -> explained better better bellow

#load dataset
df = pd.read_csv("Data/dataset_1/raw/Phishing_Legitimate_Websites.csv")

#drop unnecessary columns -> Unnamed: 0 is the ID
#drop the class lable so x contains only the featues
X = df.drop(columns=['CLASS_LABEL', 'Unnamed: 0'])
y = df['CLASS_LABEL']  #target

#split dataset into training and testing -> 80% training, 20% testing by specifying test_size = 0.2
#random_state to reproduce the same result
#stratify=y mean make the split based on the target for the training and testing set same as on the original set 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

# Train models
# Decision Tree
#random state to reproduce same output
dt = DecisionTreeClassifier(random_state=42)
# Random Forest
#n_estimators -> 1000 decision trees used for making the decision,
#random state for reproducing the same result
#n_jobs use all the available CPU cores 
rf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)

#train both models 
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluation function
def evaluate(model, name):
    #make the prediction
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    #calculate the acuracy (total_correct_predictions / total_predictions)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    #calculate the precision score (true_positive / (true_positive + false_positive))
    #meaning: among the emails predicted as spam, which fraction are actually spam
    print("Precision:", precision_score(y_test, y_pred, pos_label='spam'))
    # calculate the recall score (true_positive / (true_positive + false_negative))
    #meaning: among the actual spam emails, which fraction are correctly detected
    print("Recall:", recall_score(y_test, y_pred, pos_label='spam'))
    # calculate the F1 score - harmonic mean of Precision and Recall (2*P*R / (P+R))
    print("F1 Score:", f1_score(y_test, y_pred, pos_label='spam'))

# Evaluation of the models using the evalution function
evaluate(dt, "Website Decision Tree")
evaluate(rf, "Website Random Forest")

#save models and features on files so we do not need to retrain the model each time we need to use them 
#we use joblit.dump to read to save the objects after serializing them into pickle files
#we save both the models and the features
joblib.dump(dt, "models/website_dt.pkl")
joblib.dump(rf, "models/website_rf.pkl")
#we save the features so when we need to use them we use the same features that the models are being trained
joblib.dump(X.columns.tolist(), "models/website_features.pkl")
