import pandas as pd #to create the dataframe
import joblib #to save the trained models and objects to disk
from sklearn.model_selection import train_test_split #to split the dataset into training and testing
from sklearn.feature_extraction.text import TfidfVectorizer #convert the raw text into a TF-IDF feature vectors
from sklearn.tree import DecisionTreeClassifier #import the Decision Tree Classifier (version 1.8)
from sklearn.ensemble import RandomForestClassifier #import the RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #the performance metrics

#read the dataset and create the Dataframe
df = pd.read_csv(r"Data/dataset_2/raw/spam_ham_dataset_merged.csv")

#the training feature will be text 
X = df["text"]
#target will be the label column
y = df["label"]

#convert the text to TF-IDF vector -> convert text into numerical features
#remove the common English stop words
#limits  vocabulary size to the 5000 most important terms
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

#Learn the vocabulary and transform email text into TF-IDF vectors
X_vec = vectorizer.fit_transform(X)

#Split the dataset into training and testing sets -> 80% training, 20% testing by specifying test_size=0.2
#random state parameter help us to get the same split each time
#stratify=y means that both the training and the testing set contains the same distribution of the class as the original set
X_train, X_test, y_train, y_test = train_test_split( X_vec, y, test_size=0.2, random_state=42, stratify=y)

# Decision Tree
dt = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, random_state=42)
#Use the fit function for training
dt.fit(X_train, y_train)

# Random Forest
#n_estimators -> 200 decision trees in the forest
#random_state helps to reproduce the same result
#n_jobs uses all CPU cores for value -1, it can take the value 1(for single core), or 4(for 4 cores), but we use all the available cores to speed up training 
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
#use the fit function for training 
rf.fit(X_train, y_train)

#evaluation function to assess model performance on test data
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

#evaluate both models using the evaluation function
evaluate(dt, "Email Decision Tree")
evaluate(rf, "Email Random Forest")

#save trained models and the TF-IDF vectorizer for later usage so we do not need to retrain the models
#the function joblib.dump get the object (the tained model in this case), serializes it (convert into bytes), and saves to a disk as .pkl file
#if we do not use this files we have to train the model each time we want to make a prediction

#about the extension .pkl it stands for Pickle which is the file format for the python to store serialized python objects
joblib.dump(dt, "models/email_dt.pkl")
joblib.dump(rf, "models/email_rf.pkl")
joblib.dump(vectorizer, "models/email_vectorizer.pkl")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

#function to plot confusion matrix
def plot_confusion(model, name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=['ham','spam'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham','spam'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

#function to plot ROC curve
def plot_roc(model, name):
    #convert labels to binary (spam=1, ham=0)
    lb = LabelBinarizer()
    y_test_binary = lb.fit_transform(y_test)
    
    #predict probabilities
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]  # probability of class 'spam'
    else:
        #Decision Tree also has predict_proba, but just in case
        y_score = model.predict(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test_binary, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    plt.show()

#plot for Decision Tree
plot_confusion(dt, "Decision Tree")
plot_roc(dt, "Decision Tree")

#plot for Random Forest
plot_confusion(rf, "Random Forest")
plot_roc(rf, "Random Forest")
