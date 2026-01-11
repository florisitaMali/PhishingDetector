Enron Spam Dataset

The Enron-Spam dataset is a fantastic ressource collected by V. Metsis, I. Androutsopoulos and G. Paliouras 
and described in their publication "Spam Filtering with Naive Bayes - Which Naive Bayes?". 

Link: https://www.kaggle.com/datasets/venky73/spam-mails-dataset
Source: https://www2.aueb.gr/users/ion/data/enron-spam/

Dataset type: Binary classification
Domain: Cybersecurity / Email phishing detection
Data format: CSV
Target variable: label
Size: 7705
Feature: Raw Text

COLUMNS: 

id:	The id of the email - not usable on our case
label:	Contains the value spam (Phishing) or ham (Legitimate)
text:	The content of the e-mail. Can contain an empty string if the message had only a subject line and no body. In case of forwarded emails or replies, this also contains the original message with subject line, "from:", "to:", etc.
label_num: 	0 for ham and 1 for spam

Important note: This dataset need preprocessing