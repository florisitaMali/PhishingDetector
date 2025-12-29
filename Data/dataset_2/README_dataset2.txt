Enron Spam Dataset

The Enron-Spam dataset is a fantastic ressource collected by V. Metsis, I. Androutsopoulos and G. Paliouras 
and described in their publication "Spam Filtering with Naive Bayes - Which Naive Bayes?". 
The dataset contains a total of 17.171 spam and 16.545 non-spam ("ham") e-mail messages (33.716 e-mails total).
The original dataset and documentation can be found here.

Link: https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data
Source: http://www2.aueb.gr/users/ion/data/enron-spam/

Dataset type: Binary classification
Domain: Cybersecurity / Email phishing detection
Data format: CSV
Target variable: label

COLUMNS: 

Subject:	The subject line of the e-mail
Message:	The content of the e-mail. Can contain an empty string if the message had only a subject line and no body. In case of forwarded emails or replies, this also contains the original message with subject line, "from:", "to:", etc.
Spam/Ham:	Has the values "spam" or "ham". Whether the message was categorized as a spam message or not.
Date: 	The date the e-mail arrived. Has a YYYY-MM-DD format.

Important note: This dataset need preprocessing