import email
import email.policy
import email.parser
import os
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, precision_score, recall_score

def load_data(path):
    emails = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        with open(file_path,'rb') as f:
            mail = email.parser.BytesParser(policy=email.policy.default).parse(f)
        emails.append(mail)
    return emails


def preprocess_email(mail):
        
    mail_payload = mail.get_payload()

    if isinstance(mail_payload, list):
            # If payload is a list, concatenate the parts or handle them individually
            mail_text = " ".join(part.get_payload(decode=True).decode('utf-8', errors='ignore') for part in mail_payload if part.get_payload(decode=True))
    else:
            mail_text = mail_payload
    
    if "<hmtl>" in mail_text or "<body>" in mail_text:
        #mail_text = nltk.clean_html(mail_text)
        soup = BeautifulSoup(mail_text,'html.parser')
        mail_text = soup.get_text()

    mail_text = mail_text.lower()

    mail_text = re.sub(r"http[s]?://\S+", "URL", mail_text)

    mail_text = re.sub(r"\d+", "NUMBER", mail_text)

    mail_text = re.sub(r"[^a-zA-Z\s]", " ", mail_text)
    
    #mail_text = nltk.word_tokenize(mail_text, language='english')

    return mail_text


#declarations
spam =[]
ham = []
path_spam = 'src/Emails/spam'
path_ham = 'src/Emails/easy_ham'
processed_spam = []
processed_ham = []
nltk.download('punkt_tab')

#loading the emails
spam = load_data(path_spam)
ham = load_data(path_ham)

#preprocessing the emails
for spam_mail in spam:
    processed_spam.append(preprocess_email(spam_mail))
for ham_mail in ham:
    processed_ham.append(preprocess_email(ham_mail))


# Aufteilen in Trainings- und Testdaten 

# Labels erstellen: 1 für Spam, 0 für Ham
spam_labels = [1] * len(processed_spam)
ham_labels = [0] * len(processed_ham)

# Daten und Labels zusammenführen
emails = processed_spam + processed_ham
labels = spam_labels + ham_labels

# Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)


#vectorizing the 

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

"""
#training the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
"""


model_nb = MultinomialNB()
model_lr = LogisticRegression(max_iter=500, random_state=42)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a VotingClassifier with Soft Voting
voting_model = VotingClassifier(
    estimators=[
        ('nb', model_nb),
        ('lr', model_lr),
        ('rf', model_rf)
    ],
    voting='soft'  # Use probabilities for voting
)

# Train the VotingClassifier
voting_model.fit(X_train_vectorized, y_train)

# Test the VotingClassifier
y_pred = voting_model.predict(X_test_vectorized)

#testing the model
#y_pred = model.predict(vectorizer.transform(X_test))
print(classification_report(y_test, y_pred))
