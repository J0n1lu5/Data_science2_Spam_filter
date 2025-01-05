import email
import email.policy
import email.parser
import os
import re
import nltk
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report

# Constants
PATH_SPAM = 'src/Emails/spam'
PATH_HAM = 'src/Emails/easy_ham'

# Functions
def load_data(path):
    """
    Load email data from the specified directory.
    
    Args:
        path (str): Path to the directory containing email files.
        
    Returns:
        list: List of parsed email objects.
    """
    emails = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as f:
            mail = email.parser.BytesParser(policy=email.policy.default).parse(f)
        emails.append(mail)
    return emails

def preprocess_email(mail):
    """
    Preprocess an email by extracting and cleaning its content.
    
    Args:
        mail (email.message.EmailMessage): Parsed email object.
        
    Returns:
        str: Cleaned email text.
    """
    mail_payload = mail.get_payload()

    if isinstance(mail_payload, list):
        mail_text = " ".join(
            part.get_payload(decode=True).decode('utf-8', errors='ignore')
            for part in mail_payload if part.get_payload(decode=True)
        )
    else:
        mail_text = mail_payload or ""

    if "<html>" in mail_text or "<body>" in mail_text:
        soup = BeautifulSoup(mail_text, 'html.parser')
        mail_text = soup.get_text()

    mail_text = mail_text.lower()
    mail_text = re.sub(r"http[s]?://\S+", "URL", mail_text)
    mail_text = re.sub(r"\d+", "NUMBER", mail_text)
    mail_text = re.sub(r"[^a-zA-Z\s]", " ", mail_text)

    return mail_text

# Main script
if __name__ == "__main__":
    # Download required NLTK data
    nltk.download('punkt_tab')

    # Load and preprocess emails
    spam = load_data(PATH_SPAM)
    ham = load_data(PATH_HAM)
    
    processed_spam = [preprocess_email(mail) for mail in spam]
    processed_ham = [preprocess_email(mail) for mail in ham]

    # Create labels: 1 for spam, 0 for ham
    spam_labels = [1] * len(processed_spam)
    ham_labels = [0] * len(processed_ham)

    # Combine data and labels
    emails = processed_spam + processed_ham
    labels = spam_labels + ham_labels

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Initialize classifiers
    model_nb = MultinomialNB()
    model_lr = LogisticRegression(max_iter=500, random_state=42)
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Create a VotingClassifier with soft voting
    voting_model = VotingClassifier(
        estimators=[('nb', model_nb), ('lr', model_lr), ('rf', model_rf)],
        voting='soft'
    )

    # Train and test the VotingClassifier
    voting_model.fit(X_train_vectorized, y_train)
    y_pred = voting_model.predict(X_test_vectorized)

    # Output classification report
    print(classification_report(y_test, y_pred))
