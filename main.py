import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("dataset_full.csv")

def normalize_sheng(text):
    slang_dict = {
        "niko": "niko",
        "base": "nyumbani",
        "wasee": "watu",
        "mabeshte": "marafiki",
        "rada": "tayari",
        "tao": "mjini",
        "keja": "nyumba",
        "fiti": "nzuri",
        "poa": "nzuri",
        "bro": "ndugu"
    }
    words = text.split()
    normalized = [slang_dict.get(word.lower(), word) for word in words]
    return " ".join(normalized)

def clean_text(text):
    text = text.lower()
    text = normalize_sheng(text)
    text = text.replace("ĩ","i").replace("ũ","u")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data['clean_text'] = data['text'].apply(clean_text)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
X = vectorizer.fit_transform(data['clean_text'])
y = data['language']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print("Naive Bayes Report")
print(classification_report(y_test, nb_pred))
print("Naive Bayes Confusion Matrix")
print(confusion_matrix(y_test, nb_pred))

print("Logistic Regression Report")
print(classification_report(y_test, lr_pred))
print("Logistic Regression Confusion Matrix")
print(confusion_matrix(y_test, lr_pred))