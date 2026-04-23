import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("dataset_full.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data['clean_text'] = data['text'].apply(clean_text)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
X = vectorizer.fit_transform(data['clean_text'])
y = data['language']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Naive Bayes Report")
print(classification_report(y_test, y_pred_nb))

print("Logistic Regression Report")
print(classification_report(y_test, y_pred_lr))

cm = confusion_matrix(y_test, y_pred_lr)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()