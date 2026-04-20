import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5))
X = vectorizer.fit_transform(data['clean_text'])
y = data['language']

model = MultinomialNB()
model.fit(X, y)

st.title("Language Identification System")

user_input = st.text_input("Enter text")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    proba = model.predict_proba(vector)

    st.success("Predicted Language: " + prediction[0])
    st.write("Confidence:", max(proba[0]))