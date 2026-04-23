import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

data = pd.read_csv("dataset_full.csv")

bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(data['text'])
print("BoW Shape:", X_bow.shape)
print(bow_vectorizer.get_feature_names_out()[:20])

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(data['text'])
print("\nTF-IDF Shape:", X_tfidf.shape)
print(tfidf_vectorizer.get_feature_names_out()[:20])

ngram_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
X_ngram = ngram_vectorizer.fit_transform(data['text'])
print("\nN-grams Shape:", X_ngram.shape)
print(ngram_vectorizer.get_feature_names_out()[:20])