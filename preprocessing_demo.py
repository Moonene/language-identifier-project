import pandas as pd
import re

data = pd.read_csv("dataset_full.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize(text):
    return text.split()

slang_dict = {
    "niko": "niko",
    "base": "nyumba",
    "wasee": "watu",
    "fiti": "nzuri",
    "tao": "town"
}

def normalize_sheng(text):
    words = text.split()
    normalized = [slang_dict.get(word, word) for word in words]
    return " ".join(normalized)

data['clean_text'] = data['text'].apply(clean_text)
data['tokens'] = data['clean_text'].apply(tokenize)
data['normalized'] = data['clean_text'].apply(normalize_sheng)

print(data[['text', 'clean_text', 'tokens', 'normalized']].head(10))