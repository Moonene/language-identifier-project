import pandas as pd
import random

data = pd.read_csv("dataset.csv")

expanded = []

for i in range(10):
    for _, row in data.iterrows():
        text = row['text']
        if random.random() > 0.5:
            text = text + " sana"
        if random.random() > 0.5:
            text = text.replace("leo", "sasa")
        expanded.append([text, row['language']])

df = pd.DataFrame(expanded, columns=["text", "language"])
df.to_csv("dataset_full.csv", index=False)

print(len(df))