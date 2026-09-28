import pandas as pd
import re

def clean_text(text):
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', str(text))
    text = re.sub(r'\(Reuters\)', '', text)
    return text

# Load both files
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Clean text: remove Reuters dateline tags so the model can't "cheat" on this shortcut
fake_df["text"] = fake_df["text"].apply(clean_text)
true_df["text"] = true_df["text"].apply(clean_text)

# Combine title + body text, since headlines carry strong signal the model was missing
fake_df["text"] = fake_df["title"] + " " + fake_df["text"]
true_df["text"] = true_df["title"] + " " + true_df["text"]

# Add label column: 1 = real, 0 = fake
fake_df["label"] = 0
true_df["label"] = 1

# Combine both into one dataset
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the rows so real/fake aren't grouped together
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Peek at the result
print(combined_df.head())
print("\nTotal articles:", len(combined_df))
print("\nLabel counts:")
print(combined_df["label"].value_counts())

# Save this combined dataset so we don't have to redo this step each time
combined_df.to_csv("data/combined_dataset.csv", index=False)
print("\nSaved combined_dataset.csv to data folder")