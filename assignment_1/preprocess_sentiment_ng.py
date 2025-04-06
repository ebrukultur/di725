import os
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# File paths and constants
DATA_CSV = "data/train.csv"   # main training data
TEST_CSV = "data/test.csv"    # separate test data
OUT_DIR = "data/sentiment"    # where processed files will be saved
TEST_SIZE = 0.1               # percentage for validation split
RANDOM_SEED = 42              # reproducibility

# Make sure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Load and prepare main dataset
df = pd.read_csv(DATA_CSV)[["conversation", "customer_sentiment"]].dropna()

# Basic text cleaning
def clean_text(txt):
    txt = str(txt).lower()                         # lowercase all text
    txt = re.sub(r"http\S+", "", txt)              # remove URLs
    txt = re.sub(r"[^\w\s]", "", txt)              # remove punctuation
    return txt.strip()

# Apply cleaning to conversation column
df["conversation"] = df["conversation"].apply(clean_text)

# Label encoding
# Get all unique sentiment classes (e.g. ['neutral', 'negative', 'positive'])
unique_sents = df["customer_sentiment"].unique().tolist()

# Map each sentiment to a unique integer (e.g. {'neutral': 0, 'negative': 1, ...})
label2id = {lab: i for i, lab in enumerate(unique_sents)}
df["label"] = df["customer_sentiment"].map(label2id)

# Train-validation split (stratified by sentiment class)
train_df, val_df = train_test_split(
    df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED
)

# Load and prepare test dataset
test_df = pd.read_csv(TEST_CSV)[["conversation", "customer_sentiment"]].dropna()
test_df["conversation"] = test_df["conversation"].apply(clean_text)
test_df["label"] = test_df["customer_sentiment"].map(label2id)

# Build character-level vocabulary from training set
all_train_text = " ".join(train_df["conversation"].tolist())
chars = sorted(list(set(all_train_text)))  # sorted list of all unique chars
vocab_size = len(chars)                    # e.g. ~60–100 characters
stoi = {ch: i for i, ch in enumerate(chars)}  # char → index
itos = {i: ch for ch, i in stoi.items()}      # index → char

print("vocab_size:", vocab_size)

# Text encoding: convert conversation to sequence of indices
def encode_text(text):
    return [stoi.get(c, 0) for c in text]  # use 0 if char is unknown

# Encode training set
train_encoded, train_labels = [], []
for _, row in train_df.iterrows():
    train_encoded.append(encode_text(row["conversation"]))
    train_labels.append(row["label"])

# Encode validation set
val_encoded, val_labels = [], []
for _, row in val_df.iterrows():
    val_encoded.append(encode_text(row["conversation"]))
    val_labels.append(row["label"])

# Encode test set
test_encoded, test_labels = [], []
for _, row in test_df.iterrows():
    test_encoded.append(encode_text(row["conversation"]))
    test_labels.append(row["label"])

# Save encoded data as .pkl
train_data = {"encoded_conversations": train_encoded, "labels": train_labels}
val_data   = {"encoded_conversations": val_encoded,   "labels": val_labels}
test_data  = {"encoded_conversations": test_encoded,  "labels": test_labels}

with open(os.path.join(OUT_DIR, "train_data.pkl"), "wb") as f:
    pickle.dump(train_data, f)
with open(os.path.join(OUT_DIR, "val_data.pkl"), "wb") as f:
    pickle.dump(val_data, f)
with open(os.path.join(OUT_DIR, "test_data.pkl"), "wb") as f:
    pickle.dump(test_data, f)

# Save metadata (vocab, label info)
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
    "num_classes": len(unique_sents),
    "labels": unique_sents  # original sentiment labels in order
}
with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Preprocessing complete! Data saved to", OUT_DIR)
