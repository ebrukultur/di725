import os
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import tiktoken

DATA_CSV = "data/train.csv"
TEST_CSV = "data/test.csv"
OUT_DIR = "data/sentiment"
TEST_SIZE = 0.1
RANDOM_SEED = 42
MAX_LENGTH = 256

os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(txt):
    txt = str(txt).lower()
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt.strip()

# 1) Load main CSV
df = pd.read_csv(DATA_CSV)[["conversation", "customer_sentiment"]].dropna()
df["conversation"] = df["conversation"].apply(clean_text)

# 2) We expect exactly 3 sentiments:
#    negative, neutral, positive
#    Double-check for debugging:
unique_sents = df["customer_sentiment"].unique().tolist()
print("unique_sents in main CSV:", unique_sents)

# Map them to 0,1,2
label_map = {"negative": 0, "neutral": 1, "positive": 2}
# Or if you have unique_sents in a different order, do something like:
# label_map = {sent: i for i, sent in enumerate(sorted(unique_sents))}

df["label"] = df["customer_sentiment"].map(label_map)
train_df, val_df = train_test_split(
    df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED
)

# 3) Load test CSV
test_df = pd.read_csv(TEST_CSV)[["conversation", "customer_sentiment"]].dropna()
test_df["conversation"] = test_df["conversation"].apply(clean_text)
test_df["label"] = test_df["customer_sentiment"].map(label_map)

# 4) tiktoken GPT-2 BPE
enc = tiktoken.get_encoding("gpt2")

def encode_texts(texts):
    return [enc.encode_ordinary(txt)[:MAX_LENGTH] for txt in texts]

train_encoded = encode_texts(train_df["conversation"])
val_encoded   = encode_texts(val_df["conversation"])
test_encoded  = encode_texts(test_df["conversation"])

print("train_label_set:", set(train_df["label"]))
print("val_label_set:", set(val_df["label"]))
print("test_label_set:", set(test_df["label"]))

def save_pickle(fname, enc_convs, labs):
    with open(os.path.join(OUT_DIR, fname), "wb") as f:
        pickle.dump({
            "encoded_conversations": enc_convs,
            "labels": labs
        }, f)

save_pickle("train_data_gpt2.pkl", train_encoded, train_df["label"].tolist())
save_pickle("val_data_gpt2.pkl",   val_encoded,   val_df["label"].tolist())
save_pickle("test_data_gpt2.pkl",  test_encoded,  test_df["label"].tolist())

meta = {
    "vocab_size": enc.n_vocab,  # ~50257
    "num_classes": 3,
    "labels": ["negative", "neutral", "positive"]
}
with open(os.path.join(OUT_DIR, "meta_gpt2.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Preprocessing complete! Data saved to", OUT_DIR)
