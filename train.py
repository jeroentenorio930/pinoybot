"""
train.py

Trains and pickles the PinoyBot language-identification model.

Pipeline:
  1. Load annotated dataset (Rhian)          -- load_annotated_data()
  2. Feature extraction (features.py)        -- extract_dataset_features()
  3. Train/val/test split (Rhian)            -- split_dataset()
  4. Vectorization (Xan)                     -- DictVectorizer
  5. Model training/tuning (Xan)             -- Decision Tree vs Naive Bayes
  6. Validation performance check (Xan)
  7. Save model + vectorizer to disk (Lance)

NOTE: dataset_annotation.csv must already be cleaned via compile_dataset.py
(run separately, once) into annotated_dataset_clean.csv before this script
is run. See compile_dataset.py for details on what cleaning is applied.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# from features import extract_dataset_features  # uncomment once features.py is finalized


# load dataset (rhian)
def load_annotated_data(csv_path: str):
    """
    Loads the clean, merged annotation CSV (produced by compile_dataset.py)
    and groups it into sentences.

    Returns:
        sentences: List[List[str]]  -- tokens per sentence
        tags:      List[List[str]]  -- tag per token, same shape as sentences
    """
    # keep_default_na=False / na_filter=False: without this, pandas silently
    # re-interprets the literal placeholder string "nan" (used for the 23
    # rows with originally-missing word text) as a real missing value again
    # on read, undoing the fix made in compile_dataset.py. There are no
    # genuinely-missing cells left in the clean CSV, so this is safe.
    df = pd.read_csv(csv_path, keep_default_na=False, na_filter=False)
    df["tag"] = df["tag"].astype(str).str.strip().str.upper()

    sentences, tags = [], []
    for _, group in df.sort_values("token_id").groupby("sentence_id", sort=True):
        sentences.append(group["token"].tolist())
        tags.append(group["tag"].tolist())

    return sentences, tags


# 3. train/validation/test split (rhian)

def split_dataset(sentences, tags, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """
    Splits at the SENTENCE level (never token level) into 70/15/15
    train/val/test, so context-dependent features (prev/next word, char
    n-grams) never leak information across splits.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, "splits must sum to 1.0"

    sent_train, sent_temp, tag_train, tag_temp = train_test_split(
        sentences, tags, train_size=train_size, random_state=random_state
    )
    relative_val = val_size / (val_size + test_size)
    sent_val, sent_test, tag_val, tag_test = train_test_split(
        sent_temp, tag_temp, train_size=relative_val, random_state=random_state
    )

    return {
        "train": (sent_train, tag_train),
        "val":   (sent_val, tag_val),
        "test":  (sent_test, tag_test),
    }


# main training entry point
def train_and_pickle():
    # --- 1. Load annotated dataset (Rhian) ---
    sentences, tags = load_annotated_data("annotated_dataset_clean.csv")

    # --- 3. Train/val/test split (Rhian) ---
    splits = split_dataset(sentences, tags)
    train_sents, train_tags = splits["train"]
    val_sents, val_tags     = splits["val"]
    test_sents, test_tags   = splits["test"]

    # --- 2. Convert raw sentences into feature dictionaries (uses features.py) ---
    # Example of how this should look once features.py is finalized:
    #
    #   def flatten(sents, tag_lists):
    #       X, y = [], []
    #       for tokens, sent_tags in zip(sents, tag_lists):
    #           X.extend(extract_dataset_features(tokens))
    #           y.extend(sent_tags)
    #       return X, y
    #
    #   X_train, y_train = flatten(train_sents, train_tags)
    #   X_val,   y_val   = flatten(val_sents, val_tags)
    #   X_test,  y_test  = flatten(test_sents, test_tags)

    # --- 4. Vectorize features (Xan) ---
    #   from sklearn.feature_extraction import DictVectorizer
    #   vec = DictVectorizer()
    #   X_train_vec = vec.fit_transform(X_train)
    #   X_val_vec   = vec.transform(X_val)
    #   X_test_vec  = vec.transform(X_test)

    # --- 5. Initialize and train scikit-learn model (Xan) ---
    #   Decision Tree vs Naive Bayes, pick best

    # --- 6. Performance validation check (Xan) ---
    #   evaluate on X_val_vec / y_val

    # --- 7. Save model and vectorizer together to disk (Lance) ---
    #   with open("trained_model.pkl", "wb") as f:
    #       pickle.dump({"model": model, "vectorizer": vec}, f)

    pass


if __name__ == "__main__":
    train_and_pickle()