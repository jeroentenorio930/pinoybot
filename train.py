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
is run. See compile_dataset.py for details on what cleaning is applied,
including the decision to DROP rows with missing token text rather than
placeholder-fill them.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
    df = pd.read_csv(csv_path)
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

# flattens sentence-level (tokens, tags) into flat per-token
def flatten(sents, tag_lists):
    X, y = [], []
    for tokens, sent_tags in zip(sents, tag_lists):
        X.extend(extract_dataset_features(tokens))
        y.extend(sent_tags)
    return X, y


# 4-6. vectorization + model training/tuning + validation check (xan)

def vectorize_features(X_train, X_val, X_test):
    """
    Fits a DictVectorizer on the training feature dicts and transforms
    train/val/test with it. Fitting only on train avoids leaking
    val/test vocabulary into the feature space.
    """
    vec = DictVectorizer(sparse=True)
    X_train_vec = vec.fit_transform(X_train)
    X_val_vec = vec.transform(X_val)
    X_test_vec = vec.transform(X_test)
    return vec, X_train_vec, X_val_vec, X_test_vec


def train_and_select_model(X_train_vec, y_train, X_val_vec, y_val, verbose=True):
    """
    Trains a Decision Tree and a Multinomial Naive Bayes classifier on the
    training set, evaluates both on the validation set, and returns
    whichever scores higher (by macro F1, since tag classes are likely
    imbalanced), along with a small report of both candidates' scores.

    Decision Tree is lightly tuned over max_depth via the validation set
    (cheap grid search, no separate CV since we already have a val split).
    """
    candidates = {}

    # -- Decision Tree, tuned over max_depth on the val set --
    best_dt, best_dt_depth, best_dt_f1 = None, None, -1.0
    for depth in [None, 5, 10, 20, 30, 50]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train_vec, y_train)
        val_pred = dt.predict(X_val_vec)
        f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if f1 > best_dt_f1:
            best_dt, best_dt_depth, best_dt_f1 = dt, depth, f1
    candidates["decision_tree"] = {
        "model": best_dt,
        "val_f1_macro": best_dt_f1,
        "val_accuracy": accuracy_score(y_val, best_dt.predict(X_val_vec)),
        "params": {"max_depth": best_dt_depth},
    }

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_vec.toarray(), y_train)
    nb_val_pred = nb.predict(X_val_vec.toarray())
    candidates["naive_bayes"] = {
        "model": nb,
        "val_f1_macro": f1_score(y_val, nb_val_pred, average="macro", zero_division=0),
        "val_accuracy": accuracy_score(y_val, nb_val_pred),
        "params": {},
    }

    # pick the winner by macro F1 on validation
    best_name = max(candidates, key=lambda name: candidates[name]["val_f1_macro"])
    best = candidates[best_name]

    if verbose:
        print("=== Validation performance ===")
        for name, info in candidates.items():
            marker = " <-- selected" if name == best_name else ""
            print(
                f"{name:15s} val_acc={info['val_accuracy']:.4f} "
                f"val_f1_macro={info['val_f1_macro']:.4f} "
                f"params={info['params']}{marker}"
            )
        print()
        print(f"Selected model: {best_name}")
        print(classification_report(y_val, best["model"].predict(X_val_vec), zero_division=0))

    return best_name, best["model"], candidates

# main training entry point
def train_and_pickle():
    # --- 1. Load annotated dataset (Rhian) ---
    sentences, tags = load_annotated_data("annotated_dataset_clean.csv")

    # --- 3. Train/val/test split (Rhian) ---
    splits = split_dataset(sentences, tags)
    train_sents, train_tags = splits["train"]
    val_sents, val_tags     = splits["val"]
    test_sents, test_tags   = splits["test"]

    # --- 2. Convert raw sentences into feature dictionaries (features.py) ---
    X_train, y_train = flatten(train_sents, train_tags)
    X_val,   y_val   = flatten(val_sents, val_tags)
    X_test,  y_test  = flatten(test_sents, test_tags)

    # --- 4. Vectorize features (Xan) ---
    vec, X_train_vec, X_val_vec, X_test_vec = vectorize_features(X_train, X_val, X_test)

    # --- 5. Train + tune Decision Tree vs Naive Bayes, pick best (Xan) ---
    # --- 6. Validation performance check (Xan) ---
    best_name, model, candidates = train_and_select_model(
        X_train_vec, y_train, X_val_vec, y_val
    )

    # held-out test set report, for reference (final unbiased number)
    X_test_for_model = X_test_vec.toarray() if best_name == "naive_bayes" else X_test_vec
    test_pred = model.predict(X_test_for_model)
    print("=== Test performance (held out, unbiased) ===")
    print(f"accuracy={accuracy_score(y_test, test_pred):.4f} "
          f"f1_macro={f1_score(y_test, test_pred, average='macro', zero_division=0):.4f}")

    # --- 7. Save model and vectorizer together to disk (Lance) ---
    #   with open("trained_model.pkl", "wb") as f:
    #       pickle.dump({"model": model, "vectorizer": vec}, f)

    return best_name, model, vec


if __name__ == "__main__":
    train_and_pickle()