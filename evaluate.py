"""
evaluate.py

Evaluates the trained model using the same 70/15/15 sentence-level split
created in train.py. Since both files use random_state=42, the test set
generated here is identical to the one the model never saw during training
or validation.
"""

import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from features import extract_dataset_features
from train import load_annotated_data, split_dataset, flatten, MODEL_PATH

LABELS = ["ENG", "FIL", "CS", "OTH"]

def load_model_bundle(model_path: str = MODEL_PATH):
    """Loads the trained model, vectorizer, and model name from disk."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["vectorizer"], bundle["model_name"]


def get_test_set(csv_path: str = "annotated_dataset_clean.csv"): # for testing, use tiny_dataset.csv and refer to instructions in test_metrics.py
    """Reproduces the same test split used in train.py."""
    sentences, tags = load_annotated_data(csv_path) 
    splits = split_dataset(sentences, tags) # Generate the same dataset split
    test_sents, test_tags = splits["test"]
    
    X_test, y_test = flatten(test_sents, test_tags) # Convert sentences into token-level samples
    return X_test, y_test


def evaluate(model_path: str = MODEL_PATH):
    """Evaluates the trained model on the held-out test set."""
    print("Loading model...")
    model, vec, model_name = load_model_bundle(model_path)
    print(f"Loaded model_name='{model_name}' (from {model_path})")

    print("Rebuilding test set (same split as train.py, random_state=42)...")
    X_test, y_test = get_test_set()

    X_test_vec = vec.transform(X_test) # Convert test tokens into feature vectors
    y_pred = model.predict(X_test_vec) # Generate predictions

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=LABELS, average=None, zero_division=0
    )

    print("\n=== Test Set Performance ===")
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Test set size: {len(y_test)} tokens\n")

    print(f"{'Label':6s}{'Precision':>12s}{'Recall':>10s}{'F1-score':>10s}{'Support':>10s}")
    for label, p, r, f, s in zip(LABELS, precision, recall, f1, support):
        print(f"{label:6s}{p:12.4f}{r:10.4f}{f:10.4f}{int(s):10d}")

    print("\n--- Full classification report (incl. macro/weighted avg) ---")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    print("--- Confusion matrix (rows=true, cols=predicted) ---")
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    header = "        " + "".join(f"{l:>8s}" for l in LABELS)
    print(header)
    for label, row in zip(LABELS, cm):
        print(f"{label:8s}" + "".join(f"{v:8d}" for v in row))

    return {
        "accuracy": acc,
        "per_label": dict(zip(LABELS, zip(precision, recall, f1, support))),
        "confusion_matrix": cm,
    }

if __name__ == "__main__":
    evaluate()