"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction are implemented separately (see
features.py and train.py). The trained model + vectorizer are pickled
together by train.py (step 7) as:

    {"model": ..., "vectorizer": ..., "model_name": "decision_tree" | "naive_bayes"}

This module loads that pickle and uses it here for prediction.
"""

import os
import re
import pickle
from typing import List

from features import extract_dataset_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.pkl")

# Same tokenizer used in test_cli.py/tester.py -- splits off punctuation as
# its own token, keeps hyphenated words together (e.g. "nag-march").
# This is NOT the same tokenizer used to build the dataset -- it's just
# good enough for the quick manual demo below. tag_language() itself does
# NOT tokenize anything -- it always expects an already-tokenized list,
# per the project spec.
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*|[^\sA-Za-z0-9]")


def _tokenize(sentence: str):
    return _TOKEN_PATTERN.findall(sentence)

# Loaded once at import time and reused across calls, so repeated calls to
# tag_language() don't re-read the pickle from disk every time.
_model = None
_vectorizer = None
_model_name = None


def _load_model():
    """Loads the pickled model bundle from disk (once, lazily)."""
    global _model, _vectorizer, _model_name
    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find '{MODEL_PATH}'. Run train.py first to train "
            f"and save the model before calling tag_language()."
        )

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    _model = bundle["model"]
    _vectorizer = bundle["vectorizer"]
    _model_name = bundle["model_name"]


# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
    """
    if not tokens:
        return []

    # 1. Load the trained model + vectorizer (lazy, cached after first call)
    _load_model()

    # 2. Extract features for every token, using the SAME feature
    #    extraction function used during training, so the feature space
    #    matches exactly.
    feature_dicts = extract_dataset_features(tokens)

    # 3. Vectorize using the vectorizer fitted during training (do NOT
    #    re-fit here -- must reuse the same fitted vocabulary/columns).
    X = _vectorizer.transform(feature_dicts)

    # NOTE: both Decision Tree and MultinomialNB accept the sparse matrix
    # directly, so no dense conversion is needed here.

    # 4. Predict tags
    predicted = _model.predict(X)

    # 5. Convert predictions to plain strings and return
    tags = [str(tag) for tag in predicted]
    return tags


if __name__ == "__main__":
    # Simple interactive demo. Uses the same punctuation-aware tokenizer as
    # test_cli.py/tester.py, so e.g. "kita." correctly splits into "kita"
    # and "." as two separate tokens instead of staying glued together.
    print("PinoyBot quick demo. Type a sentence and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            sentence = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not sentence:
            continue
        if sentence.lower() in ("quit", "exit"):
            print("Bye!")
            break

        tokens = _tokenize(sentence)
        if not tokens:
            print("(no tokens found)")
            continue

        tags = tag_language(tokens)

        if len(tags) != len(tokens):
            print(f"WARNING: got {len(tags)} tags for {len(tokens)} tokens -- "
                  f"tag_language() should return one tag per token.")

        for tok, tag in zip(tokens, tags):
            print(f"  {tok!r:20s} -> {tag}")
        print()