"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List

from features import extract_dataset_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.pkl")
 
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
    
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)
    _load_model()

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)
    feature_dicts = extract_dataset_features(tokens)

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)
    X = _vectorizer.transform(feature_dicts)
    #if _model_name == "naive_bayes":
    #    X = X.toarray()

    predicted = _model.predict(X)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]
    tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    return tags

if __name__ == "__main__":
    # Example usage
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags:  ", tags)