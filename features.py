import re
from typing import List, Dict, Any

def extract_word_features(tokens: List[str], index: int) -> Dict[str, Any]:
    """
    Extracts linguistic features for a token at a specific index.
    """
    word = tokens[index]
    word_lower = word.lower()

    # TO DO: Insert feature detection logic

    return {
        # return values here
    }

def extract_dataset_features(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Transforms a sequence of tokens into a list of feature dictionaries.
    """
    return [extract_word_features(tokens, i) for i in range(len(tokens))]

