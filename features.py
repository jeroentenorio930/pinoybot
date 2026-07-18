import re
from typing import List, Dict, Any

FILIPINO_AFFIXES = {
    "nag", "na", "pina", "pa", "pag", "mag", "ipag", "ka", "i",
    "um", "in", "an", "ma", "naka", "nakaka", "mapa", "pinaka"
    # you guys can extend the list pa if may iba kayong naiisip
}

FOREIGN_LETTERS = set("cfjqvxzw")
ENGLISH_DIGRAPHS = {"th", "ph", "wh", "ck", "sq", "cl"}

def looks_foreign(word: str) -> bool:
    word = word.lower()
    if not word:
        return False
    if any(ch in FOREIGN_LETTERS for ch in word):
        return True
    if any(word.startswith(digraph) or digraph in word for digraph in ENGLISH_DIGRAPHS):
        return True
    return False

def extract_hyphen_affix_features(word: str) -> Dict[str, Any]:
    features = {
        "hyphen_is_numeral_prefix": False,
        "hyphen_affix_exact_match": False,
        "hyphen_root_looks_foreign": False,
    }

    if "-" not in word:
        return features
    
    parts = word.split("-", 1) #we split the prefix and the rest of the word (only first hyphen)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return features

    prefix, root = parts[0].lower(), parts[1]

    # step 0 - numerical prefix ("12-wheeler", "22-anyos")
    if prefix.isdigit():
        features["hyphen_is_numeral_prefix"] = True
        features["hyphen_root_looks_foreign"] = looks_foreign(root)
        return features
    
    # step 1 - exact affix match (not substring or startswith)
    if prefix in FILIPINO_AFFIXES:
        features["hyphen_affix_exact_match"] = True

        #step 2 - only meaningful once step 1 holds
        features["hyphen_root_looks_foreign"] = looks_foreign(root)

    return features

def extract_word_features(tokens: List[str], index: int) -> Dict[str, Any]:
    """
    Extracts linguistic features for a token at a specific index.
    """
    word = tokens[index]
    word_lower = word.lower()

    # TO DO: Insert feature detection logic
    # other members' features get added here via features.update(...) calls, same pattern

    features = {}

    features.update(extract_hyphen_affix_features(word))

    return features

def extract_dataset_features(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Transforms a sequence of tokens into a list of feature dictionaries.
    """
    return [extract_word_features(tokens, i) for i in range(len(tokens))]

