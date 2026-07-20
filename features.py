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

# This is JR
def extract_ngram (word: str) -> Dict[str, Any]:
    """
    this extracts internal structure features of the word, including character n-grams 
    and word properties like capitalization and vowel ratios.
    """
    word_lower = word.lower()
    word_length = len(word)
    
    features = {}

    # n-grams (prefixes and suffixes)
    features["prefix_2"] = word_lower[:2] if word_length >= 2 else word_lower
    features["prefix_3"] = word_lower[:3] if word_length >= 3 else word_lower
    
    features["suffix_2"] = word_lower[-2:] if word_length >= 2 else word_lower
    features["suffix_3"] = word_lower[-3:] if word_length >= 3 else word_lower

    # word properties
    features["is_capitalized"] = word.istitle()
    features["is_all_uppercase"] = word.isupper()
    
    # check language pattern
    vowels = set("aeiou")
    vowel_count = sum(1 for char in word_lower if char in vowels)
    features["vowel_ratio"] = vowel_count / word_length if word_length > 0 else 0.0

    # check if naulit yung syllable (idk how to call it)
    features["has_2char_reduplication"] = (word_length >= 4 and word_lower[:2] == word_lower[2:4])
    features["has_3char_reduplication"] = (word_length >= 6 and word_lower[:3] == word_lower[3:6])

    return features

def extract_context(tokens: List[str], index: int) -> Dict[str, Any]:
    """
    this extracts features based on the surrounding environment of the target word 
    (previous and next words) to capture context clues.
    """
    features = {}
    
    # sentence end flags
    features["is_start_of_sentence"] = (index == 0)
    features["is_end_of_sentence"] = (index == len(tokens) - 1)

    # previous word
    if index > 0:
        prev_word = tokens[index - 1]
        
        features["prev_word_lower"] = prev_word.lower()
        
        # grammatical context: does the previous word end in a vowel? 
        features["prev_word_ends_vowel"] = prev_word.lower()[-1] in "aeiou" if prev_word else False
        features["prev_word_is_capitalized"] = prev_word.istitle()
    else:
        # padding
        features["prev_word_lower"] = "<START>"
        features["prev_word_ends_vowel"] = False
        features["prev_word_is_capitalized"] = False

    # next word
    if index < len(tokens) - 1:
        next_word = tokens[index + 1]
        
        features["next_word_lower"] = next_word.lower()
        features["next_word_is_capitalized"] = next_word.istitle()
    else:

        features["next_word_lower"] = "<END>"
        features["next_word_is_capitalized"] = False

    return features

# End of JR

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
    features.update(extract_ngram(word))
    features.update(extract_context(tokens, index))

    return features

def extract_dataset_features(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Transforms a sequence of tokens into a list of feature dictionaries.
    """
    return [extract_word_features(tokens, i) for i in range(len(tokens))]

