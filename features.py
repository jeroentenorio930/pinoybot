import re
import string
from typing import List, Dict, Any

FILIPINO_AFFIXES = {
    "nag", "na", "pina", "pa", "pag", "mag", "ipag", "ka", "i", "um", "in", "an", "ma", 
    "naka", "nakaka", "mapa", "pinaka", "kaka", "kina", "ni",
    "paki", "ipaki", "magpa", "nagpa", "pagpapa", "pinapa", "nagpapa", "magpapa", 
    "mapapa", "napapa",
    "maka", "makapag", "nakapag",
    "magka", "nagka", "pagka", "pagkaka", 
    "nagkaka", "magkaka",
    "pinag", "pang", "mang", "nang", "ipang", "ipinang", "isa", "isina",
    "pinakama", "ipinaka", "mala",
    "tig", "taga", "tagapag", "tag", 
    "naki", 
    "makipag", "nakipag", "pakikipag",  
    "hin", "han"
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

def get_max_consecutive_consonants(word: str) -> int:
    """
    this counts the maximum number of consecutive consonants in a word.
    """
    max_count = 0
    current_count = 0
    for char in word.lower():
        if char.isalpha() and char not in "aeiou":
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count

def extract_fused_affix_features(word: str) -> Dict[str, Any]:
    """
    this is to detects the intra-word code-switching without hyphens (e.g., 'ipost', 'naglunch').
    """
    features = {
        "has_fused_filipino_affix": False,
        "fused_root_looks_foreign": False
    }
    
    # If it has a hyphen, Rhian's function already handles it
    if "-" in word:
        return features
        
    word_lower = word.lower()
    longest_affix = ""
    
    # Find the longest matching prefix
    for affix in FILIPINO_AFFIXES:
        if word_lower.startswith(affix):
            if len(word_lower) >= len(affix) + 3:
                if len(affix) > len(longest_affix):
                    longest_affix = affix
                    
    if longest_affix:
        features["has_fused_filipino_affix"] = True
        root = word_lower[len(longest_affix):]
        
        if looks_foreign(root) or get_max_consecutive_consonants(root) >= 2:
            features["fused_root_looks_foreign"] = True
            
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

    # detects abbrevs
    features["has_no_vowels"] = (vowel_count == 0 and word_length > 0 and word.isalpha())

    # consonant clusers
    features["max_consecutive_consonants"] = get_max_consecutive_consonants(word)
    features["ends_with_consonant_cluster"] = False
    if word_length >= 2:
        last_two = word_lower[-2:]
        if last_two[0].isalpha() and last_two[1].isalpha():
            if last_two[0] not in vowels and last_two[1] not in vowels:
                features["ends_with_consonant_cluster"] = True

    # check if naulit yung syllable (idk how to call it)
    # features["has_2char_reduplication"] = (word_length >= 4 and word_lower[:2] == word_lower[2:4])
    # features["has_3char_reduplication"] = (word_length >= 6 and word_lower[:3] == word_lower[3:6])

    return features

def extract_context(tokens: List[str], index: int) -> Dict[str, Any]:
    """
    this extracts features based on the surrounding environment of the target word 
    (previous and next words) to capture context clues.
    """
    features = {}
    word = tokens[index]

    # sentence end flags
    features["is_start_of_sentence"] = (index == 0)
    features["is_end_of_sentence"] = (index == len(tokens) - 1)

    features["is_capitalized_mid_sentence"] = (word.istitle() and not features["is_start_of_sentence"])
    
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

# start of drea

def extract_symbols_and_numbers(word:str) -> Dict[str,Any]:
    """
    this extracts features that detect whether a word contains numbers or symbols.
    it ignores the hyphen, except for when the token is a single-character symbol.
    """
    features = {
        "is_symbol": False
    }

    # check if the word is a single-character symbol 
    if len(word) == 1 and word in string.punctuation:
        features["is_symbol"] = True

    # check for any numbers
    features["contains_number"] = bool(re.search(r'\d', word))

    # check for characters that are not letters, numbers, or hyphen
    features["contains_symbol"] = bool(re.search(r'[^a-zA-Z0-9\-]', word))

    return features

def extract_reduplication(word_lower:str) -> Dict[str, Any]:
    """
    this extracts features that have filipino reduplication by checking
    the first four letters of the word to see whether it follows the 
    CVCV structure for digrams and CCVCCV for trigrams.
    """

    features = {
        "has_digram_reduplication": False,
        "has_trigram_reduplication": False
    }

    if len(word_lower) >= 6:
        first_digram = word_lower[:2]
        second_digram = word_lower[2:4]

        # check if digrams are the same and if the first one follows CV structure
        if first_digram == second_digram and re.match(r'[^aeiou][aeiou]', first_digram):
            features["has_digram_reduplication"] = True
        
        first_trigram = word_lower[:3]
        second_trigram = word_lower[3:6]

        # check if trigrams are the same and if the first one follows CCV structure
        if first_trigram == second_trigram and re.match(r'[^aeiou][^aeiou][aeiou]', first_trigram):
            features["has_trigram_reduplication"] = True

    return features

# end of drea


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
    features.update(extract_fused_affix_features(word))
    features.update(extract_ngram(word))
    features.update(extract_context(tokens, index))
    features.update(extract_symbols_and_numbers(word))
    features.update(extract_reduplication(word_lower))

    features["word_lower"] = word_lower

    return features

def extract_dataset_features(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Transforms a sequence of tokens into a list of feature dictionaries.
    """
    return [extract_word_features(tokens, i) for i in range(len(tokens))]

