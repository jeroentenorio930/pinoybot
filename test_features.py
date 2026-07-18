"""
Rhian's test file, will delete soon also 

test_features.py

Quick sanity tests for extract_hyphen_affix_features, based on the
edge cases worked through in feature design (see report notes).
Run directly: python test_features.py
"""

from features import extract_hyphen_affix_features

test_cases = [
    # (word, expected_features)
    ("i-determine",       {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": True,  "hyphen_root_looks_foreign": True}), 
    # ^ for now this is FAIL when test_features is run because proper identification of
    # english words has not yet been established (looks_foreign() is letter/digraph-based only,
    # and "determine" doesn't trigger either check)
    ("nag-march",         {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": True,  "hyphen_root_looks_foreign": True}),
    ("na-award",          {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": True,  "hyphen_root_looks_foreign": True}),
    ("pina-explain",      {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": True,  "hyphen_root_looks_foreign": True}),
    ("pag-agos",          {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": True, "hyphen_root_looks_foreign": False}),
    ("pag-aaral",         {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": True, "hyphen_root_looks_foreign": False}),
    ("kawalang-kamatayan",{"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": False, "hyphen_root_looks_foreign": False}),
    ("co-actors",         {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": False, "hyphen_root_looks_foreign": False}),
    ("COVID-19",          {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": False, "hyphen_root_looks_foreign": False}),
    ("12-wheeler",        {"hyphen_is_numeral_prefix": True,  "hyphen_affix_exact_match": False, "hyphen_root_looks_foreign": True}),
    ("22-anyos",          {"hyphen_is_numeral_prefix": True,  "hyphen_affix_exact_match": False, "hyphen_root_looks_foreign": False}),
    ("-",                 {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": False, "hyphen_root_looks_foreign": False}),
    ("--",                {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": False, "hyphen_root_looks_foreign": False}),
    ("kita",              {"hyphen_is_numeral_prefix": False, "hyphen_affix_exact_match": False, "hyphen_root_looks_foreign": False}),  # no hyphen at all
]

def run_tests():
    passed = 0
    failed = 0
    for word, expected in test_cases:
        result = extract_hyphen_affix_features(word)
        if result == expected:
            passed += 1
            print(f"PASS: {word!r} -> {result}")
        else:
            failed += 1
            print(f"FAIL: {word!r}")
            print(f"   expected: {expected}")
            print(f"   got:      {result}")
    print(f"\n{passed} passed, {failed} failed out of {len(test_cases)}")

if __name__ == "__main__":
    run_tests()