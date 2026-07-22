"""
test_pinoybot.py

Tests pinoybot.py's tag_language() function specifically -- not the model's
accuracy (that's evaluate.py's job), but whether the INTEGRATION code
(loading, feature extraction, vectorizing, predicting, formatting output)
is correct and won't crash or misbehave when the automated checker calls it.

Three kinds of checks:

  CHECK 1: Output contract -- for various inputs, does tag_language()
           return a list of the same length as the input, containing
           only valid tags (ENG/FIL/CS/OTH)?

  CHECK 2: Edge cases -- empty list, single token, punctuation-only,
           very long input, tokens with unusual characters. Does it
           crash on any of these?

  CHECK 3: Consistency with the "raw" path -- pinoybot.py re-implements
           (load model -> extract features -> vectorize -> predict) by
           hand. This check runs the SAME steps independently (bypassing
           tag_language entirely) on real test-set sentences, and
           confirms tag_language() gives IDENTICAL output. This proves
           pinoybot.py's glue code isn't silently doing something
           different from what train.py/evaluate.py assume.
"""

import pickle

from pinoybot import tag_language
from features import extract_dataset_features
from train import load_annotated_data, split_dataset

VALID_TAGS = {"ENG", "FIL", "CS", "OTH"}


def check_output_contract():
    print("=== CHECK 1: output contract (length + valid tags) ===")
    test_inputs = [
        ["Love", "kita", "."],
        ["Ang", "ganda", "ng", "araw", "!"],
        ["nag-shopping", "kami", "kahapon"],
        ["a"],
        ["#BringHimHome"],
        ["67", "2:00", "AM"],
    ]
    passed = 0
    failed = 0
    for tokens in test_inputs:
        tags = tag_language(tokens)
        len_ok = len(tags) == len(tokens)
        tags_ok = all(t in VALID_TAGS for t in tags)
        ok = len_ok and tags_ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: tag_language({tokens!r}) -> {tags!r}")
        if not len_ok:
            print(f"      length mismatch: got {len(tags)} tags for {len(tokens)} tokens")
        if not tags_ok:
            bad = [t for t in tags if t not in VALID_TAGS]
            print(f"      invalid tag(s) returned: {bad}")
        passed += ok
        failed += (not ok)
    print(f"\n{passed} passed, {failed} failed\n")
    return failed == 0


def check_edge_cases():
    print("=== CHECK 2: edge cases ===")
    edge_cases = {
        "empty list": [],
        "single token": ["kita"],
        "punctuation only": [".", ",", "!", "?"],
        "single character": ["a"],
        "mixed case weirdness": ["NAG-LUNCH", "SoBrAnG", "OMG!!"],
        "numbers": ["123", "45.6", "2:00"],
        "long input (100 tokens)": ["salamat"] * 100,
    }
    passed = 0
    failed = 0
    for name, tokens in edge_cases.items():
        try:
            tags = tag_language(tokens)
            len_ok = len(tags) == len(tokens)
            tags_ok = all(t in VALID_TAGS for t in tags)
            ok = len_ok and tags_ok
            status = "PASS" if ok else "FAIL"
            print(f"{status}: {name:28s} -> {len(tokens)} tokens in, {len(tags)} tags out")
            passed += ok
            failed += (not ok)
        except Exception as e:
            print(f"FAIL: {name:28s} -> CRASHED with {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed\n")
    return failed == 0


def check_consistency_with_raw_path():
    print("=== CHECK 3: tag_language() vs. raw model+vectorizer path ===")

    with open("trained_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    vec = bundle["vectorizer"]

    sentences, tags = load_annotated_data("annotated_dataset_clean.csv")
    splits = split_dataset(sentences, tags)
    test_sents, _ = splits["test"]

    sample_sents = test_sents[:20]  # first 20 test sentences, arbitrary sample

    all_match = True
    total_tokens_checked = 0
    for tokens in sample_sents:
        if not tokens:
            continue
        # tag_language()'s path
        via_tag_language = tag_language(tokens)

        # raw path, done independently, bypassing tag_language entirely
        feature_dicts = extract_dataset_features(tokens)
        X = vec.transform(feature_dicts)
        via_raw = [str(t) for t in model.predict(X)]

        match = via_tag_language == via_raw
        all_match &= match
        total_tokens_checked += len(tokens)
        if not match:
            print(f"MISMATCH on tokens={tokens!r}")
            print(f"  tag_language(): {via_tag_language}")
            print(f"  raw path:       {via_raw}")

    status = "PASS" if all_match else "FAIL"
    print(f"{status}: checked {len(sample_sents)} sentences ({total_tokens_checked} tokens) -- "
          f"tag_language() output {'matches' if all_match else 'does NOT match'} the raw model+vectorizer path")
    return all_match


if __name__ == "__main__":
    r1 = check_output_contract()
    r2 = check_edge_cases()
    r3 = check_consistency_with_raw_path()

    print("=" * 60)
    if r1 and r2 and r3:
        print("ALL CHECKS PASSED -- pinoybot.py's integration is verified correct.")
    else:
        print("SOME CHECKS FAILED -- see details above.")
