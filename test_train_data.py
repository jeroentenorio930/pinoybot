"""
test_train_data.py

Sanity checks for Rhian's part of train.py: load_annotated_data() and
split_dataset(). Run directly: python test_train_data.py
"""

from train import load_annotated_data, split_dataset

CSV_PATH = "annotated_dataset_clean.csv"


def run_tests():
    sentences, tags = load_annotated_data(CSV_PATH)

    # --- basic shape checks ---
    assert len(sentences) == len(tags), "sentences/tags length mismatch"
    for s, t in zip(sentences, tags):
        assert len(s) == len(t), f"token/tag count mismatch in a sentence: {s}"
    print(f"PASS: loaded {len(sentences)} sentences, shapes match")

    total_tokens = sum(len(s) for s in sentences)
    print(f"PASS: {total_tokens} total tokens loaded")

    # --- all tags are valid ---
    valid_tags = {"ENG", "FIL", "CS", "OTH"}
    all_tags = {tag for sent_tags in tags for tag in sent_tags}
    bad_tags = all_tags - valid_tags
    assert not bad_tags, f"found invalid tags: {bad_tags}"
    print(f"PASS: all tags valid, tags seen = {sorted(all_tags)}")

    # --- missing-token rows should have been dropped entirely, not kept as
    #     a literal "nan" placeholder (team decision, superseded the earlier
    #     placeholder-fill approach) ---
    has_literal_nan = any("nan" in s for s in sentences)
    assert not has_literal_nan, (
        "found literal 'nan' token(s) -- these rows should have been dropped "
        "by compile_dataset.py, not kept as placeholders"
    )
    print("PASS: no literal 'nan' placeholder tokens present (correctly dropped)")

    # --- no empty-string or None tokens slipped through ---
    for s in sentences:
        for tok in s:
            assert tok is not None and tok != "", f"found empty/None token in: {s}"
    print("PASS: no empty or None tokens")

    # --- split_dataset ---
    splits = split_dataset(sentences, tags)
    train_s, train_t = splits["train"]
    val_s, val_t = splits["val"]
    test_s, test_t = splits["test"]

    total_split_sentences = len(train_s) + len(val_s) + len(test_s)
    assert total_split_sentences == len(sentences), "split lost or duplicated sentences"
    print(f"PASS: split sizes -> train={len(train_s)}, val={len(val_s)}, test={len(test_s)} "
          f"(sums to {total_split_sentences}, matches total {len(sentences)})")

    # roughly 70/15/15 (allow some tolerance since split_dataset splits on count, not exact ratio)
    train_ratio = len(train_s) / len(sentences)
    val_ratio = len(val_s) / len(sentences)
    test_ratio = len(test_s) / len(sentences)
    print(f"PASS: split ratios -> train={train_ratio:.3f}, val={val_ratio:.3f}, test={test_ratio:.3f}")
    assert 0.65 <= train_ratio <= 0.75, f"train ratio off: {train_ratio}"
    assert 0.10 <= val_ratio <= 0.20, f"val ratio off: {val_ratio}"
    assert 0.10 <= test_ratio <= 0.20, f"test ratio off: {test_ratio}"

    # --- no sentence-level leakage across splits ---
    # (checked by identity via id(), since sentences are lists of tokens
    #  and could coincidentally be equal lists otherwise)
    train_ids = {id(s) for s in train_s}
    val_ids = {id(s) for s in val_s}
    test_ids = {id(s) for s in test_s}
    assert not (train_ids & val_ids), "train/val overlap detected"
    assert not (train_ids & test_ids), "train/test overlap detected"
    assert not (val_ids & test_ids), "val/test overlap detected"
    print("PASS: no sentence-level overlap between train/val/test")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()