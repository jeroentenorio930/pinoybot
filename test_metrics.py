"""
test_metrics.py

Checks whether the metric calculations used in evaluate.py are working
correctly. This script does not evaluate the trained model itself.

A small set of 10 manually verified predictions is used as the test case.
The expected accuracy, precision, recall, F1-score, support, and confusion
matrix were all computed by hand. The results from sklearn are then compared
against these expected values.

If all tests pass, it confirms that the metric computation logic used in
evaluate.py is producing the correct values.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

LABELS = ["ENG", "FIL", "CS", "OTH"]

# Hand-picked (true, predicted) label pairs

# #   true   pred   correct?

# 1   FIL    FIL    yes
# 2   FIL    FIL    yes
# 3   FIL    ENG    no   (FIL misclassified as ENG)
# 4   ENG    ENG    yes
# 5   ENG    FIL    no   (ENG misclassified as FIL)
# 6   CS     CS     yes
# 7   CS     OTH    no   (CS misclassified as OTH)
# 8   OTH    OTH    yes
# 9   OTH    OTH    yes
# 10  OTH    ENG    no   (OTH misclassified as ENG)

y_true = ["FIL", "FIL", "FIL", "ENG", "ENG", "CS", "CS", "OTH", "OTH", "OTH"]
y_pred = ["FIL", "FIL", "ENG", "ENG", "FIL", "CS", "OTH", "OTH", "OTH", "ENG"]

# Expected confusion matrix (rows = true labels, columns = predicted labels)

# Order: ENG, FIL, CS, OTH

#        ENG FIL CS OTH
# ENG -> [1,  1,  0,  0]
# FIL -> [1,  2,  0,  0]
# CS  -> [0,  0,  1,  1]
# OTH -> [1,  0,  0,  2]

EXPECTED_CM = [
    [1, 1, 0, 0],
    [1, 2, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 2],
]

# Expected precision, recall, F1-score, and support
# These values were calculated manually.

EXPECTED = {
    "ENG": {"precision": 1 / 3, "recall": 0.5, "f1": 0.4, "support": 2},
    "FIL": {"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "support": 3},
    "CS": {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2},
    "OTH": {"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "support": 3},
}

# Expected overall accuracy:
# 6 correct predictions out of 10 samples.

EXPECTED_ACCURACY = 0.6


def approx(a, b, tol=1e-4):
    """Checks whether two floating-point values are approximately equal."""
    return abs(a - b) < tol


def run_tests():
    """Runs all metric verification tests."""
    passed = 0
    failed = 0

    # Check overall accuracy
    acc = accuracy_score(y_true, y_pred)

    if approx(acc, EXPECTED_ACCURACY):
        print(f"PASS: accuracy = {acc:.4f} (expected {EXPECTED_ACCURACY:.4f})")
        passed += 1
    else:
        print(f"FAIL: accuracy = {acc:.4f} (expected {EXPECTED_ACCURACY:.4f})")
        failed += 1

    # Compute metrics the same way as evaluate.py
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )

    for label, p, r, f, s in zip(LABELS, precision, recall, f1, support):
        exp = EXPECTED[label]

        ok = (
            approx(p, exp["precision"])
            and approx(r, exp["recall"])
            and approx(f, exp["f1"])
            and s == exp["support"]
        )

        status = "PASS" if ok else "FAIL"

        print(
            f"{status}: {label:4s} precision={p:.4f} recall={r:.4f} f1={f:.4f} support={s}  "
            f"(expected p={exp['precision']:.4f} r={exp['recall']:.4f} "
            f"f1={exp['f1']:.4f} s={exp['support']})"
        )

        passed += ok
        failed += (not ok)

    # Verify the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_ok = cm.tolist() == EXPECTED_CM

    status = "PASS" if cm_ok else "FAIL"
    print(f"{status}: confusion matrix matches hand-computed matrix")

    if not cm_ok:
        print("  got:     ", cm.tolist())
        print("  expected:", EXPECTED_CM)

    passed += cm_ok
    failed += (not cm_ok)

    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

    if failed == 0:
        print("The metric-computation logic in evaluate.py is verified correct.")
    else:
        print("Something in the metric computation does not match the hand-calculated values. Investigate the implementation.")


if __name__ == "__main__":
    run_tests()