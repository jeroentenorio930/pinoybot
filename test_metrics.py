"""
test_metrics.py

This does NOT test the trained model or the dataset. It tests whether
the METRIC-COMPUTATION CODE in evaluate.py (precision_recall_fscore_support,
confusion_matrix, accuracy_score, and how we call/print them) is correct.

How: we hand-pick a tiny y_true/y_pred pair (10 tokens), hand-compute
precision/recall/F1/support/confusion-matrix for each with pen and paper
(shown in the comments below), then check that sklearn -- called the exact
same way evaluate.py calls it -- produces the same numbers.

If this test passes, you know the ARITHMETIC in evaluate.py is trustworthy.
It says nothing about whether the model itself is good; that's a separate
question answered by running evaluate.py on the real test set.

Run: python test_metrics.py
"""

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

LABELS = ["ENG", "FIL", "CS", "OTH"]

# --- The 10 hand-picked (true, predicted) pairs ---
#  #   true   pred   correct?
#  1   FIL    FIL    yes
#  2   FIL    FIL    yes
#  3   FIL    ENG    no   (FIL misclassified as ENG)
#  4   ENG    ENG    yes
#  5   ENG    FIL    no   (ENG misclassified as FIL)
#  6   CS     CS     yes
#  7   CS     OTH    no   (CS misclassified as OTH)
#  8   OTH    OTH    yes
#  9   OTH    OTH    yes
#  10  OTH    ENG    no   (OTH misclassified as ENG)
y_true = ["FIL", "FIL", "FIL", "ENG", "ENG", "CS", "CS", "OTH", "OTH", "OTH"]
y_pred = ["FIL", "FIL", "ENG", "ENG", "FIL", "CS", "OTH", "OTH", "OTH", "ENG"]

# --- Hand-computed confusion matrix (rows=true, cols=predicted) ---
# order: ENG, FIL, CS, OTH
#            ENG  FIL  CS  OTH
#   ENG   [   1,   1,  0,   0 ]   <- true ENG: 1 correct, 1 predicted FIL
#   FIL   [   1,   2,  0,   0 ]   <- true FIL: 2 correct, 1 predicted ENG
#   CS    [   0,   0,  1,   1 ]   <- true CS:  1 correct, 1 predicted OTH
#   OTH   [   1,   0,  0,   2 ]   <- true OTH: 2 correct, 1 predicted ENG
EXPECTED_CM = [
    [1, 1, 0, 0],
    [1, 2, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 2],
]

# --- Hand-computed precision / recall / F1 / support per label ---
# precision = TP / (TP + FP)   recall = TP / (TP + FN)   F1 = 2PR/(P+R)
#
# ENG: TP=1, FP=2 (FIL->ENG, OTH->ENG), FN=1 (ENG->FIL)
#      precision = 1/3 = 0.3333   recall = 1/2 = 0.5   F1 = 0.4        support=2
# FIL: TP=2, FP=1 (ENG->FIL), FN=1 (FIL->ENG)
#      precision = 2/3 = 0.6667   recall = 2/3 = 0.6667  F1 = 0.6667   support=3
# CS:  TP=1, FP=0, FN=1 (CS->OTH)
#      precision = 1/1 = 1.0      recall = 1/2 = 0.5     F1 = 0.6667   support=2
# OTH: TP=2, FP=1 (CS->OTH), FN=1 (OTH->ENG)
#      precision = 2/3 = 0.6667   recall = 2/3 = 0.6667  F1 = 0.6667   support=3
EXPECTED = {
    "ENG": {"precision": 1 / 3,  "recall": 0.5,      "f1": 0.4,       "support": 2},
    "FIL": {"precision": 2 / 3,  "recall": 2 / 3,     "f1": 2 / 3,     "support": 3},
    "CS":  {"precision": 1.0,    "recall": 0.5,       "f1": 2 / 3,     "support": 2},
    "OTH": {"precision": 2 / 3,  "recall": 2 / 3,     "f1": 2 / 3,     "support": 3},
}

# --- Hand-computed overall accuracy ---
# 4 correct out of 10 total (rows 1,2,4,6,8,9 are correct = 6 correct)
# recount: rows 1,2,4,6,8,9 -> that's 6 correct out of 10
EXPECTED_ACCURACY = 0.6


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


def run_tests():
    passed = 0
    failed = 0

    # --- accuracy ---
    acc = accuracy_score(y_true, y_pred)
    if approx(acc, EXPECTED_ACCURACY):
        print(f"PASS: accuracy = {acc:.4f} (expected {EXPECTED_ACCURACY:.4f})")
        passed += 1
    else:
        print(f"FAIL: accuracy = {acc:.4f} (expected {EXPECTED_ACCURACY:.4f})")
        failed += 1

    # --- precision / recall / f1 / support, called the same way evaluate.py does ---
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    for label, p, r, f, s in zip(LABELS, precision, recall, f1, support):
        exp = EXPECTED[label]
        ok = approx(p, exp["precision"]) and approx(r, exp["recall"]) and approx(f, exp["f1"]) and s == exp["support"]
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {label:4s} precision={p:.4f} recall={r:.4f} f1={f:.4f} support={s}  "
              f"(expected p={exp['precision']:.4f} r={exp['recall']:.4f} f1={exp['f1']:.4f} s={exp['support']})")
        passed += ok
        failed += (not ok)

    # --- confusion matrix ---
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
        print("Something in the metric computation does not match hand-calculated values -- investigate.")


if __name__ == "__main__":
    run_tests()
