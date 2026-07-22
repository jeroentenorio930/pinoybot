"""
verify_metrics.py

Performs a few sanity checks on the evaluation metrics reported by
evaluate.py using the full annotated dataset.

Since it's not practical to manually verify thousands of test tokens,
this script validates the results in two different ways.

CHECK 1:
Recomputes accuracy, precision, recall, and F1-score directly from the
confusion matrix using the standard TP, FP, and FN formulas. The results
are then compared against sklearn's metrics to confirm they match.

CHECK 2:
Compares the trained model against a simple majority-class baseline
(a classifier that always predicts the most common label). The trained
model should perform better than this baseline to show that it has learned
useful patterns from the data.
"""

from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

from evaluate import load_model_bundle, get_test_set

LABELS = ["ENG", "FIL", "CS", "OTH"]


def manual_metrics_from_confusion_matrix(cm, labels):
    """
    Computes evaluation metrics directly from the confusion matrix
    without using sklearn's metric functions.

    TP = True Positives
    FP = False Positives
    FN = False Negatives
    """
    n = len(labels)
    total = cm.sum()
    correct = sum(cm[i][i] for i in range(n))
    accuracy = correct / total

    results = {}

    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp # Predicted as this label but incorrect
        fn = cm[i, :].sum() - tp # Actual label but predicted differently
        support = cm[i, :].sum() # Total occurrences of the label

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support)
        }

    return accuracy, results


def run():
    """Runs both metric verification checks."""
    print("Loading model and rebuilding test set...")

    model, vec, model_name = load_model_bundle()
    X_test, y_test = get_test_set()

    X_test_vec = vec.transform(X_test) # Convert test data into feature vectors
    y_pred = model.predict(X_test_vec) # Generate predictions

    # sklearn metrics (same values reported by evaluate.py)
    sk_accuracy = accuracy_score(y_test, y_pred)
    sk_precision, sk_recall, sk_f1, sk_support = precision_recall_fscore_support(
        y_test, y_pred, labels=LABELS, average=None, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    # Check 1: recompute metrics manually from the confusion matrix
    manual_accuracy, manual_results = manual_metrics_from_confusion_matrix(cm, LABELS)

    print("\n=== CHECK 1: sklearn vs. manual confusion-matrix arithmetic ===")
    print(f"{'Label':6s}{'sklearn P':>12s}{'manual P':>10s}{'sklearn R':>12s}{'manual R':>10s}"
          f"{'sklearn F1':>12s}{'manual F1':>10s}   match?")

    all_match = True

    for i, label in enumerate(LABELS):
        m = manual_results[label]

        p_match = abs(sk_precision[i] - m["precision"]) < 1e-9
        r_match = abs(sk_recall[i] - m["recall"]) < 1e-9
        f_match = abs(sk_f1[i] - m["f1"]) < 1e-9

        ok = p_match and r_match and f_match
        all_match &= ok

        print(f"{label:6s}{sk_precision[i]:12.4f}{m['precision']:10.4f}"
              f"{sk_recall[i]:12.4f}{m['recall']:10.4f}"
              f"{sk_f1[i]:12.4f}{m['f1']:10.4f}   {'YES' if ok else 'NO -- MISMATCH'}")

    acc_match = abs(sk_accuracy - manual_accuracy) < 1e-9
    all_match &= acc_match

    print(f"\naccuracy: sklearn={sk_accuracy:.6f}  manual={manual_accuracy:.6f}  "
          f"match={'YES' if acc_match else 'NO -- MISMATCH'}")

    print(f"\nAll per-class + accuracy numbers agree between sklearn and manual "
          f"confusion-matrix math: {'YES -- verified consistent' if all_match else 'NO -- investigate!'}")

    # Check 2: compare against a majority-class baseline
    label_counts = Counter(y_test)
    majority_label, majority_count = label_counts.most_common(1)[0]
    baseline_accuracy = majority_count / len(y_test)

    print("\n=== CHECK 2: majority-class baseline sanity check ===")
    print(f"Most common label in test set: {majority_label} ({majority_count}/{len(y_test)} = {baseline_accuracy:.4f})")
    print(f"A classifier that ALWAYS guesses '{majority_label}' would score {baseline_accuracy:.4f} accuracy.")
    print(f"Your trained model scored {sk_accuracy:.4f} accuracy.")

    if sk_accuracy > baseline_accuracy:
        print(f"-> Model beats the naive baseline by {sk_accuracy - baseline_accuracy:.4f} -- it's learning something real.")
    else:
        print(f"-> WARNING: model does NOT beat the naive baseline -- something is likely wrong.")


if __name__ == "__main__":
    run()