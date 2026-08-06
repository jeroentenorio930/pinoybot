"""
eval_external.py

Evaluates PinoyBot against external test_data and test_labels files.
Run this script from inside your pinoybot/ folder:

    python eval_external.py
"""

import os
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from pinoybot import tag_language

LABELS = ["ENG", "FIL", "CS", "OTH"]
DATA_FILE = "test_data.txt"
LABELS_FILE = "test_labels.txt"


def load_test_files(data_path: str, labels_path: str):
    """
    Parses sentences and labels that are separated by pipe characters (|).
    """
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Could not find '{data_path}' or '{labels_path}' in the current directory."
        )

    sentences = []
    sentence_labels = []

    with open(data_path, "r", encoding="utf-8") as f_data, open(labels_path, "r", encoding="utf-8") as f_labels:
        for data_line, label_line in zip(f_data, f_labels):
            data_line = data_line.strip()
            label_line = label_line.strip()
            
            if not data_line or not label_line:
                continue
                
            # Split by the pipe character based on the raw text format
            tokens = data_line.split("|")
            labels = [lbl.upper() for lbl in label_line.split("|")]
            
            # Only add them if the counts perfectly match to avoid crashing
            if len(tokens) == len(labels):
                sentences.append(tokens)
                sentence_labels.append(labels)
            else:
                print(f"Skipping a misaligned line: {len(tokens)} tokens vs {len(labels)} tags.")

    return sentences, sentence_labels


def run_evaluation():
    print(f"--- Loading '{DATA_FILE}' and '{LABELS_FILE}' ---")
    sentences, sentence_labels = load_test_files(DATA_FILE, LABELS_FILE)

    all_true = []
    all_pred = []
    all_tokens = []

    print(f"Successfully loaded {sum(len(s) for s in sentences)} tokens across {len(sentences)} reconstructed sentences.")
    print("Running PinoyBot predictions (this may take a few seconds)...")
    
    # Run predictions sentence by sentence to preserve context features
    for tokens, true_tags in zip(sentences, sentence_labels):
        pred_tags = tag_language(tokens)

        # Filter out any unexpected labels from predictions just in case
        clean_preds = [t if t in LABELS else "OTH" for t in pred_tags]
        clean_trues = [t if t in LABELS else "OTH" for t in true_tags]

        all_tokens.extend(tokens)
        all_true.extend(clean_trues)
        all_pred.extend(clean_preds)

    print("\nEvaluation Complete! Calculating metrics...")

    # 1. Overall Metrics
    acc = accuracy_score(all_true, all_pred)
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        all_true, all_pred, labels=LABELS, average=None, zero_division=0
    )
    cm = confusion_matrix(all_true, all_pred, labels=LABELS)

    # 2. Print Detailed Table with TP, FP, FN
    print("\n" + "=" * 70)
    print("                      EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy: {acc:.4f} ({acc * 100:.2f}%)\n")

    print(
        f"{'Label':<6} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}"
    )
    print("-" * 75)

    for i, label in enumerate(LABELS):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = p_class[i]
        r = r_class[i]
        f1 = f1_class[i]
        sup = support_class[i]

        print(
            f"{label:<6} | {tp:<6} | {fp:<6} | {fn:<6} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {sup:<8}"
        )

    # 3. Print Confusion Matrix
    print("\n" + "=" * 70)
    print("             CONFUSION MATRIX (Rows=True, Cols=Predicted)")
    print("=" * 70)
    header = "          " + "".join(f"{lbl:>8}" for lbl in LABELS)
    print(header)
    for label, row in zip(LABELS, cm):
        print(f"{label:<8}" + "".join(f"{val:>8}" for val in row))

    # 4. Error Analysis (Most Common Misclassifications)
    print("\n" + "=" * 70)
    print("        ERROR ANALYSIS: MOST COMMON MISCLASSIFICATIONS")
    print("=" * 70)

    misclassifications = defaultdict(list)
    for tok, true_lbl, pred_lbl in zip(all_tokens, all_true, all_pred):
        if true_lbl != pred_lbl:
            misclassifications[(true_lbl, pred_lbl)].append(tok)

    sorted_errors = sorted(
        misclassifications.items(), key=lambda item: len(item[1]), reverse=True
    )

    if not sorted_errors:
        print("No misclassifications found! Perfect 100% score.")
    else:
        for (true_lbl, pred_lbl), word_list in sorted_errors[:8]:
            count = len(word_list)
            # Pick up to 5 unique word samples
            samples = list(dict.fromkeys(word_list))[:5]
            sample_str = ", ".join(f"'{w}'" for w in samples)
            print(
                f"• True: {true_lbl:<4} -> Predicted: {pred_lbl:<4} | Count: {count:<4} | Examples: {sample_str}"
            )

    print("\nDone! Copy these tables directly into your PDF report.")


if __name__ == "__main__":
    run_evaluation()