"""
compile_dataset.py

Rhian's dataset-cleaning script. Run this ONCE (not every training run) to
turn the raw dataset_annotation.csv into a clean, tidy CSV that train.py
consumes.

Usage:
    python compile_dataset.py

Produces:
    annotated_dataset_clean.csv  -- the file train.py's load_annotated_data()
                                     reads from.
    words_with_nan.csv           -- rows that were dropped, for manual review.

NOTE: tag typos (ENF/OHT/EGNG/OT/e) and the tag=NaN rows have already been
manually corrected to their proper values (e.g. NaN -> OTH) directly in
dataset_annotation.csv. The TAG_TYPO_MAP below is kept only as a safety net
in case a typo slips back in later -- it won't affect already-fixed rows.

NOTE (superseded 2026-07-20): earlier versions of this script filled the 23
rows with a missing `word` cell with the literal placeholder string "nan"
to preserve sentence-position alignment. The team has since decided to
treat these as genuinely missing and DROP them instead, matching what was
being done manually in R (r_data_cleaning.R). This script now does that
drop directly, so the separate R step is no longer needed -- everyone
should regenerate annotated_dataset_clean.csv from this single script
instead of running R afterward. If any manual tag corrections were made
directly on nan_removed_41884rows.csv, make sure the SAME corrections are
also applied to dataset_annotation.csv (the raw source), otherwise re-running
this script will produce a clean CSV that doesn't reflect those fixes.
"""

import pandas as pd

VALID_TAGS = {"ENG", "FIL", "CS", "OTH"}

TAG_TYPO_MAP = {
    "ENF": "ENG",
    "OHT": "OTH",
    "EGNG": "ENG",
    "OT": "OTH",
    "E": "ENG",
}


def compile_dataset(
    input_csv: str = "dataset_annotation.csv",
    output_csv: str = "annotated_dataset_clean.csv",
    flagged_csv: str = "words_with_nan.csv",
) -> pd.DataFrame:
    """
    Cleans the final annotated CSV into a tidy version with columns:
    sentence_id, token_id, token, tag.

    Steps:
      1. Rename raw columns (word -> token, answer -> tag).
      2. Strip whitespace from tags (e.g. "FIL " -> "FIL").
      3. Auto-correct any remaining known typo tags via TAG_TYPO_MAP (safety net).
      4. Drop rows with a missing `word` cell (23 rows, mostly dropped
         quote-mark tokens) and rows whose tag is still invalid after
         correction -- both are written to `flagged_csv` for the record,
         not silently discarded.
      5. Recompute token_id as position-within-sentence (word_id in the
         source file is a GLOBAL row counter, not a per-sentence index).
      6. Drop exact duplicate (sentence_id, token_id, token, tag) rows.

    Writes the clean result to `output_csv` and returns it as a DataFrame.
    """
    df = pd.read_csv(input_csv)
    df = df.rename(columns={"word": "token", "answer": "tag"})

    if "token_id" in df.columns:
        df = df.drop(columns=["token_id"])

    flagged_frames = []

    # --- tag cleanup ---
    df["tag"] = df["tag"].astype(str).str.strip().str.upper()
    df["tag"] = df["tag"].replace("NAN", pd.NA)  # str(NaN) == "nan" -> "NAN" after upper()
    df["tag"] = df["tag"].replace(TAG_TYPO_MAP)

    bad_tag_mask = ~df["tag"].isin(VALID_TAGS)
    if bad_tag_mask.any():
        flagged_frames.append(df[bad_tag_mask].assign(reason="invalid_tag"))
        print(f"[compile_dataset] flagging {bad_tag_mask.sum()} rows with unrecoverable tags: "
              f"{df.loc[bad_tag_mask, 'tag'].unique().tolist()}")
    df = df[~bad_tag_mask]

    # --- missing token cleanup ---
    # Team decision: treat these as genuinely missing and drop, rather than
    # filling with a placeholder. Logged to flagged_csv either way.
    null_token_mask = df["token"].isnull()
    if null_token_mask.any():
        flagged_frames.append(df[null_token_mask].assign(reason="missing_token"))
        print(f"[compile_dataset] dropping {null_token_mask.sum()} rows with missing token text")
    df = df[~null_token_mask]

    if flagged_frames:
        pd.concat(flagged_frames, ignore_index=True).to_csv(flagged_csv, index=False)
        print(f"[compile_dataset] wrote flagged/dropped rows -> {flagged_csv} (for the record)")
    else:
        print("[compile_dataset] no rows flagged -- dataset is clean")

    # --- recompute token_id as position-within-sentence ---
    df["token"] = df["token"].astype(str)
    df = df.sort_values(["sentence_id", "word_id"])
    df["token_id"] = df.groupby("sentence_id").cumcount()

    # --- drop exact duplicates ---
    before = len(df)
    df = df.drop_duplicates(subset=["sentence_id", "token_id", "token", "tag"])
    dupes_dropped = before - len(df)
    if dupes_dropped:
        print(f"[compile_dataset] dropped {dupes_dropped} exact duplicate rows")

    clean = df[["sentence_id", "token_id", "token", "tag"]].sort_values(
        ["sentence_id", "token_id"]
    ).reset_index(drop=True)

    clean.to_csv(output_csv, index=False)
    print(f"[compile_dataset] wrote {len(clean)} rows, "
          f"{clean['sentence_id'].nunique()} sentences -> {output_csv}")
    return clean


if __name__ == "__main__":
    compile_dataset(
        input_csv="dataset_annotation.csv",
        output_csv="annotated_dataset_clean.csv",
        flagged_csv="words_with_nan.csv",
    )