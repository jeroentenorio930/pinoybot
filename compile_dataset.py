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

NOTE: tag typos (ENF/OHT/EGNG/OT/e) and the tag=NaN rows have already been
manually corrected to their proper values (e.g. NaN -> OTH) directly in
dataset_annotation.csv. The TAG_TYPO_MAP below is kept only as a safety net
in case a typo slips back in later -- it won't affect already-fixed rows.

NOTE: rows with a missing `word` cell (23 rows, mostly dropped quote-mark
tokens, all already tagged OTH) are NOT dropped. Dropping them mid-sentence
would shift token positions and break prev/next-word context features for
the words next to them. Instead, the missing cell is filled with the
literal placeholder string "nan" and the row is kept in place.

IMPORTANT for whoever writes n-gram/context features: "nan" appears as a
literal token string in the clean dataset (not a missing value) -- treat it
as an ordinary word, don't filter it out or special-case it.
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
    flagged_csv: str = "flagged_for_review.csv",
) -> pd.DataFrame:
    """
    Cleans the final annotated CSV into a tidy version with columns:
    sentence_id, token_id, token, tag.

    Steps:
      1. Rename raw columns (word -> token, answer -> tag).
      2. Strip whitespace from tags (e.g. "FIL " -> "FIL").
      3. Auto-correct any remaining known typo tags via TAG_TYPO_MAP (safety net).
      4. Fill missing `word` cells with the literal placeholder "nan"
         (kept in place, not dropped, to preserve sentence/context
         alignment for neighboring tokens). Drop rows whose tag is still
         invalid after correction -- these are written to `flagged_csv`
         for manual review, not guessed.
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

    # --- missing token handling ---
    # Kept in place (not dropped) so sentence structure and token positions
    # stay intact for neighboring words' context features. Filled with the
    # literal placeholder string "nan" rather than guessed.
    null_token_mask = df["token"].isnull()
    if null_token_mask.any():
        print(f"[compile_dataset] filling {null_token_mask.sum()} missing token cells with placeholder 'nan'")
    df["token"] = df["token"].fillna("nan")

    if flagged_frames:
        pd.concat(flagged_frames, ignore_index=True).to_csv(flagged_csv, index=False)
        print(f"[compile_dataset] wrote flagged rows -> {flagged_csv} (please review as a team)")
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
        flagged_csv="flagged_for_review.csv",
    )