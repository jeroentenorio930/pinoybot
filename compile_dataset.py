"""
compile_dataset.py

This is a the data-cleaning script that should be run only once to turn the raw dataset_annotation.csv 
into a new and clean CSV that train.py consumes.

Usage:
    python compile_dataset.py

Produces:
    annotated_dataset_clean.csv  -- the file train.py's load_annotated_data() reads from.
    words_with_nan.csv           -- rows that were dropped (words containing "nan" values).
"""
import pandas as pd

VALID_TAGS = {"ENG", "FIL", "CS", "OTH"}

# the dataset_annotation.csv (not cleaned) has tag typos before cleaning
# TAG_TYPO_MAP below is kept only as a safety net in case a typo slips back in later
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
    This function cleans the final annotated CSV into a tidy version with columns: 
    - sentence_id, token_id, token, tag.

    Writes the clean result to `output_csv` and returns it as a DataFrame.
    """
    df = pd.read_csv(input_csv)
    df = df.rename(columns={"word": "token", "answer": "tag"}) # rename columns: word to token

    if "token_id" in df.columns:
        df = df.drop(columns=["token_id"])

    flagged_frames = []

    # tag cleanup
    df["tag"] = df["tag"].astype(str).str.strip().str.upper() # strip whitespace and convert to uppercase
    df["tag"] = df["tag"].replace("NAN", pd.NA)  # str(NaN) == "nan" -> "NAN" after upper()
    df["tag"] = df["tag"].replace(TAG_TYPO_MAP) # auto correct any remaining typos (this is just a safety net guys)

    bad_tag_mask = ~df["tag"].isin(VALID_TAGS)
    if bad_tag_mask.any():
        flagged_frames.append(df[bad_tag_mask].assign(reason="invalid_tag")) # we just add the reason column as to why the rows were flagged as invalid
        print(f"[compile_dataset] flagging {bad_tag_mask.sum()} rows with unrecoverable tags: " f"{df.loc[bad_tag_mask, 'tag'].unique().tolist()}")
    df = df[~bad_tag_mask] # we drop them from the working dataset and log them to flagged_csv instead for the record

    # missing token cleanup
    # treat "nan" words as genuinely missing and drop. we log this to flagged_csv.
    null_token_mask = df["token"].isnull()
    if null_token_mask.any():
        flagged_frames.append(df[null_token_mask].assign(reason="missing_token")) # we just add the reason column as to why the rows were flagged as invalid
        print(f"[compile_dataset] dropping {null_token_mask.sum()} rows with missing token text")
    df = df[~null_token_mask]

    # write out everything that we flagged or dropped to a separate CSV for the record, so we can inspect it later if needed
    if flagged_frames:
        pd.concat(flagged_frames, ignore_index=True).to_csv(flagged_csv, index=False)
        print(f"[compile_dataset] wrote flagged/dropped rows -> {flagged_csv} (for the record)")
    else:
        print("[compile_dataset] no rows flagged -- dataset is clean")

    # recompute token_id as position-within-sentence
    df["token"] = df["token"].astype(str)
    df = df.sort_values(["sentence_id", "word_id"]) # sort by sentence_id and word_id to ensure it keeps the original word order within each sentence
    df["token_id"] = df.groupby("sentence_id").cumcount() # refresh token_id as position-within-sentence (0-indexed)

    # drop exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["sentence_id", "token_id", "token", "tag"])
    dupes_dropped = before - len(df)
    if dupes_dropped:
        print(f"[compile_dataset] dropped {dupes_dropped} exact duplicate rows")

    # keep only the 4 columns that train.py actually needs and make sure everything is properly sorted
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