# PinoyBot: Filipino Code-Switched Language Identifier

PinoyBot is a word-level language identifier for Filipino-English code-switched text. Given a list of tokens, it tags each one as English, Filipino, an intra-word code-switch, or other. It is a supervised machine learning classifier (scikit-learn) trained on a hand-annotated corpus, using engineered word-structure and context features instead of dictionary lookups.

**Course:** Major Course Output 2 (De La Salle University) | **Task:** word-level language identification for Filipino code-switched text

```python
>>> from pinoybot import tag_language
>>> tag_language(["Madami", "ang", "nag-march", "sa", "EDSA", "monument", "at", "Luneta", "Park", "."])
['FIL', 'FIL', 'CS', 'FIL', 'OTH', 'ENG', 'FIL', 'OTH', 'ENG', 'OTH']
```

---

## Tags

| Tag | Meaning | Examples |
|---|---|---|
| `FIL` | Filipino / Tagalog, including words with native affixes | `bawal`, `sikat`, `mga` |
| `ENG` | English word with English spelling, even in a Filipino sentence | `corrupt`, `monument`, `Park` |
| `CS` | Intra-word code-switch: Filipino and English elements in one word | `naglunch`, `nag-march`, `pina-explain` |
| `OTH` | Punctuation, symbols, numbers, names, abbreviations, hashtags, onomatopoeia | `EDSA`, `Luneta`, `DLSU`, `13`, `.` |

Misspellings and non-standard spellings are tagged by their intended word. Abbreviations are always `OTH`.

## Results

Evaluated on a held-out test set of **5,847 tokens** (15% of the data, split at the sentence level):

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| FIL | 0.98 | 0.98 | 0.98 | 4,520 |
| OTH | 0.89 | 0.88 | 0.88 | 780 |
| ENG | 0.85 | 0.84 | 0.85 | 519 |
| CS | 0.62 | 0.29 | 0.39 | 28 |

**Overall accuracy: 95.2%**, macro F1: 0.78.

Confusion matrix (rows = true, columns = predicted):

| | ENG | FIL | CS | OTH |
|---|---|---|---|---|
| **ENG** | 438 | 40 | 1 | 40 |
| **FIL** | 33 | 4,434 | 4 | 49 |
| **CS** | 11 | 9 | 8 | 0 |
| **OTH** | 31 | 62 | 0 | 687 |

Model selection on the validation set (selected by macro F1, because the classes are heavily imbalanced):

| Model | Val. accuracy | Val. macro F1 |
|---|---|---|
| **Decision Tree** (`max_depth=None`) | 0.949 | 0.761 |
| Multinomial Naive Bayes | 0.944 | 0.668 |

These numbers come from running `train.py` and then `evaluate.py`, so you can reproduce them (see Usage).

### Where it struggles

- **`CS` is the weak class.** Only 28 of the 5,847 test tokens are code-switched, and the model finds fewer than a third of them (recall 0.29). Intra-word switches are rare in the data (196 of about 41,900 tokens), which makes them hard to learn from word-structure features alone.
- **Errors among the big three are spread out.** Of the roughly 280 wrong predictions, about 80 are English words predicted as `FIL` or `OTH`, about 90 are `OTH` tokens predicted as `FIL` or `ENG`, and about 80 are `FIL` tokens predicted as `OTH` or `ENG` (see the confusion matrix).

## How it works

```
dataset_annotation.csv --(compile_dataset.py)--> annotated_dataset_clean.csv
        |
        v
 train.py:  sentence-level 70/15/15 split (random_state=42)
            -> features.py  (per-token feature dictionaries)
            -> DictVectorizer (fit on the training set only)
            -> train Decision Tree (tuned over max_depth) and Naive Bayes
            -> pick the best by validation macro F1
            -> save model + vectorizer to trained_model.pkl
        |
        v
 pinoybot.py:  tag_language(tokens) loads the pickle once, extracts features,
               vectorizes, predicts, and returns one tag per token
```

Splitting happens at the **sentence level, never the token level**, so context features (previous and next word, character n-grams) cannot leak information across train, validation, and test sets. The vectorizer is fit on the training set only for the same reason.

### Features (`features.py`)

No dictionaries or external language-identification models are used. Every feature is computed from the word itself and its neighbours.

| Group | What it captures |
|---|---|
| **Hyphenated affixes** | Numeric prefixes (`12-wheeler`), exact Filipino affix match (`nag-`, `pina-`, `na-`), and whether the root looks foreign (`nag-march`) |
| **Fused affixes** | Intra-word switches without a hyphen (`naglunch`, `ipost`): longest Filipino prefix match plus a check on whether the remaining root looks foreign |
| **Character structure** | 2- and 3-character prefixes and suffixes, vowel ratio, no-vowel words (abbreviations), longest consonant run, ending consonant cluster |
| **Capitalization** | Title case, all caps, capitalized mid-sentence |
| **Context** | Previous and next word, whether the previous word ends in a vowel, sentence start and end, neighbour capitalization |
| **Symbols and numbers** | Single-character symbols, digits, non-alphanumeric characters |
| **Reduplication** | Filipino digram and trigram reduplication patterns |
| **Word identity** | The lowercased word itself |

"Looks foreign" is a letter-based heuristic (it checks for the letters `c, f, j, q, v, x, z, w` and English digraphs such as `th`, `ph`, `ck`), not a dictionary lookup.

## Data

- Source text: the Corpus of Historical Filipino English (CoHFiE), provided by the DLSU College of Computer Studies Center for Language Technologies (CeLT).
- Every token was annotated with `ENG`, `FIL`, `CS`, or `OTH` following the course annotation guidelines.
- The cleaned dataset contains **41,884 tokens in 2,200 sentences**: 32,100 `FIL`, 5,916 `OTH`, 3,672 `ENG`, and 196 `CS`.
- `compile_dataset.py` cleans the raw annotations: it normalizes tag text, drops rows with invalid tags or missing tokens (logged in `words_with_nan.csv`), removes duplicates, and rebuilds token positions within each sentence.

## Requirements

- Python 3
- scikit-learn and pandas: `pip install scikit-learn pandas`

## Usage

```bash
# 1. (once) clean the raw annotations
python compile_dataset.py

# 2. train, select the best model, and save trained_model.pkl
python train.py

# 3. evaluate on the held-out test set
python evaluate.py

# 4. try it interactively
python pinoybot.py        # type tokens separated by spaces
python tester.py          # type a normal sentence; a simple tokenizer splits it
```

`pinoybot.py` needs `trained_model.pkl` in the same folder and raises a clear error if it is missing; run `train.py` to create it.

### Using it from code

```python
from pinoybot import tag_language

tag_language(["Love", "kita", "."])   # ['ENG', 'FIL', 'OTH']
```

The function takes a list of tokens and returns a list of tags of the same length, each one `ENG`, `FIL`, `CS`, or `OTH`. An empty list returns an empty list.

## Tests

| File | What it checks |
|---|---|
| `test_pinoybot.py` | `tag_language()` output contract (length and valid tags), edge cases (empty, single token, punctuation-only, very long input), and that it matches the raw load-extract-vectorize-predict path |
| `test_features.py` | Hyphenated-affix feature edge cases (`nag-march`, `12-wheeler`, `COVID-19`, `-`) |
| `test_train_data.py` | Dataset loading and the 70/15/15 sentence-level split |
| `test_metrics.py` | The metric calculations against hand-computed values |
| `verify_metrics.py` | Recomputes metrics from the confusion matrix and compares the model against a majority-class baseline |
| `eval_external.py` | Evaluates the tagger on the separate `test_data.txt` / `test_labels.txt` files |

## Project files

| File | Purpose |
|---|---|
| `pinoybot.py` | `tag_language()`: loads the trained model and tags tokens |
| `features.py` | Feature extraction |
| `train.py` | Split, vectorize, train, tune, select, and save the model |
| `evaluate.py` | Test-set evaluation (accuracy, precision, recall, F1, confusion matrix) |
| `compile_dataset.py` | Cleans the raw annotations |
| `annotated_dataset_clean.csv` | Cleaned, annotated dataset used for training |
| `trained_model.pkl` | Saved model, vectorizer, and model name |
| `specifications.pdf` | Course project specification |

## Contributors

Contributions as marked in the source comments: Rhian (hyphenated-affix features, dataset loading and splitting), Xan (vectorization, model training and tuning), Lance (model saving), JR (character n-gram and context features), Drea (symbol, number, and reduplication features).
