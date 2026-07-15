import pickle

def load_annotated_data(csv_path: str):
    """
    Loads manually annotated dataset.
    """
    df = pd.read_csv(csv_path)

    unique_sentences = df["sentence_id"].unique()

def train_and_pickle():
    # 1. Load annotated dataset - call prev function

    # 2. Convert raw strings into feature dictionaries

    # 3. Train-Validation-Test Split (70-15-15)

    # 4. Vectorize features

    # 5. Initialize and train scikit-learn model

    # 6. Performance validation check

    # 7. Save model and vectorizer together to disk

if __name__ == "__main__":
    train_and_pickle()

