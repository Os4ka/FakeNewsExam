from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_features(data: pd.DataFrame):
    data = data.copy()
    data["combined_text"] = data["title"].astype(str) + " " + data["text"].astype(str)
    X = data["combined_text"]
    y = data["label"]
    return X, y


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def vectorize_text(
    X_train, X_test
) -> Tuple[TfidfVectorizer, "scipy.sparse.csr_matrix", "scipy.sparse.csr_matrix"]:
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return vectorizer, X_train_vec, X_test_vec
