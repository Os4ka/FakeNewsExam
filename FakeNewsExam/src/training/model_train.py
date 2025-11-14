from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.dataprocessing.data_loader import load_and_prepare_data


def train_and_save_model():
    # 1) Load cleaned data (combined_text + label)
    print("Loading and preparing data...")
    data = load_and_prepare_data("data")

    X = data["combined_text"]
    y = data["label"]

    # 2) Train/test split (same as Notebook 2)
    print("Splitting data (train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3) TF-IDF vectorization (same params as in notebook)
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4) Train Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # 5) Evaluation (classification report + confusion matrix)
    print("Evaluating model on test data...")
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # 6) Save model + vectorizer (matches your joblib.dump in Notebook 2)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(model, models_dir / "fake_news_model.pkl")
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.pkl")

    print("Saved model and vectorizer to 'models/' directory.")


if __name__ == "__main__":
    train_and_save_model()