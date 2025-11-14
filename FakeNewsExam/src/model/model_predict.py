# src/model_predict.py
from pathlib import Path
import joblib


class FakeNewsClassifier:
    def __init__(self, models_dir: str = "models"):
        models_path = Path(models_dir)
        self.model = joblib.load(models_path / "fake_news_model.pkl")
        self.vectorizer = joblib.load(models_path / "tfidf_vectorizer.pkl")

    def predict_news(self, text: str):
        vec = self.vectorizer.transform([text])
        pred = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)[0]
        return pred, proba

    def predict_with_explanation(self, text: str, top_n: int = 10):
        """
        Returns: predicted label, probabilities, and a list of (word, contribution) tuples
        showing which words pushed most towards the prediction.
        """
        vec = self.vectorizer.transform([text])
        pred = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)[0]

        # Get feature names and coefficients
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[0]  # binary classification -> one row

        # Convert sparse vector to dense
        values = vec.toarray()[0]

        # Contribution of each word towards the "REAL" side of the decision
        contributions = values * coefs

        # Only keep words that actually appear in this text
        indices = [i for i, v in enumerate(values) if v != 0.0]

        words_contrib = [(feature_names[i], contributions[i]) for i in indices]

        # In sklearn, coef_ corresponds to classes_[1] (usually "REAL" here)
        # Positive contrib -> pushes towards REAL
        # Negative contrib -> pushes towards FAKE
        classes = list(self.model.classes_)

        if pred == classes[1]:  # predicted "REAL"
            # sort by most positive contribution
            words_contrib.sort(key=lambda x: x[1], reverse=True)
        else:  # predicted "FAKE"
            # sort by most negative contribution first
            words_contrib.sort(key=lambda x: x[1])

        top_words = words_contrib[:top_n]

        return pred, proba, top_words
