# src/main.py
from .model.model_predict import FakeNewsClassifier


def run_cli():
    clf = FakeNewsClassifier()

    print("Fake News Detector (type 'q' to quit)\n")
    while True:
        text = input("Enter a news headline or article text: ")
        if text.lower() in ("q", "quit", "exit"):
            break

        label, probs, top_words = clf.predict_with_explanation(text, top_n=8)

        print(f"\nPrediction: {label}")
        print(f"Probabilities [FAKE, REAL]: {probs}")

        if top_words:
            print("\nTop words that influenced this prediction:")
            for word, score in top_words:
                # score is the contribution towards REAL (positive) or FAKE (negative)
                print(f"  {word:20s}  contribution: {score:.4f}")
        else:
            print("\n(No informative words found in this text.)")

        print("-" * 60)


if __name__ == "__main__":
    run_cli()
