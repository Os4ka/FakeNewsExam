# src/main.py
from .model.model_predict import FakeNewsClassifier


def run_cli():
    clf = FakeNewsClassifier()

    print("Fake News Detector (type 'q' to quit)\n")
    while True:
        print("Enter a news article text (Press enter once to access next line & press enter twice to activate prediction):")


        lines = []
        while True:
            line = input()
            if line.strip().lower() in ("q", "quit", "exit"):
                return 
            if line == "":
                break  
            lines.append(line)

        text = "\n".join(lines)
        # --------------------------------

        if not text.strip():
            print("No text entered. Try again.\n")
            continue

        label, probs, top_words = clf.predict_with_explanation(text, top_n=8)

        print(f"\nPrediction: {label}")
        print(f"Probabilities [FAKE, REAL]: {probs}")

        if top_words:
            print("\nTop words that influenced this prediction:")
            for word, score in top_words:
 
                print(f"  {word:20s}  contribution: {score:.4f}")
        else:
            print("\n(No informative words found in this text.)")

        print("-" * 60)


if __name__ == "__main__":
    run_cli()
