import pandas as pd
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Apply the same cleaning as in Notebook 1:
    - fix encoding issues
    - remove strange characters
    - collapse whitespace
    """
    if isinstance(text, str):
        try:
            text = text.encode("latin1").decode("utf-8")
        except UnicodeEncodeError:
            # if encoding fails, just keep original
            pass

        text = (
            text.replace("Â", "")
                .replace("â€™", "'")
                .replace("â€œ", '"')
                .replace("â€", '"')
                .replace("â€“", "-")
                .replace("â€”", "-")
                .replace("â€˜", "'")
                .replace("â€¦", "...")
        )

        # remove remaining non-ASCII characters
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        # collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Replicates the Notebook 1 EDA pipeline:
    - load Fake.csv and True.csv
    - clean title/text
    - add labels
    - combine
    - drop subject/date
    - drop duplicates and NaNs
    - create combined_text and drop original title/text
    Returns a DataFrame with columns: combined_text, label
    """
    data_path = Path(data_dir)
    fake_path = data_path / "Fake.csv"
    true_path = data_path / "True.csv"

    # Load raw data (same encoding as in notebook)
    fake_df = pd.read_csv(fake_path, encoding="latin1")
    true_df = pd.read_csv(true_path, encoding="latin1")

    # Clean text fields
    for df in (fake_df, true_df):
        df["title"] = df["title"].apply(clean_text)
        df["text"] = df["text"].apply(clean_text)

    # Add labels
    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    # Combine into one dataset
    data = pd.concat([fake_df, true_df], ignore_index=True)

    # Drop subject and date (like Notebook 1)
    for col in ["subject", "date"]:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Remove duplicates and rows with missing title/text
    data = data.drop_duplicates(subset=["title", "text"], keep="first")
    data = data.dropna(subset=["title", "text"]).reset_index(drop=True)

    # Create combined_text and drop original title/text
    data["combined_text"] = data["title"] + " " + data["text"]
    data = data.drop(columns=["title", "text"])

    # Keep only the fields used in Notebook 2
    data = data[["combined_text", "label"]]

    return data