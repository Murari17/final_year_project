from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize


def ensure_tokenizer_resources() -> None:
    # Newer NLTK versions may require both resources for sentence tokenization.
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


base_dir = Path(__file__).resolve().parent
input_path = base_dir / "data" / "cleaned" / "chemistry_clean.txt"
output_path = base_dir / "data" / "cleaned" / "sentences.txt"

ensure_tokenizer_resources()

text = input_path.read_text(encoding="utf-8")
sentences = sent_tokenize(text)

output_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")