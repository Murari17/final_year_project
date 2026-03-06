import sentencepiece as spm
from pathlib import Path


base_dir = Path(__file__).resolve().parent
input_path = base_dir / "dataset" / "training_data.txt"
model_prefix = base_dir / "tokenizer"

if not input_path.exists():
    raise FileNotFoundError(f"Training data not found: {input_path}")

spm.SentencePieceTrainer.train(
    input=str(input_path),
    model_prefix=str(model_prefix),
    vocab_size=4000,
    model_type='bpe',
    hard_vocab_limit=False,
)

print("Tokenizer trained")