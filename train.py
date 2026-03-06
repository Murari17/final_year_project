import torch
import sentencepiece as spm
from pathlib import Path
import os
from model_def import TransformerModel

def main() -> None:
    sp = spm.SentencePieceProcessor()
    base_dir = Path(__file__).resolve().parent
    tokenizer_path = base_dir / "tokenizer.model"
    dataset_path = base_dir / "dataset" / "training_data.txt"
    model_out_path = base_dir / "model.pth"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    sp.load(str(tokenizer_path))

    text = dataset_path.read_text(encoding="utf-8")
    records = [r.strip() for r in text.split("\n\n") if r.strip()]
    tokens: list[int] = []
    eos_id = sp.eos_id()

    for record in records:
        tokens.extend(sp.encode(record))
        if eos_id != -1:
            tokens.append(eos_id)

    data = torch.tensor(tokens, dtype=torch.long)

    vocab_size = sp.get_piece_size()
    seq_len = 64
    epochs = int(os.getenv("EPOCHS", "50"))

    model = TransformerModel(vocab_size=vocab_size, max_seq_len=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i:i + seq_len].unsqueeze(0)
            y = data[i + 1:i + seq_len + 1].unsqueeze(0)

            pred = model(x)
            loss = loss_fn(pred.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print("Epoch", epoch + 1, "Loss:", total_loss)

    torch.save(model.state_dict(), model_out_path)
    print("Training complete")


if __name__ == "__main__":
    main()