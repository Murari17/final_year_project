import torch
import sentencepiece as spm
from pathlib import Path
import os
from model_def import TransformerModel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


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

    # 90/10 train/validation split
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    vocab_size = sp.get_piece_size()
    seq_len = 128        # increased from 64 for better context
    batch_size = 32
    epochs = int(os.getenv("EPOCHS", "50"))
    warmup_epochs = 5
    patience = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device} | vocab: {vocab_size} | tokens: {len(data)}")

    model = TransformerModel(vocab_size=vocab_size, max_seq_len=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # LR warmup for 5 epochs, then cosine decay to near-zero
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # --- Training pass ---
        model.train()
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_data) - seq_len * batch_size, seq_len * batch_size):
            batch_x, batch_y = [], []
            for j in range(batch_size):
                start = i + j * seq_len
                if start + seq_len + 1 > len(train_data):
                    break
                batch_x.append(train_data[start:start + seq_len])
                batch_y.append(train_data[start + 1:start + seq_len + 1])

            if not batch_x:
                continue

            x = torch.stack(batch_x).to(device)
            y = torch.stack(batch_y).to(device)

            pred = model(x)
            loss = loss_fn(pred.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(1, num_batches)

        # --- Validation pass ---
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_data) - seq_len * batch_size, seq_len * batch_size):
                batch_x, batch_y = [], []
                for j in range(batch_size):
                    start = i + j * seq_len
                    if start + seq_len + 1 > len(val_data):
                        break
                    batch_x.append(val_data[start:start + seq_len])
                    batch_y.append(val_data[start + 1:start + seq_len + 1])

                if not batch_x:
                    continue

                x = torch.stack(batch_x).to(device)
                y = torch.stack(batch_y).to(device)
                pred = model(x)
                loss = loss_fn(pred.reshape(-1, vocab_size), y.reshape(-1))
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1:>3} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {lr_now:.6f}")

        # Save best model; stop if no improvement for `patience` epochs
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_out_path)
            print(f"           ↳ Saved best model (val: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} — no improvement for {patience} epochs.")
                break

        scheduler.step()

    print("Training complete")


if __name__ == "__main__":
    main()