from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_step(script_name: str, extra_env: dict[str, str] | None = None) -> None:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"\n=== Running: {script_name} ===")
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed ({script_name}) with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full multi-PDF training pipeline.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for train.py")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer training")
    args = parser.parse_args()

    # Preprocessing and dataset build
    run_step("textextraction.py")
    run_step("clean_text.py")
    run_step("sentence_split.py")
    run_step("create_dataset.py")

    if not args.skip_tokenizer:
        run_step("train_tokenizer.py")

    if not args.skip_train:
        run_step("train.py", extra_env={"EPOCHS": str(args.epochs)})

    print("\nPipeline complete.")
    print("Run: python generate.py")


if __name__ == "__main__":
    main()
