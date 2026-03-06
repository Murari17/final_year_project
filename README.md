# Chemistry QA Trainer (Local, Multi-PDF)

This project builds a local QA system from one or more chemistry PDFs.
It does not use any external LLM API.

## What It Does

1. Extracts text from all PDFs in `data/raw/`
2. Cleans and merges extracted text
3. Splits text into sentences
4. Builds synthetic QA-style training pairs
5. Trains a SentencePiece tokenizer
6. Trains a small local Transformer model
7. Answers questions using generation with retrieval fallback

## Project Structure

- `run_pipeline.py` - one-command end-to-end pipeline
- `textextraction.py` - extract text from all PDFs in `data/raw/`
- `clean_text.py` - clean and merge extracted text
- `sentence_split.py` - split cleaned corpus into sentences
- `create_dataset.py` - build QA-style training dataset
- `train_tokenizer.py` - train SentencePiece tokenizer
- `model_def.py` - Transformer model definition
- `train.py` - train model and save `model.pth`
- `generate.py` - interactive Q&A
- `data/raw/` - source PDFs
- `data/extracted/` - generated extracted text
- `data/cleaned/` - generated cleaned corpus and sentence list
- `dataset/training_data.txt` - generated training pairs

## Requirements

- Python 3.11+ (3.13 also works in this repo)
- Windows PowerShell (commands below are PowerShell)

## Setup

```powershell
# from project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Start (One Command)

Put any number of PDFs in `data/raw/`, then run:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --epochs 30
```

This runs extraction, cleaning, sentence split, dataset build, tokenizer training, and model training.

## Ask Questions

```powershell
.\.venv\Scripts\python.exe generate.py
```

Example:

```text
Ask: What is hardness of water?
```

## Optional Pipeline Flags

```powershell
# run preprocessing and tokenizer only
.\.venv\Scripts\python.exe run_pipeline.py --skip-train

# run preprocessing and model training but reuse existing tokenizer
.\.venv\Scripts\python.exe run_pipeline.py --skip-tokenizer --epochs 30

# set custom training epochs
.\.venv\Scripts\python.exe run_pipeline.py --epochs 50
```

## Manual Step-by-Step (Optional)

```powershell
.\.venv\Scripts\python.exe textextraction.py
.\.venv\Scripts\python.exe clean_text.py
.\.venv\Scripts\python.exe sentence_split.py
.\.venv\Scripts\python.exe create_dataset.py
.\.venv\Scripts\python.exe train_tokenizer.py
$env:EPOCHS='30'; .\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe generate.py
```

## Notes on Output Quality

- Better quality depends mainly on better and larger PDF data.
- More epochs can help, but data quality matters more.
- `generate.py` includes retrieval fallback from `data/cleaned/sentences.txt` when generated text is weak.

## Troubleshooting

### 1) `ModuleNotFoundError`
Install dependencies inside `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2) SentencePiece vocab-size errors
`train_tokenizer.py` already uses `hard_vocab_limit=False`, so it should adapt to corpus size.

### 3) PyTorch warning about NumPy
Install NumPy:

```powershell
python -m pip install numpy
```

### 4) No output / poor answers
- Ensure PDFs are in `data/raw/`
- Re-run full pipeline
- Increase epochs
- Add more domain-relevant PDFs

## Reproducibility Tip

Always run commands from project root (`c:\Users\murar\Desktop\fyp`) with `.venv` activated.
