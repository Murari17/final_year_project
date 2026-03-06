from pathlib import Path

from pdfminer.high_level import extract_text

BASE_DIR = Path(__file__).resolve().parent
raw_dir = BASE_DIR / "data" / "raw"
extracted_dir = BASE_DIR / "data" / "extracted"

pdf_files = sorted(raw_dir.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in: {raw_dir}")

extracted_dir.mkdir(parents=True, exist_ok=True)

for old_txt in extracted_dir.glob("*.txt"):
    old_txt.unlink()

for pdf_path in pdf_files:
    output_txt = extracted_dir / f"{pdf_path.stem}.txt"
    text = extract_text(str(pdf_path))
    with output_txt.open("w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted: {pdf_path.name} -> {output_txt.name}")

print(f"Extraction complete for {len(pdf_files)} PDF file(s).")