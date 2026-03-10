from pathlib import Path
import re

import sentencepiece as spm
import torch

from model_def import TransformerModel


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "to", "for", "and", "or", "on", "with", "by",
}


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def normalize_output_text(text: str) -> str:
    # Clean common OCR artifacts and squash repeated spaces.
    text = text.replace("\u2047", " ")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_retrieved_sentence(sentence: str) -> str:
    sentence = normalize_output_text(sentence)

    # Drop common heading prefixes like "TYPES OF HARDNESS ..." before the real definition.
    parts = sentence.split(":", 1)
    if len(parts) == 2:
        prefix, rest = parts[0].strip(), parts[1].strip()
        if prefix and prefix == prefix.upper() and len(prefix.split()) <= 6:
            sentence = rest

    # Also handle headings without a colon, e.g. "TYPES OF HARDNESS Hardness of ..."
    m = re.match(r"^([A-Z][A-Z\s]{5,})\s+(.+)$", sentence)
    if m:
        heading = m.group(1).strip()
        rest = m.group(2).strip()
        if 1 <= len(heading.split()) <= 6 and len(rest) >= 20:
            sentence = rest

    # Remove trailing list markers like "...: 1." or "... 1."
    sentence = re.sub(r"[:\s]+\d+\.?$", "", sentence).strip()

    return sentence


def score_sentence(question: str, sentence: str) -> int:
    q_words = set(tokenize_words(question))
    if not q_words:
        return 0

    key_words = {w for w in q_words if w not in STOPWORDS} or q_words
    s_words = tokenize_words(sentence)
    if not s_words:
        return 0

    overlap = sum(1 for w in s_words if w in key_words)
    if overlap == 0:
        return 0

    s_text = sentence.lower()
    bonus = 0
    q_lower = question.lower().strip()
    if (q_lower.startswith("what is") or q_lower.startswith("what are")) and (" is " in s_text or " are " in s_text):
        bonus += 2

    keyword_density = overlap / max(1, len(s_words))
    bonus += int(keyword_density * 10)
    return overlap + bonus


def trim_to_question_focus(question: str, sentence: str) -> str:
    sentence = normalize_output_text(sentence)
    q_lower = question.lower().strip().rstrip("?.!")

    # For definition questions, anchor directly to the subject phrase when possible.
    subject = ""
    if q_lower.startswith("what is "):
        subject = q_lower[len("what is "):].strip()
    elif q_lower.startswith("what are "):
        subject = q_lower[len("what are "):].strip()

    if subject:
        idx = sentence.lower().find(subject)
        if idx > 0:
            sentence = sentence[idx:]
            return sentence.strip()

    q_words = [w for w in tokenize_words(question) if w not in STOPWORDS]
    if not q_words:
        return sentence

    s_lower = sentence.lower()
    indices: list[int] = []
    for w in q_words:
        match = re.search(rf"\b{re.escape(w)}\b", s_lower)
        if match:
            indices.append(match.start())

    if not indices:
        return sentence

    first_idx = min(indices)
    # Keep normal starts intact; trim only when there is obvious leading noise.
    if first_idx > 25:
        sentence = sentence[first_idx:]

    return sentence.strip()


def retrieve_best_sentence(question: str, sentences: list[str]) -> str:
    q_words = set(tokenize_words(question))
    if not q_words:
        return ""

    key_words = {w for w in q_words if w not in STOPWORDS} or q_words

    # If possible, only keep lines that include all key terms from the question.
    strict_candidates = [s for s in sentences if all(k in s.lower() for k in key_words)]
    candidates = strict_candidates if strict_candidates else sentences

    best_sentence = ""
    best_score = 0

    for sentence in candidates:
        sentence = clean_retrieved_sentence(sentence)
        if len(sentence) < 20 or len(sentence) > 300:
            continue

        score = score_sentence(question, sentence)
        if score > best_score or (score == best_score and best_sentence and len(sentence) < len(best_sentence)):
            best_score = score
            best_sentence = sentence

    return best_sentence.strip()


def looks_weak_answer(question: str, answer: str) -> bool:
    answer = normalize_output_text(answer)
    if not answer or len(answer) < 30:
        return True

    bad_markers = ["ask:", "answer:", "\u2047", "nps"]
    answer_l = answer.lower()
    if any(marker in answer_l for marker in bad_markers):
        return True

    q_words = set(tokenize_words(question))
    a_words = set(tokenize_words(answer))
    q_key_words = {w for w in q_words if w not in STOPWORDS} or q_words

    return bool(q_key_words and len(q_key_words.intersection(a_words)) == 0)


def generate_answer(
    model: TransformerModel,
    sp: spm.SentencePieceProcessor,
    question: str,
    max_new_tokens: int = 80,
    temperature: float = 0.6,
    top_k: int = 10,
    repetition_penalty: float = 1.15,
) -> str:
    prompt_text = f"Ask: {question}\nAnswer:"
    input_ids = sp.encode(prompt_text)
    generated_ids: list[int] = []
    eos_id = sp.eos_id()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = input_ids + generated_ids
            if len(context_ids) > model.max_seq_len:
                context_ids = context_ids[-model.max_seq_len:]

            x = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0)
            logits = model(x)[0, -1]

            if repetition_penalty > 1.0 and generated_ids:
                for token_id in set(generated_ids[-50:]):
                    logits[token_id] = logits[token_id] / repetition_penalty

            if temperature > 0:
                logits = logits / temperature

            if top_k > 0:
                k = min(top_k, logits.size(0))
                values, indices = torch.topk(logits, k)
                probs = torch.softmax(values, dim=-1)
                sample_idx = torch.multinomial(probs, num_samples=1)
                next_id = int(indices[sample_idx].item())
            else:
                next_id = int(torch.argmax(logits).item())

            if next_id == eos_id:
                break

            generated_ids.append(next_id)

            partial = sp.decode(generated_ids)
            if "Ask:" in partial or "\nAsk:" in partial:
                break
            if "\nAnswer:" in partial:
                break

    answer = sp.decode(generated_ids).strip()
    for marker in ["\nAsk:", "Ask:", "\nAnswer:", "Answer:"]:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()

    if "\n" in answer:
        answer = answer.split("\n", 1)[0].strip()

    if not answer:
        answer = "I need more training data to answer that clearly."

    return normalize_output_text(answer)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    tokenizer_path = base_dir / "tokenizer.model"
    model_path = base_dir / "model.pth"
    sentences_path = base_dir / "data" / "cleaned" / "sentences.txt"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))

    state_dict = torch.load(model_path, map_location="cpu")
    max_seq_len = 256
    if "pos_embedding.weight" in state_dict:
        max_seq_len = int(state_dict["pos_embedding.weight"].shape[0])

    model = TransformerModel(vocab_size=sp.get_piece_size(), max_seq_len=max_seq_len)
    model.load_state_dict(state_dict)
    model.eval()

    prompt = input("Ask: ")
    generated = generate_answer(model, sp, prompt)
    prompt_l = prompt.lower().strip()
    is_definition_question = prompt_l.startswith("what is") or prompt_l.startswith("what are")

    if sentences_path.exists():
        with sentences_path.open("r", encoding="utf-8") as file:
            sentences = [line.strip() for line in file if line.strip()]

        retrieved = retrieve_best_sentence(prompt, sentences)
        generated_score = score_sentence(prompt, generated)
        retrieved_score = score_sentence(prompt, retrieved) if retrieved else 0

        if looks_weak_answer(prompt, generated) or retrieved_score >= generated_score + 2 or (is_definition_question and retrieved_score >= generated_score):
            if retrieved:
                generated = trim_to_question_focus(prompt, retrieved)

    print(generated)


if __name__ == "__main__":
    main()
