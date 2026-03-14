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

    # Drop leading bullets/numbering fragments.
    sentence = re.sub(r"^(?:[-*•➢]|\d+[.)])\s*", "", sentence).strip()

    # Drop common heading prefixes like "TYPES OF HARDNESS ..." before the real definition.
    parts = sentence.split(":", 1)
    if len(parts) == 2:
        prefix, rest = parts[0].strip(), parts[1].strip()
        if prefix and len(rest) >= 20 and not any(c in prefix for c in ".?!;"):
            prefix_words = prefix.split()
            # Strip if: ALL-CAPS heading ("TYPES OF HARDNESS") or short title-case label
            # ("Causes", "Causes of Hardness", "Note") with ≤ 5 words.
            is_heading_label = (
                prefix == prefix.upper()
                or (len(prefix_words) <= 5 and all(w[0].isupper() for w in prefix_words if len(w) > 2))
            )
            if is_heading_label:
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
    subject = ""
    if q_lower.startswith("what is "):
        subject = q_lower[len("what is "):].strip().rstrip("?.!")
    elif q_lower.startswith("what are "):
        subject = q_lower[len("what are "):].strip().rstrip("?.!")

    # Strong preference for direct definition form at the sentence start.
    if subject:
        if re.match(rf"^{re.escape(subject)}\s+is\b", s_text):
            bonus += 5
        elif re.match(rf"^(?:a|an|the)\s+{re.escape(subject)}\s+is\b", s_text):
            bonus += 4  # Leading article is fine, still a definition opener.
        if re.match(rf"^{re.escape(subject)}\s+are\b", s_text):
            bonus += 3
        elif re.match(rf"^(?:a|an|the)\s+{re.escape(subject)}\s+are\b", s_text):
            bonus += 2
        # Also reward other clear definitional verbs at the sentence start.
        definitional_verbs = ("deals with", "refers to", "involves", "consists of", "means", "describes")
        for dv in definitional_verbs:
            if re.match(rf"^{re.escape(subject)}\s+{re.escape(dv)}\b", s_text):
                bonus += 4
                break

        # Reward sentences where the asked subject is actually predicated.
        subj_predicate = rf"\b{re.escape(subject)}\b\s+(?:is|are|deals\s+with|refers\s+to|involves|consists\s+of|means|describes)\b"
        if re.search(subj_predicate, s_text):
            bonus += 3
        else:
            # Subject appears only as a mention (e.g., "pollution control") — weak for definitions.
            bonus -= 2

    # Penalize pure classification sentences: "X is mainly/of two types" — not a definition.
    if re.search(r"\bis (?:mainly|primarily|generally|largely|of)?\s*\w*\s*types?\b", s_text):
        bonus -= 3

    if q_lower.startswith("what is") and " is " in s_text:
        bonus += 2
    if q_lower.startswith("what are") and " are " in s_text:
        bonus += 2

    # Penalize noisy list fragments and OCR-heavy lines.
    if any(tok in sentence for tok in ("➢", "•", "->")):
        bonus -= 3
    if re.search(r"\b(i|ii|iii|iv)\b", s_text):
        bonus -= 1
    if len(sentence) > 220:
        bonus -= 2

    # Penalize naming constructs ("X is called Y" names a type, doesn't define the subject).
    if " is called " in s_text or " are called " in s_text:
        bonus -= 5

    # Penalize meta-textbook sentences (about the book, not about the concept).
    meta_patterns = ["discussed in", "described in", "covered in", "dealt with", "mentioned in", "referred to in"]
    if any(p in s_text for p in meta_patterns):
        bonus -= 6

    # Penalize consequence/connector sentence openers (not definitional).
    consequence_openers = ("thus,", "hence,", "therefore,", "however,", "moreover,", "furthermore,", "so,")
    if s_text.startswith(consequence_openers):
        bonus -= 4

    # Penalize sentences starting with "this" (usually a reference, not a definition).
    if s_text.startswith("this "):
        bonus -= 3

    # Penalize measurement/expression sentences — they describe units, not concepts.
    if re.search(r"\bis (?:expressed|measured|calculated|given|represented|denoted|indicated)\b", s_text):
        bonus -= 4

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
        # Try to find the subject in a clear definitional role so we can trim any
        # leading OCR noise and land at the natural definition start.
        definitional_verb_starts = (" is ", " are ", " deals with ", " refers to ", " involves ", " consists of ", " means ", " describes ")
        for dv in definitional_verb_starts:
            pattern = subject + dv.rstrip()
            idx = sentence.lower().find(pattern)
            if idx > 0:
                # Guard: skip if the subject is a prepositional object
                # e.g. "cause of pollution is" — "of" precedes "pollution".
                words_before = sentence[:idx].rstrip().split()
                if words_before and words_before[-1].lower() in ("of", "for", "from", "to", "in", "with", "by", "about", "on", "at"):
                    continue
                sentence = sentence[idx:]
                return sentence.strip()
        # Fallback: only trim to raw subject position when definitionally safe.
        idx = sentence.lower().find(subject)
        if idx > 0:
            # Same preposition guard.
            words_before = sentence[:idx].rstrip().split()
            preceded_by_prep = words_before and words_before[-1].lower() in ("of", "for", "from", "to", "in", "with", "by", "about", "on", "at")
            if not preceded_by_prep:
                tail_after = sentence.lower()[idx + len(subject):]
                if tail_after.startswith(" is ") or tail_after.startswith(" are "):
                    sentence = sentence[idx:]
                    return sentence.strip()

        # For definition questions we don't apply the generic keyword trim — returning
        # the full cleaned sentence is safer than producing a wrongly anchored fragment.
        return sentence.strip()

    # Generic keyword trim for non-definition questions (factual, how-many, etc.).
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
    q_lower = question.lower().strip().rstrip("?.!")

    # Definition questions: prefer local definition-style lines first.
    subject = ""
    if q_lower.startswith("what is "):
        subject = q_lower[len("what is "):].strip()
    elif q_lower.startswith("what are "):
        subject = q_lower[len("what are "):].strip()

    subject_variants: list[str] = []
    if subject:
        subject_variants.append(subject)
        if subject.startswith("a "):
            subject_variants.append(subject[2:].strip())
        if subject.startswith("an "):
            subject_variants.append(subject[3:].strip())
        if subject.startswith("the "):
            subject_variants.append(subject[4:].strip())

    DEFINITIONAL_VERBS = (" is ", " are ", " deals with ", " refers to ", " involves ", " consists of ", " means ", " describes ")
    definition_candidates: list[str] = []
    if subject_variants:
        for s in sentences:
            s_l = s.lower()
            if not any(dv in s_l for dv in DEFINITIONAL_VERBS):
                continue
            # Skip naming constructs — they label types, not define concepts.
            if " is called " in s_l or " are called " in s_l:
                continue
            # Find where the subject first appears.
            subj_pos = -1
            for v in subject_variants:
                if v and v in s_l:
                    subj_pos = s_l.find(v)
                    break
            if subj_pos == -1:
                continue
            # Require the subject to appear in the first 55% of the sentence so we
            # don't pick up sentences where the topic is mentioned only in a clause.
            if subj_pos <= len(s_l) * 0.55:
                definition_candidates.append(s)

    # If possible, only keep lines that include all key terms from the question.
    strict_candidates = [s for s in sentences if all(k in s.lower() for k in key_words)]

    if definition_candidates:
        candidates = definition_candidates
    elif strict_candidates:
        candidates = strict_candidates
    else:
        candidates = sentences

    best_sentence = ""
    best_score = 0

    for sentence in candidates:
        sentence = clean_retrieved_sentence(sentence)
        if len(sentence) < 20 or len(sentence) > 450:
            continue

        score = score_sentence(question, sentence)
        if score > best_score or (score == best_score and best_sentence and len(sentence) > len(best_sentence)):
            best_score = score
            best_sentence = sentence

    return best_sentence.strip()


def is_factual_question(q: str) -> bool:
    """Who/where/when/which/how-many questions — retrieval always wins."""
    return q.lower().strip().startswith(("how many", "which", "when", "where", "who"))


def is_definition_question(q: str) -> bool:
    """Definition-style questions — prefer retrieval when it scores well."""
    return q.lower().strip().startswith(("what is", "what are", "define", "explain", "describe"))


def is_direct_definition_sentence(question: str, sentence: str) -> bool:
    """True when sentence starts with the asked subject in a definitional form."""
    q = question.lower().strip().rstrip("?.!")
    s = normalize_output_text(sentence).lower()

    subject = ""
    if q.startswith("what is "):
        subject = q[len("what is "):].strip()
    elif q.startswith("what are "):
        subject = q[len("what are "):].strip()

    if not subject:
        return False

    subject_forms = [subject]
    if subject.startswith("a "):
        subject_forms.append(subject[2:].strip())
    if subject.startswith("an "):
        subject_forms.append(subject[3:].strip())
    if subject.startswith("the "):
        subject_forms.append(subject[4:].strip())

    definitional_starts = (
        " is ", " are ", " deals with ", " refers to ", " involves ", " consists of ", " means ", " describes "
    )
    for form in subject_forms:
        if not form:
            continue
        for tail in definitional_starts:
            if s.startswith(form + tail):
                return True
            if s.startswith("a " + form + tail) or s.startswith("an " + form + tail) or s.startswith("the " + form + tail):
                return True
    return False


def enforce_question_grammar(question: str, answer: str) -> str:
    """Fix obvious singular/plural mismatch and capitalise the first letter."""
    q = question.lower().strip().rstrip("?.!")
    a = normalize_output_text(answer)
    if q.startswith("what is "):
        subject = q[len("what is "):].strip()
        if subject and a.lower().startswith(subject + " are "):
            a = a[:len(subject)] + " is " + a[len(subject) + len(" are "):]
    # Capitalise first letter.
    if a:
        a = a[0].upper() + a[1:]
    return a


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
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = input_ids + generated_ids
            if len(context_ids) > model.max_seq_len:
                context_ids = context_ids[-model.max_seq_len:]

            x = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0).to(device)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    max_seq_len = 512
    if "pos_embedding.weight" in state_dict:
        max_seq_len = int(state_dict["pos_embedding.weight"].shape[0])

    model = TransformerModel(vocab_size=sp.get_piece_size(), max_seq_len=max_seq_len)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "The saved model checkpoint does not match the current model architecture. "
            "Retrain the model with `python run_pipeline.py --epochs 50` or `python train.py`, "
            "then run generate.py again."
        ) from exc
    model.to(device)
    model.eval()

    # Load local sentence index (fully offline keyword retrieval).
    sentences: list[str] = []
    if sentences_path.exists():
        with sentences_path.open("r", encoding="utf-8") as file:
            sentences = [line.strip() for line in file if line.strip()]

    prompt = input("Ask: ")
    generated = generate_answer(model, sp, prompt)

    if sentences:
        retrieved = retrieve_best_sentence(prompt, sentences)

        generated_score = score_sentence(prompt, generated)
        retrieved_score = score_sentence(prompt, retrieved) if retrieved else 0

        # Factual questions (who/where/when/which/how many) -> always use retrieval
        if is_factual_question(prompt):
            if retrieved:
                generated = trim_to_question_focus(prompt, retrieved)
        # Definition / open-ended -> hybrid decision
        elif looks_weak_answer(prompt, generated) or retrieved_score >= generated_score + 2 or (is_definition_question(prompt) and retrieved_score >= generated_score):
            if retrieved:
                if is_definition_question(prompt):
                    # For "what is/are" questions, only trust retrieval when it
                    # looks like an actual definition of the asked subject.
                    if is_direct_definition_sentence(prompt, retrieved) or retrieved_score >= generated_score + 4:
                        generated = trim_to_question_focus(prompt, retrieved)
                else:
                    generated = trim_to_question_focus(prompt, retrieved)

    generated = enforce_question_grammar(prompt, generated)

    print(generated)


if __name__ == "__main__":
    main()
