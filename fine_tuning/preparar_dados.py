"""
Prepara e converte dataset CSV (medquad.csv) para formato JSONL compatível com mlx_lm LoRA.
Gera train.jsonl (80%) e valid.jsonl (20%) em fine_tuning/data/.

Trunca exemplos longos antes do treino para evitar alocação de buffers grandes
e o warning "Some sequences are longer than N tokens".
"""

import csv
import json
import random
import os

MAX_SEQ_LENGTH = 512
# ~3.5 chars/token para português/inglês misturado (conservador)
CHARS_PER_TOKEN = 3.5
MAX_SEQ_CHARS = int(MAX_SEQ_LENGTH * CHARS_PER_TOKEN)

SYSTEM_PROMPT = (
    "You are ConsultasMedica, a clinical assistant specialized in women's health. "
    "Follow INCA, FEBRASGO, WHO and Ministry of Health guidelines. "
    "Never prescribe medications or dosages. Always recommend medical evaluation when necessary."
)

# Chars consumidos por overhead fixo (tokens especiais + system prompt + headers)
_OVERHEAD_CHARS = len(
    f"<|begin_of_text|>"
    f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
    f"<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"
    f"<|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>"
)


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Trunca no limite de sentença mais próximo antes de max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in (".\n", ". ", "!\n", "! ", "?\n", "? "):
        pos = truncated.rfind(sep)
        if pos > max_chars * 0.5:
            return truncated[: pos + 1]
    return truncated


def format_example(item: dict) -> str:
    instrucao = item.get("focus_area") or "Orientação clínica"
    user_msg = f"Instrução: {instrucao}\n\nPaciente: {item['question']}"

    # Budget disponível para conteúdo variável
    content_budget = MAX_SEQ_CHARS - _OVERHEAD_CHARS - len(user_msg)
    answer = _truncate_at_sentence(item["answer"], max(content_budget, 80))

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
    )


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "medquad.csv")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    # Drop rows with empty question or answer
    raw = [r for r in raw if r.get("question", "").strip() and r.get("answer", "").strip()]

    truncated_count = 0
    examples = []
    for item in raw:
        original_len = len(item["answer"])
        text = format_example(item)
        examples.append({"text": text})
        if len(item["answer"]) < original_len:
            truncated_count += 1

    random.seed(42)
    random.shuffle(examples)

    split = int(len(examples) * 0.8)
    train_set = examples[:split]
    valid_set = examples[split:]

    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for ex in valid_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    max_chars = max(len(ex["text"]) for ex in examples)
    print(f"Fonte             : medquad.csv ({len(raw)} linhas válidas)")
    print(f"Dataset preparado : {len(train_set)} treino | {len(valid_set)} validação")
    print(f"Exemplos truncados: {truncated_count}/{len(examples)}")
    print(f"Maior sequência   : {max_chars} chars (~{int(max_chars / CHARS_PER_TOKEN)} tokens estimados)")
    print(f"Arquivos salvos em: {output_dir}")


if __name__ == "__main__":
    main()
