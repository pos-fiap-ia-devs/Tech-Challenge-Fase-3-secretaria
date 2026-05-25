"""
Inferência com modelo ConsultasMedica fine-tunado (MLX).
Usa adaptadores LoRA ou modelo fundido se disponível.

Execução:
    uv run python fine_tuning/inference.py "Dor pélvica intensa e febre"
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ADAPTER_DIR = os.path.join(BASE_DIR, "fine_tuning", "adapters")
FUSED_MODEL_DIR = os.path.join(BASE_DIR, "fine_tuning", "fused_model")

SYSTEM_PROMPT = (
    "You are ConsultasMedica, a clinical assistant specialized in women's health. "
    "Follow INCA, FEBRASGO, WHO and Ministry of Health guidelines. "
    "Never prescribe medications or dosages. Always recommend medical evaluation when necessary."
)


def build_prompt(user_input: str) -> str:
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def run_inference(user_input: str, max_tokens: int = 256):
    from mlx_lm import load, generate

    fused_exists = os.path.exists(os.path.join(FUSED_MODEL_DIR, "config.json"))
    adapters_exist = os.path.exists(os.path.join(ADAPTER_DIR, "adapters.safetensors"))

    if fused_exists:
        print(f"Carregando modelo fundido: {FUSED_MODEL_DIR}")
        model, tokenizer = load(FUSED_MODEL_DIR)
    elif adapters_exist:
        print(f"Carregando modelo base + adaptadores LoRA: {ADAPTER_DIR}")
        model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_DIR)
    else:
        print(f"Fine-tuning não encontrado. Carregando modelo base: {MODEL_ID}")
        model, tokenizer = load(MODEL_ID)

    prompt = build_prompt(user_input)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    print(f"\nResposta ConsultasMedica:\n{response}")
    return response


if __name__ == "__main__":
    if len(sys.argv) < 2:
        user_input = "Instrução: Classificar risco ginecológico\n\nPaciente: Sangramento intenso e dor pélvica aguda"
    else:
        user_input = " ".join(sys.argv[1:])

    run_inference(user_input)
