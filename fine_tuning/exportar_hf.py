"""
Exporta modelo ConsultasMedica fine-tunado para HuggingFace Hub.
Funde adaptadores LoRA no modelo base e faz upload.

Pré-requisito:
    uv run huggingface-cli login

Execução:
    uv run python fine_tuning/export_to_hf.py --hf-user SEU_USERNAME
    uv run python fine_tuning/export_to_hf.py --hf-user SEU_USERNAME --repo-name meu-modelo
"""

import subprocess
import sys
import os
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ADAPTER_DIR = os.path.join(BASE_DIR, "fine_tuning", "adapters")
FUSED_MODEL_DIR = os.path.join(BASE_DIR, "fine_tuning", "fused_model")


def check_login():
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"HuggingFace autenticado como: {user['name']}")
        return user["name"]
    except Exception:
        print("Erro: não autenticado no HuggingFace.")
        print("Execute: uv run huggingface-cli login")
        sys.exit(1)


def check_adapters():
    adapter_file = os.path.join(ADAPTER_DIR, "adapters.safetensors")
    if not os.path.exists(adapter_file):
        print(f"Erro: adaptadores não encontrados em {ADAPTER_DIR}")
        print("Execute o fine-tuning primeiro: uv run python fine_tuning/train.py")
        sys.exit(1)


def fuse_and_upload(repo_id: str):
    os.makedirs(FUSED_MODEL_DIR, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", MODEL_ID,
        "--adapter-path", ADAPTER_DIR,
        "--save-path", FUSED_MODEL_DIR,
        "--upload-repo", repo_id,
    ]
    print(f"\nFundindo adaptadores e enviando para: {repo_id}")
    print("Isso pode levar alguns minutos...\n")
    subprocess.run(cmd, check=True)


def create_model_card(repo_id: str, hf_user: str):
    from huggingface_hub import HfApi
    card_content = f"""---
language:
- pt
license: llama3.1
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
tags:
- medical
- portuguese
- women-health
- fine-tuned
- mlx
- lora
- saude-da-mulher
---

# ConsultasMedica - Assistente Clínico de Saúde da Mulher

Modelo fine-tunado com LoRA (MLX / Apple Silicon) para assistência clínica em saúde da mulher.

## Modelo Base
`meta-llama/Meta-Llama-3.1-8B-Instruct` (via `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`)

## Fine-tuning
- **Método:** LoRA (Low-Rank Adaptation)
- **Framework:** [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple Silicon / MLX)
- **Dataset:** 50 exemplos sintéticos de saúde da mulher
- **Protocolos:** INCA, FEBRASGO, OMS, Ministério da Saúde (Brasil)

## Domínios
- Classificação de risco ginecológico (VERMELHO/AMARELO/VERDE)
- Orientações pré-natais
- Detecção de violência doméstica
- Rastreamento de câncer (mama e colo do útero)
- ISTs, contracepção, menopausa, saúde reprodutiva

## Limitações e Ética
- **Nunca prescreve medicamentos ou dosagens**
- Uso exclusivo como suporte a profissionais de saúde
- Não substitui avaliação médica presencial
- Desenvolvido para fins acadêmicos (FIAP Pós Tech IA para Devs - Fase 3)

## Uso
```python
from mlx_lm import load, generate

model, tokenizer = load("{repo_id}")
prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nVocê é ConsultasMedica, assistente clínico especializado em saúde da mulher.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nInstrução: Classificar risco\\n\\nPaciente: Dor pélvica aguda e febre<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```
"""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Model card criado em: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Exportar ConsultasMedica para HuggingFace Hub")
    parser.add_argument("--hf-user", type=str, help="Seu username no HuggingFace")
    parser.add_argument("--repo-name", type=str, default="consultas-medica-saude-mulher",
                        help="Nome do repositório (default: consultas-medica-saude-mulher)")
    args = parser.parse_args()

    hf_user = check_login()

    if args.hf_user and args.hf_user != hf_user:
        print(f"Aviso: autenticado como '{hf_user}', usando esse username.")

    repo_id = f"{hf_user}/{args.repo_name}"

    check_adapters()

    print(f"\n=== Exportando ConsultasMedica para HuggingFace ===")
    print(f"Repositório : {repo_id}")
    print(f"Adaptadores : {ADAPTER_DIR}")
    print(f"Modelo base : {MODEL_ID}\n")

    fuse_and_upload(repo_id)
    create_model_card(repo_id, hf_user)

    print(f"\n=== Exportação concluída ===")
    print(f"Modelo disponível em: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
