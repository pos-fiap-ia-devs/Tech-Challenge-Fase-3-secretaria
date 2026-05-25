"""
Pipeline de fine-tuning LoRA via Apple MLX (otimizado para Apple Silicon M1/M2/M3).
Usa mlx_lm com Meta-Llama-3.1-8B-Instruct quantizado 4-bit.

Execução:
    uv run python fine_tuning/train.py
    uv run python fine_tuning/train.py --iters 200 --batch-size 2
"""

import subprocess
import sys
import os
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modelo MLX pré-quantizado (sem necessidade de conversão)
MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

DATA_DIR = os.path.join(BASE_DIR, "fine_tuning", "data")
ADAPTER_DIR = os.path.join(BASE_DIR, "fine_tuning", "adapters")
FUSED_MODEL_DIR = os.path.join(BASE_DIR, "fine_tuning", "fused_model")


def check_data():
    train = os.path.join(DATA_DIR, "train.jsonl")
    valid = os.path.join(DATA_DIR, "valid.jsonl")
    if not os.path.exists(train) or not os.path.exists(valid):
        print("Dados não encontrados. Executando prepare_data.py...")
        subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "fine_tuning", "prepare_data.py")],
            check=True
        )


def run_training(iters: int, batch_size: int, lora_layers: int, save_every: int):
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL_ID,
        "--train",
        "--data", DATA_DIR,
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--num-layers", str(lora_layers),
        "--save-every", str(save_every),
        "--adapter-path", ADAPTER_DIR,
        "--learning-rate", "1e-4",
        "--val-batches", "5",
        "--max-seq-length", "512",
    ]
    print("\nIniciando fine-tuning LoRA...")
    print(f"Modelo base    : {MODEL_ID}")
    print(f"Iterações      : {iters}")
    print(f"Batch size     : {batch_size}")
    print(f"LoRA layers    : {lora_layers}")
    print("Max seq length : 512 (M1 memory safe)")
    print(f"Adaptadores    : {ADAPTER_DIR}\n")
    subprocess.run(cmd, check=True)


def fuse_model():
    """Funde adaptadores LoRA no modelo base para inferência standalone."""
    os.makedirs(FUSED_MODEL_DIR, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", MODEL_ID,
        "--adapter-path", ADAPTER_DIR,
        "--save-path", FUSED_MODEL_DIR,
    ]
    print("\nFundindo adaptadores LoRA no modelo base...")
    subprocess.run(cmd, check=True)
    print(f"Modelo fundido salvo em: {FUSED_MODEL_DIR}")


def test_inference():
    """Testa o modelo fine-tunado com prompt de exemplo."""
    from fine_tuning.inference import run_inference
    prompt = "Instrução: Classificar risco ginecológico\n\nPaciente: Sangramento intenso e dor pélvica aguda"
    print("\nTestando inferência com modelo fine-tunado...")
    print(f"Prompt: {prompt}\n")
    run_inference(prompt)


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA ConsultasMedica (MLX)")
    parser.add_argument("--iters", type=int, default=100, help="Número de iterações (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lora-layers", type=int, default=4, help="Camadas LoRA (default: 4)")
    parser.add_argument("--save-every", type=int, default=50, help="Salvar adaptadores a cada N iters")
    parser.add_argument("--skip-fuse", action="store_true", help="Pular fusão do modelo após treino")
    args = parser.parse_args()

    check_data()
    run_training(args.iters, args.batch_size, args.lora_layers, args.save_every)

    if not args.skip_fuse:
        fuse_model()

    print("\nFine-tuning concluído.")
    print(f"Adaptadores LoRA: {ADAPTER_DIR}")
    print(f"Modelo fundido  : {FUSED_MODEL_DIR}")


if __name__ == "__main__":
    main()
