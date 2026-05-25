#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- PRÉ-REQUISITOS ---
echo "=== ConsultasMedica - Verificando pré-requisitos ==="

if ! command -v uv &>/dev/null; then
    echo "ERRO: 'uv' não encontrado. Instale em: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "AVISO: fine-tuning via MLX requer Apple Silicon (arm64)."
    echo "         Detectado: $(uname -m). O treino pode falhar."
    printf "  Continuar mesmo assim? [s/N]: "
    read -r CONFIRM
    [[ "$CONFIRM" =~ ^[sS]$ ]] || exit 0
fi

if ! command -v ollama &>/dev/null; then
    echo "AVISO: 'ollama' não encontrado. O dashboard não funcionará sem ele."
    echo "       Instale em: https://ollama.com/download"
else
    if ! ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        echo "AVISO: modelo 'nomic-embed-text' não encontrado."
        echo "       Execute: ollama pull nomic-embed-text"
    fi
fi

echo ""
echo "=== ConsultasMedica - Pipeline de Fine-tuning ==="
echo "Diretório: $SCRIPT_DIR"
echo ""

LLM_MODEL="llama3.1:8b"
echo "[Modelo] llama3.1:8b selecionado."
if ! ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
    echo "[Modelo] Baixando llama3.1:8b (pode demorar)..."
    ollama pull llama3.1:8b
else
    echo "[Modelo] llama3.1:8b já disponível."
fi

# Atualiza main.py com o modelo escolhido
sed -i '' "s/OllamaLLM(model=\"[^\"]*\"/OllamaLLM(model=\"$LLM_MODEL\"/" main.py
echo "[Modelo] main.py atualizado → $LLM_MODEL"

echo ""

# --- MENU: IMPORTAÇÃO MEDQUAD ---
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Importar dados do MedQuAD antes de treinar?"
echo "  (saída: data/medquad.csv)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1) Importar do GitHub  (~2-3 min, inglês)"
echo "  2) Pular               (usar data/medquad.csv existente)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  Escolha [1/2]: "
read -r MEDQUAD_CHOICE

case "$MEDQUAD_CHOICE" in
    1)
        echo ""
        echo "[MedQuAD] Importando do GitHub → data/medquad.csv ..."
        uv run python fine_tuning/importar_medquad.py --max-per-category 80
        ;;
    *)
        echo ""
        if [[ ! -f "data/medquad.csv" ]]; then
            echo "ERRO: data/medquad.csv não encontrado. Escolha opção 1."
            exit 1
        fi
        echo "[MedQuAD] Usando data/medquad.csv existente."
        ;;
esac

echo ""

# --- MENU: ITERAÇÕES ---
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Quantas iterações de treino?"
echo "  (M1 Pro 16GB: peak mem ~2.5 GB — sem limite de RAM)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1) 100 iterações  (~5 min)   rápido"
echo "  2) 300 iterações  (~15 min)  padrão"
echo "  3) 500 iterações  (~25 min)  recomendado  [sweet spot]"
echo "  4) 1000 iterações (~50 min)  avançado     [risco overfitting]"
echo "  5) Custom — digitar número"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  Escolha [1/2/3/4/5]: "
read -r ITERS_CHOICE

case "$ITERS_CHOICE" in
    1) TRAIN_ITERS=100 ;;
    2) TRAIN_ITERS=300 ;;
    3) TRAIN_ITERS=500 ;;
    4) TRAIN_ITERS=1000 ;;
    5)
        printf "  Número de iterações: "
        read -r TRAIN_ITERS
        if ! [[ "$TRAIN_ITERS" =~ ^[0-9]+$ ]] || [[ "$TRAIN_ITERS" -lt 10 ]]; then
            echo "  Valor inválido. Usando 300."
            TRAIN_ITERS=300
        fi
        ;;
    *) TRAIN_ITERS=300 ; echo "  Opção inválida. Usando 300 iterações." ;;
esac

echo ""

# --- LIMPEZA ---
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Limpando artefatos anteriores..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
rm -rf data/faiss_index
rm -f  fine_tuning/data/train.jsonl fine_tuning/data/valid.jsonl
rm -f  fine_tuning/adapters/0000*_adapters.safetensors \
       fine_tuning/adapters/adapters.safetensors
rm -rf fine_tuning/fused_model
python3 -c "
import sqlite3, os
db = os.path.join('data', 'prontuarios.db')
if os.path.exists(db):
    conn = sqlite3.connect(db)
    conn.execute('DELETE FROM atendimentos')
    conn.execute('DELETE FROM sqlite_sequence WHERE name=\"atendimentos\"')
    conn.commit()
    conn.close()
"
echo "[Limpeza] FAISS index, train/valid JSONL, adapters, fused_model e audit DB removidos."
echo ""

# --- PIPELINE ---
echo "[1/5] Instalando dependências..."
uv sync

echo ""
echo "[2/5] Preparando dados (data/medquad.csv → fine_tuning/data/train.jsonl + valid.jsonl)..."
uv run python fine_tuning/preparar_dados.py

echo ""
echo "[3/5] Treinando modelo (LoRA / MLX) — $TRAIN_ITERS iterações..."
uv run python fine_tuning/treinar_modelo.py --iters "$TRAIN_ITERS" --batch-size 2

echo ""
echo "[4/5] Testando inferência..."
uv run python fine_tuning/testar_inferencia.py "Sangramento intenso e dor pélvica aguda"

echo ""
echo "[5/5] Avaliando modelo..."
uv run python fine_tuning/avaliar_modelo.py

echo ""
echo "=== Pipeline concluído ==="
echo "Modelo LLM : $LLM_MODEL"
echo ""
echo "Iniciando dashboard Streamlit..."
uv run streamlit run main.py
