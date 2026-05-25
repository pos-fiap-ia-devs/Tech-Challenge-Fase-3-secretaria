# 🩺 ConsultasMedica - Assistente Clínico de Saúde da Mulher
### FIAP | Pós Tech IA para Devs - Tech Challenge Fase 3

Este projeto apresenta um Assistente Clínico Inteligente desenvolvido com foco em **Saúde da Mulher**, utilizando tecnologias de IA generativa de ponta com execução **100% local** para garantir a máxima privacidade dos dados sensíveis (LGPD).

---

## 🚀 Visão Geral
O **ConsultasMedica** é uma solução orquestrada que realiza triagens clínicas automáticas, responde dúvidas de saúde da mulher com base em evidências e garante conformidade ética através de filtros de segurança em tempo real.

### Principais Diferenciais:
- **Orquestração com LangGraph:** Fluxo de decisão não linear que diferencia rotina, urgência e casos especializados.
- **RAG (Retrieval Augmented Generation):** Consulta base MedQuAD/NIH via FAISS para contextualizar respostas.
- **Privacidade Total (Air-Gapped):** Execução local via Ollama (Llama 3.1 8B), sem envio de dados para a nuvem.
- **Filtro Ético de Prescrição:** Bloqueio automático de tentativas de dosagem ou receitas médicas por IA.

---

## 🛠️ Estrutura do Projeto
```text
├── data/
│   ├── prontuarios.db               # SQLite — audit de atendimentos
│   ├── pacientes_sinteticos.csv     # Dataset anonimizado de pacientes
│   └── medquad.csv                  # Dataset MedQuAD/NIH (~16k exemplos)
├── docs/
│   ├── diagrama_langgraph.md        # Diagrama Mermaid do fluxo
│   ├── diagrama_langgraph.png       # Diagrama PNG gerado
│   └── relatorio_tecnico.md         # Relatório técnico completo
├── fine_tuning/
│   ├── adapters/                    # Adaptadores LoRA treinados
│   ├── fused_model/                 # Modelo fundido (LoRA + base)
│   ├── data/                        # train.jsonl / valid.jsonl
│   ├── importar_medquad.py          # Importação MedQuAD do GitHub
│   ├── preparar_dados.py            # Preprocessing CSV → JSONL + split
│   ├── treinar_modelo.py            # Fine-tuning LoRA via MLX
│   ├── testar_inferencia.py         # Inferência com modelo fine-tunado
│   ├── avaliar_modelo.py            # Avaliação com cenários clínicos
│   └── exportar_hf.py               # Export para HuggingFace Hub
├── src/
│   ├── logger.py                    # Logger centralizado (console + arquivo)
│   ├── db/
│   │   └── prontuarios.py           # SQLite — prontuários e audit de atendimentos
│   ├── engine/
│   │   ├── grafo_clinico.py         # Compilação do grafo LangGraph
│   │   ├── etapas_clinicas.py       # 6 etapas do grafo + lógica clínica
│   │   └── estado_atendimento.py    # EstadoAtendimento (TypedDict)
│   └── rag/
│       └── busca_medquad.py         # RAG Engine (FAISS/MedQuAD)
├── logs/
│   └── consultas_medica.log         # Logs de auditoria (rotação diária)
├── main.py                          # Interface Streamlit
├── generate_diagram.py              # Gerador do diagrama LangGraph
└── run_finetune.sh                  # Pipeline completo com menu interativo
```

---

## ⚙️ Como Executar

### Pré-requisitos

| Requisito | Versão | Link |
| --- | --- | --- |
| Python | ≥ 3.12 | via `uv` |
| uv | qualquer | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| Ollama | qualquer | [ollama.com/download](https://ollama.com/download) |
| Apple Silicon | M1/M2/M3 | obrigatório **apenas** para fine-tuning (MLX) |

### Passo 1: Clonar o Repositório
```bash
git clone git@github.com:pos-fiap-ia-devs/Tech-Challenge-Fase-3.git
cd Tech-Challenge-Fase-3
```

### Passo 2: Baixar Modelos Locais (Ollama)

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Passo 3: Instalar Dependências
```bash
uv sync
```

### Passo 4: Executar o Dashboard
```bash
uv run streamlit run main.py
```

---

### (Opcional) Pipeline de Fine-tuning

> **Requer Apple Silicon (M1/M2/M3)** — MLX não roda em x86.

```bash
# dar permissão de execução (apenas na primeira vez)
chmod +x run_finetune.sh

# executar pipeline completo
./run_finetune.sh
```

Menu interativo pergunta:

1. Importar MedQuAD do GitHub (filtragem automática saúde da mulher)
2. Número de iterações (100 / 300 / 500 / 1000 / custom)

O script verifica `uv`, `ollama` e arquitetura antes de iniciar. Limpa artefatos anteriores (FAISS index, adapters, fused_model, audit DB) automaticamente.

---

## 🧪 Cenários de Teste

Cenários clínicos validados automaticamente via `fine_tuning/avaliar_modelo.py`:

1. Triagem com sangramento vaginal intenso (urgência → `nivel_risco` VERMELHO).
2. Acolhimento em casos de violência doméstica (protocolo 180).
3. Rastreamento mamografia (hardcoded → faixa etária 50–69 anos).
4. Pré-natal inicial (hardcoded → Ácido Fólico).
5. Informação sobre câncer de mama (RAG MedQuAD).
6. Consulta prontuário por nome (DB SQLite → dados da paciente).
7. Filtro ético — bloqueio de prescrição (teste direto `etapa_etica`).

---

## 🎓 Instituição
**FIAP - Pós Tech IA para Devs**   
**Tech Challenge - Fase 3**  

**Autor:** [Wellson Almeida dos Santos]
wellson.digital@gmail.com

**Autor:** [Nelson Seiji Takahashi]
seiji8503@gmail.com


**Projeto**  - https://github.com/pos-fiap-ia-devs/Tech-Challenge-Fase-3

---
> [!NOTE]
> Este projeto foi desenvolvido como prova de conceito para o uso de Agentes de IA em ambientes clínicos controlados.
