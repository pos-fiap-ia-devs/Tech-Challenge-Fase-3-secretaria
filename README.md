# 🩺 ConsultasMedica - Assistente Clínico de Saúde da Mulher
### FIAP | Pós Tech IA para Devs - Tech Challenge Fase 3

Este projeto apresenta um Assistente Clínico Inteligente desenvolvido com foco em **Saúde da Mulher**, utilizando tecnologias de IA generativa de ponta com execução **100% local** para garantir a máxima privacidade dos dados sensíveis (LGPD).

---

## 🚀 Visão Geral
O **ConsultasMedica** é uma solução orquestrada que realiza triagens clínicas automáticas, responde dúvidas de saúde da mulher com base em evidências e garante conformidade ética através de filtros de segurança em tempo real.

### Principais Diferenciais:
- **Orquestração com LangGraph:** Fluxo de decisão não linear que diferencia rotina, urgência e casos especializados.
- **RAG (Retrieval Augmented Generation):** Consulta MedQuAD/NIH e protocolos Markdown curados (`data/protocols/*.md`) no **mesmo índice FAISS** para contextualizar respostas.
- **Privacidade Total (Air-Gapped):** Execução local via Ollama (Llama 3.1 8B), sem envio de dados para a nuvem.
- **Filtro Ético de Prescrição:** Bloqueio automático de tentativas de dosagem ou receitas médicas por IA.

---

## 🛠️ Estrutura do Projeto
```text
├── data/
│   ├── prontuarios.db               # SQLite — audit de atendimentos
│   ├── pacientes_sinteticos.csv     # Dataset anonimizado de pacientes
│   ├── medquad.csv                  # Dataset MedQuAD/NIH (~16k exemplos)
│   ├── protocols/                   # Protocolos Markdown curados (indexados no FAISS)
│   │   └── breast_cancer_screening.md  # Diretriz INCA para detecção precoce
│   ├── faiss_index/                 # Índice FAISS persistido (MedQuAD + protocolos)
│   └── synthetic/
│       ├── synthetic_patients.csv           # Pacientes sintéticas (fluxo Breast Cancer)
│       └── synthetic_breast_exam_results.csv  # Exames simulados Wisconsin (30 features)
├── docs/
│   ├── diagrama_langgraph.md        # Diagrama Mermaid do fluxo
│   ├── diagrama_langgraph.png       # Diagrama PNG gerado
│   ├── breast_cancer_flow.md        # Documentação do fluxo Breast Cancer
│   ├── phase2_breast_cancer_model_card.md  # Model card do modelo RF+GA da Fase 2
│   ├── project_context.md           # Contexto e escopo do projeto
│   └── relatorio_tecnico.md         # Relatório técnico completo
├── fine_tuning/
│   ├── adapters/                    # adapter_config.json (pesos LoRA gerados localmente*)
│   ├── fused_model/                 # Modelo fundido LoRA+base (gerado localmente*)
│   ├── data/                        # train.jsonl / valid.jsonl (gerados localmente*)
│   ├── importar_medquad.py          # Importação MedQuAD do GitHub
│   ├── preparar_dados.py            # Preprocessing CSV → JSONL + split 80/20
│   ├── treinar_modelo.py            # Fine-tuning LoRA via MLX (Apple Silicon)
│   ├── testar_inferencia.py         # Inferência com modelo fine-tunado
│   ├── avaliar_modelo.py            # Avaliação com cenários clínicos (9 cenários)
│   └── exportar_hf.py               # Export para HuggingFace Hub
├── models/
│   ├── breast_cancer_rf_ga_pipeline.joblib  # Pipeline RF+GA da Fase 2 (scikit-learn)
│   └── breast_cancer_rf_ga_metadata.json   # Métricas e hiperparâmetros do modelo
├── outputs/
│   └── audit_logs.jsonl             # Auditoria estruturada JSONL (fluxo Breast Cancer)
├── src/
│   ├── logger.py                    # Logger centralizado (console + arquivo + rotação)
│   ├── db/
│   │   └── prontuarios.py           # SQLite — prontuários e audit de atendimentos
│   ├── engine/
│   │   ├── grafo_clinico.py         # Compilação do grafo LangGraph (6 nós)
│   │   ├── etapas_clinicas.py       # Etapas clínicas: urgência, violência, obstetrícia, prevenção, prontuário
│   │   ├── etapa_breast_cancer.py   # Nó Breast Cancer — triagem acadêmica (RAG + tool + audit)
│   │   └── estado_atendimento.py    # EstadoAtendimento (TypedDict compartilhado)
│   ├── tools/
│   │   └── breast_cancer_phase2_tool.py  # Tool ML Fase 2 (RF+GA joblib ou regra de contingência)
│   ├── safety/
│   │   └── response_validator.py    # Validador de segurança textual (sem diagnóstico/prescrição)
│   ├── logging_audit/
│   │   └── audit_logger.py          # Audit logger JSONL estruturado
│   └── rag/
│       └── busca_medquad.py         # RAG Engine (FAISS: MedQuAD + protocols/*.md)
├── logs/
│   └── consultas_medica.log         # Logs de auditoria (rotação diária, 30 dias)
├── main.py                          # Interface Streamlit
├── generate_diagram.py              # Gerador do diagrama LangGraph
├── test_rag_protocols.py            # Teste direto do retriever RAG (verifica metadados)
└── run_finetune.sh                  # Pipeline completo com menu interativo

* Artefatos de fine-tuning (pesos LoRA, dados JSONL, modelo fundido) são gerados localmente
  pelo pipeline fine_tuning/ e não estão incluídos no repositório por tamanho.
  Requerem Apple Silicon (MLX). Métricas do treino estão em docs/relatorio_tecnico.md §2.
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

## 🎀 Fluxo Breast Cancer / Prevenção Oncológica

MVP acadêmico de **apoio à triagem** em câncer de mama, integrado ao LangGraph. O fluxo reaproveita conceitualmente o trabalho da **Fase 2** (Breast Cancer Wisconsin Dataset, Random Forest + Algoritmo Genético) via `src/tools/breast_cancer_phase2_tool.py`, usando **dados sintéticos** em `data/synthetic/`.

> **Aviso:** este fluxo **não emite diagnóstico definitivo**, **não prescreve medicamentos** e **não substitui avaliação por profissional de saúde**. É uma demonstração educacional — **não deve ser utilizado em contexto clínico real**.

O chat utiliza **Ollama (Llama 3.1 8B base)** nos fluxos conversacionais. O **fine-tuning LoRA** documentado em `fine_tuning/` é um **pipeline demonstrativo** — **não está em produção** como modelo principal do dashboard.

### Seleção de paciente sintética

A etapa `etapa_breast_cancer` resolve o caso em `data/synthetic/` nesta ordem:

1. **`patient_id` no relato** — ex.: `P002`, `P006` (`explicit_id`, sem aviso);
2. **Nome cadastrado** — ex.: Carla Mendes → `P005`, Ana Silva → `P001` (`name_match`, sem aviso);
3. **Nenhuma identificação** — usa **P002** como caso demonstrativo padrão e exibe aviso discreto (`default_demo_case`);
4. **ID explícito inexistente** — ex.: `P999` **não** cai em P002; retorna mensagem segura pedindo ID/nome válido (`P001`–`P008`), sem executar tool nem modelo;
5. **CSVs indisponíveis** — contingência hardcoded com aviso (`hardcoded_fallback`).

### Modelo da Fase 2 (triagem tabular)

A tool `breast_cancer_phase2_tool.py` tenta carregar `models/breast_cancer_rf_ga_pipeline.joblib`. Com modelo e **30 features** do exame sintético, usa `inference_method="phase2_joblib_model"`. O CSV da Fase 3 usa nomes com underscore; a tool adapta automaticamente para nomes com espaço esperados pelo pipeline (ex.: `concave_points_mean` → `concave points_mean`). Se o joblib falhar, faltar dependência (ex.: scikit-learn) ou faltarem features, usa **regra acadêmica de contingência** (`rule_based_fallback` interno), informada na resposta — **sem** expor termos técnicos como `joblib` ou listas de features na UI.

### Fallbacks e transparência

O nó `etapa_breast_cancer` retorna **`fallback_info`** no state (seleção de paciente, inferência, RAG). Fallbacks que alteram paciente, método de análise ou fonte documental **não são silenciosos**: avisos discretos na resposta (após o resumo da paciente) e/ou registro em logs e no **summary** do `outputs/audit_logs.jsonl` (ex.: `Fallback: default_demo_case`, `rule_based_fallback`, `rag_unavailable`). Detalhes internos (`model_loaded`, `model_path`, métricas Recall/Specificity) permanecem fora da resposta ao usuário.

### RAG no fluxo Breast Cancer

O motor em `src/rag/busca_medquad.py` indexa no **mesmo FAISS** usado pelo restante do app:

- **MedQuAD** (`data/medquad.csv`) — perguntas e respostas filtradas por relevância em saúde da mulher;
- **Protocolos curados** (`data/protocols/*.md`), incluindo `data/protocols/breast_cancer_screening.md` (diretriz INCA curada para detecção precoce).

O nó `etapa_breast_cancer` consulta esse índice via **`search_func`**. A síntese prioriza **bullets reais** do protocolo recuperado; a resposta **não exibe** metadados técnicos do FAISS (`Category:`, `SourceType:`, `Protocol:`, `Content:`).

### Teste rápido no dashboard

1. Execute o dashboard:
   ```bash
   uv run streamlit run main.py
   ```
2. Prompts sugeridos:
   ```text
   Analisar exame de câncer de mama da paciente P002 com histórico familiar.
   ```
   ```text
   Analisar exame de câncer de mama da paciente Carla Mendes com histórico familiar.
   ```
   ```text
   Preciso de apoio à triagem de câncer de mama com dados sintéticos.
   ```
   ```text
   Analisar exame de câncer de mama da paciente P999.
   ```

### Resultado esperado

| Item | Valor esperado |
| --- | --- |
| Rota LangGraph | `breast_cancer` |
| Paciente | **P002** (ID explícito ou demo), **P005** (Carla Mendes por nome); **P999** → mensagem de ID inválido, **sem** P002 |
| Aviso de caso demo | apenas quando **nenhuma** paciente foi identificada no relato |
| `fallback_info` | presente no retorno do grafo (rastreabilidade acadêmica) |
| Método na resposta | linha *Método da análise:* (modelo Fase 2 ou regra de contingência), quando a triagem é executada |
| Saída | nível de atenção clínica, triagem acadêmica, **sem** diagnóstico definitivo, prescrição ou dosagem |
| Seção RAG | **Trechos aplicáveis do protocolo recuperado** (síntese com bullets quando disponíveis; **sem** metadados FAISS na UI) |
| Fontes | `phase2_breast_cancer_model_card.md`, `breast_cancer_screening.md`, **MedQuAD/FAISS** |
| Método de inferência | **`phase2_joblib_model`** (joblib + scikit-learn) ou **`rule_based_fallback`** |
| Segurança | `response_validator` + recomendação de avaliação profissional em risco alto |
| Auditoria JSONL | nova linha em `outputs/audit_logs.jsonl` |
| Sidebar | novo registro em **Audit Log (SQLite)** via `etapa_etica` |

Documentação detalhada: `docs/breast_cancer_flow.md` e `docs/phase2_breast_cancer_model_card.md`.

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
8. **Triagem Breast Cancer — P002 explícito** (`phase2_joblib_model`, seleção por ID).
9. **Triagem Breast Cancer — Carla Mendes** (`name_match` → P005, resolução por nome).

Testes unitários do fluxo Breast Cancer e do retriever RAG:
```bash
python test_rag_protocols.py              # verifica metadados breast_cancer_screening.md no FAISS
python -m src.engine.etapa_breast_cancer  # 7 cenários internos (P001–P008, P999, demos)
python -m src.tools.breast_cancer_phase2_tool  # carregamento do modelo RF+GA + predição
```

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
