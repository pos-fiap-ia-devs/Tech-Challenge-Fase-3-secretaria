# Fluxo Breast Cancer — FemCare AI MVP

Documentação acadêmica do fluxo principal de **apoio à triagem em câncer de mama** integrado ao MVP FemCare AI (Tech Challenge FIAP Fase 3 — desafio Secretaria).

> **Aviso:** este é um **MVP educacional**. O fluxo **não emite diagnóstico definitivo**, **não prescreve medicamentos** e **não substitui avaliação por profissional de saúde**. Não deve ser utilizado em contexto clínico real.

---

## 1. Objetivo do fluxo

O fluxo Breast Cancer tem como objetivo:

- interpretar **dados sintéticos** de paciente e exame compatíveis com o contexto oncológico de mama;
- classificar o **nível de atenção clínica** (`alto`, `moderado`, `baixo`) com apoio do módulo reaproveitado da Fase 2;
- apresentar **explicação**, **recomendação**, **limitações** e **fontes** em linguagem cautelosa;
- aplicar **guardrails de segurança** e registrar **auditoria** da interação;
- demonstrar integração entre Streamlit, LangGraph, ML tabular, validação e logs.

O resultado é **apoio à triagem acadêmica**, nunca confirmação de câncer.

---

## 2. Como o usuário aciona no Streamlit

1. Executar a aplicação (`streamlit run main.py`).
2. Aguardar o carregamento dos motores de IA (Ollama + grafo LangGraph).
3. Digitar no campo de chat uma pergunta relacionada a **triagem ou classificação de exame de mama**, preferencialmente mencionando um `patient_id` sintético (`P001`–`P008`).
4. A mensagem é enviada ao grafo como:

```python
{"relato": "<texto digitado pelo usuário>"}
```

5. A resposta aparece no chat; o histórico recente fica disponível na sidebar (**Audit Log SQLite**).

Todos os pacientes referenciados são **fictícios** — não representam pessoas reais.

---

## 3. Exemplos de prompts

### Exemplo A — triagem com histórico familiar

```text
Analisar exame de câncer de mama da paciente P002 com histórico familiar.
```

- Aciona o roteador Breast Cancer (termos: `câncer de mama`, `exame de mama`).
- Identifica `P002` no relato.
- Carrega dados sintéticos de P002 nos CSVs.

### Exemplo B — classificação de exame

```text
Classificar exame de mama da paciente P003.
```

- Aciona o roteador Breast Cancer (termos: `classificar exame`, `exame de mama`).
- Identifica `P003` no relato.
- Retorna classificação de atenção clínica com base nas features Wisconsin simuladas de P003.

### Exemplo C — paciente por nome sintético

```text
Analisar exame de câncer de mama da paciente Carla Mendes com histórico familiar.
```

- Não contém `P00X` explícito; resolve **Carla Mendes** → `P005` nos CSVs (`name_match`).
- **Sem** aviso de caso demonstrativo padrão.

### Exemplo D — relato sem identificação de paciente

```text
Preciso de apoio à triagem de câncer de mama com dados sintéticos.
```

- Nenhum `patient_id` nem nome reconhecido → caso demonstrativo **P002** (`default_demo_case`).
- Exibe aviso discreto: nenhuma paciente específica foi identificada; foi usado P002 como demonstração.

### Exemplo E — ID explícito inexistente

```text
Analisar exame de câncer de mama da paciente P999.
```

- Detecta `P999`, mas **não** há par paciente/exame nos CSVs.
- **Não** usa P002 nem executa `analyze_breast_cancer_case`.
- Resposta segura pedindo `patient_id` válido (`P001`–`P008`) ou nome cadastrado; `nivel_risco` = `AMARELO`; audit com `safety_status` = `not_executed`.

> **Nota:** perguntas genéricas como *"Quando fazer mamografia?"* **não** acionam este fluxo — são tratadas pelo nó de **prevenção**.

---

## 4. Como o roteador do LangGraph identifica o fluxo

Arquivo: `src/engine/grafo_clinico.py` — função interna `classificar_relato`.

O roteador converte o relato para minúsculas e verifica **substrings** na ordem abaixo (primeiro match vence):

| Prioridade | Destino |
| --- | --- |
| 1 | urgência |
| 2 | violência |
| 3 | obstetrícia |
| 4 | **breast_cancer** |
| 5 | prontuário |
| 6 | prevenção (default) |

### Termos que acionam `breast_cancer`

- `câncer de mama`, `cancer de mama`
- `nódulo`, `nodulo`
- `biopsia`, `biópsia`
- `exame de mama`
- `classificar exame`
- `triagem de mama`
- `wisconsin`, `radius_mean`, `concavity`

Log registrado: `Roteador → breast_cancer`

---

## 5. Ordem do fluxo

```text
Streamlit (main.py)
    ↓
LangGraph (grafo_clinico.py → classificar_relato)
    ↓
etapa_breast_cancer (src/engine/etapa_breast_cancer.py)
    ↓
search_func → FAISS (MedQuAD + data/protocols/*.md)  [busca_medquad.py]
    ↓
síntese RAG (bullets do protocolo → sentenças → fallback estruturado; sem metadados na UI)
    ↓
_select_case (patient_id / nome / P002 demo / hardcoded)
    ↓
dados sintéticos (CSVs em data/synthetic/)
    ↓
breast_cancer_phase2_tool (phase2_joblib_model ou rule_based_fallback)
    ↓
response_validator (validate_medical_response)
    ↓
audit_logger (write_audit_log → outputs/audit_logs.jsonl)
    ↓
etapa_etica / seguranca_etica (filtro ético + SQLite)
    ↓
resposta final exibida no Streamlit
```

### Etapas resumidas

| Etapa | Módulo | Ação |
| --- | --- | --- |
| Interface | `main.py` | Envia `relato`, injeta `search_func`, exibe `resposta_final` |
| Roteamento | `grafo_clinico.py` | Direciona para nó `breast_cancer` |
| RAG | `busca_medquad.py` + `search_func` | Recupera chunks MedQuAD e protocolos curados no mesmo FAISS |
| Apresentação RAG | `etapa_breast_cancer.py` | Prioriza bullets reais do protocolo; não exibe metadados FAISS na resposta |
| Seleção de caso | `etapa_breast_cancer.py` (`_select_case`) | ID explícito, nome sintético, P002 demo ou contingência hardcoded |
| Orquestração | `etapa_breast_cancer.py` | Seleção, `fallback_info`, avisos de transparência, Markdown, validação, auditoria |
| ML / triagem | `breast_cancer_phase2_tool.py` | `phase2_joblib_model` (joblib + 30 features) ou `rule_based_fallback` |
| Segurança | `response_validator.py` | Bloqueia linguagem inadequada |
| Auditoria JSONL | `audit_logger.py` | Append em `outputs/audit_logs.jsonl` |
| Filtro ético | `etapas_clinicas.py` → `etapa_etica` | Filtro adicional + SQLite |

### Indexação RAG (`src/rag/busca_medquad.py`)

O índice FAISS em `data/faiss_index/` é construído a partir de:

- **MedQuAD** — `data/medquad.csv` (~1.155 documentos filtrados);
- **Protocolos curados** — `data/protocols/*.md`, incluindo `breast_cancer_screening.md` (**17 chunks** no estado atual do repositório).

Metadados por chunk de protocolo:

| Campo | Exemplo |
| --- | --- |
| `source` | `breast_cancer_screening.md` |
| `source_type` | `curated_protocol` |
| `category` / `categoria` | `breast_cancer_screening` |
| `path` | `data/protocols/breast_cancer_screening.md` |

**Evidência de recuperação (busca direta no retriever):** consultas sobre rastreamento de câncer de mama retornam documentos com `metadata["source"] == "breast_cancer_screening.md"` e `metadata["source_type"] == "curated_protocol"`.

### RAG e síntese do protocolo (`src/engine/etapa_breast_cancer.py`)

O fluxo consulta o índice **FAISS** (MedQuAD + protocolos) e prioriza o documento curado **`breast_cancer_screening.md`** quando a consulta é relevante.

O chunk bruto do retriever pode conter linhas técnicas (`Category:`, `SourceType:`, `Protocol:`, `Content:`). Esses metadados **não são exibidos** na resposta ao usuário.

Ordem de síntese aplicável (`_build_rag_synthesis`):

1. **Bullets reais** do trecho recuperado (ex.: seção 6.4 — alterações que devem chamar atenção);
2. **Sentenças úteis** extraídas do conteúdo limpo;
3. **Fallback estruturado** quando o trecho não permite extração direta.

A seção **Trechos aplicáveis do protocolo recuperado** apresenta documento, seção e síntese em linguagem cautelosa — apoio documental à triagem, **não** diagnóstico definitivo.

Logs técnicos registram a fonte da síntese (`recovered_protocol_bullets`, `recovered_protocol_sentence`, `structured_fallback`) para rastreabilidade em desenvolvimento.

---

## 6. Dados usados

### `data/synthetic/synthetic_patients.csv`

Pacientes fictícias com campos como:

- `patient_id`, `nome_sintetico`, `idade`, `gestante`
- `historico_familiar_cancer_mama`
- `ultima_mamografia`, `ultimo_papanicolau`
- `sintomas_relatados`, `observacoes`

### `data/synthetic/synthetic_breast_exam_results.csv`

Exames simulados no formato Wisconsin, vinculados por `patient_id`:

- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`
- `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave_points_mean`

### Seleção do caso sintético

Implementada em `_select_case` (`etapa_breast_cancer.py`). Retorna também `selection_method` para transparência na resposta e nos logs.

| Prioridade | Condição | `selection_method` | Aviso na resposta |
| --- | --- | --- | --- |
| 1 | `patient_id` no relato (`P002`, `P006`, …) e par paciente/exame nos CSVs | `explicit_id` | Não |
| 2 | Nome sintético cadastrado (`nome_sintetico` / `nome`), ex.: **Carla Mendes**, **Ana Silva** | `name_match` | Não |
| 3 | Nenhuma paciente identificada → **P002** como caso demonstrativo padrão | `default_demo_case` | Sim — observação sobre P002 |
| 4 | CSVs indisponíveis ou incompletos → dicionários `_FALLBACK_*` | `hardcoded_fallback` | Sim — contingência sem arquivos sintéticos |
| — | `patient_id` informado mas **ausente** nos CSVs (ex.: **P999**) | `explicit_id_not_found` | Resposta dedicada; **sem** fallback silencioso para P002 |

**Exemplos de identificação:**

- `P002` ou `P006` no texto → `explicit_id` (mesmo que P002 seja o padrão acadêmico, **não** gera aviso de demo).
- `Carla Mendes` → `P005` (`name_match`).
- Relato genérico sobre triagem, sem ID nem nome → `P002` + aviso de caso demonstrativo.
- `P999` → mensagem de ID inválido; tool e modelo **não** são acionados.

### Transparência e `fallback_info`

Após a triagem (ou quando ela não é executada), `etapa_breast_cancer` devolve **`fallback_info`** no state parcial, por exemplo:

- `patient_selection_method`, `default_demo_case_used`, `hardcoded_fallback_used`
- `model_inference_method`, `model_loaded`, `model_inference_failed` (uso em logs; **não** na UI)
- `rag_available`, `rag_source`, `rag_payload_available`

A função `_build_transparency_notice` monta avisos Markdown **discretos** (após o resumo da paciente, antes do nível de atenção):

| Situação | Aviso ao usuário (resumo) |
| --- | --- |
| `default_demo_case` | Caso demonstrativo P002 |
| `hardcoded_fallback` | CSVs sintéticos indisponíveis |
| `phase2_joblib_model` | Método da análise: modelo treinado da Fase 2 |
| `rule_based_fallback` | Regra acadêmica de contingência (+ modo degradado se inferência falhou) |
| RAG sem payload útil | Protocolo não recuperado nesta consulta |

A síntese RAG estruturada interna (quando bullets não estão disponíveis) é registrada em log; **não** precisa de aviso extra se ainda houver documento/trecho na resposta.

---

## 7. Campos principais da resposta

A resposta final em Markdown agrega informações da tool e da etapa. Campos centrais:

| Campo | Descrição | Exemplo |
| --- | --- | --- |
| **patient_id** | ID sintético utilizado | `P002` |
| **risk_level** | Nível de atenção clínica | `alto`, `moderado`, `baixo` |
| **prediction_label** | Rótulo cauteloso do modelo | `maior atenção clínica` |
| **probability** | Probabilidade estimada (0–1) | `0.92` |
| **explanation** | Justificativa em linguagem natural | Features Wisconsin acima/abaixo dos limiares |
| **recommended_action** | Conduta sugerida (triagem) | Encaminhar para avaliação profissional |
| **limitations** | Aviso de limitação clínica | Não é diagnóstico definitivo |
| **sources** | Documentos de referência | `phase2_breast_cancer_model_card.md`, `breast_cancer_screening.md`, **`MedQuAD/FAISS`** (quando RAG retorna contexto) |
| **Trechos do protocolo** | Síntese RAG integrada | Seção *Trechos aplicáveis do protocolo recuperado* (sem `Category:` / `Protocol:` / `Content:`) |
| **inference_method** | Modo de inferência | **`phase2_joblib_model`** quando `models/breast_cancer_rf_ga_pipeline.joblib` carrega e as 30 features estão disponíveis; senão **`rule_based_fallback`** |
| **Avisos na resposta** | Transparência | Paciente, método de análise e RAG (quando aplicável); linguagem amigável, sem jargão técnico |
| **`fallback_info`** | Rastreabilidade no state | Metadados para integração, logs e testes locais |

O grafo também atualiza:

- `nivel_risco`: `VERMELHO` (alto), `AMARELO` (moderado), `VERDE` (baixo)
- `flow`: `breast_cancer`
- `fallback_info`: dicionário de fallbacks (quando a etapa é executada)

---

## 8. Como os logs são gerados

### `outputs/audit_logs.jsonl`

Gerado por `write_audit_log` dentro de `etapa_breast_cancer`. Um evento JSON por linha (append):

```json
{
  "timestamp": "2026-05-23T23:27:21-03:00",
  "flow": "breast_cancer",
  "risk_level": "alto",
  "sources": ["phase2_breast_cancer_model_card.md", "breast_cancer_screening.md"],
  "safety_status": "approved",
  "sensitive_case": false,
  "summary": "Caso sintético P002 analisado — maior atenção clínica"
}
```

Quando há fallback relevante, o **summary** pode incluir sufixo textual, por exemplo:

- `... Fallback: default_demo_case.`
- `... Fallback: rule_based_fallback.`
- `... Fallback: rag_unavailable.`
- `Caso sintético unknown não analisado. ID explícito não encontrado (P999).`

Campos extras de `fallback_info` **não** são gravados no JSONL se o `audit_logger` sanitizar o payload — a rastreabilidade principal fica no **summary** e nos logs da etapa.

### SQLite existente (via `etapa_etica`)

Após o nó Breast Cancer, o fluxo passa por `seguranca_etica`, que grava em `data/prontuarios.db` (tabela `atendimentos`):

- relato original
- nível de risco (`VERMELHO` / `AMARELO` / `VERDE`)
- resposta final

Exibido na sidebar do Streamlit como **Audit Log (SQLite)**.

### Logs técnicos

Registros adicionais em console e `logs/consultas_medica.log` (roteamento, tool, validação).

---

## 9. Segurança

O fluxo obedece às regras do MVP FemCare AI:

| Regra | Implementação |
| --- | --- |
| **Não diagnostica** | Usa termos como *maior atenção clínica*; nunca afirma câncer confirmado |
| **Não prescreve** | `response_validator` + `etapa_etica` bloqueiam prescrições |
| **Não informa dosagem** | Padrões de dosagem (`mg`, `ml`, posologia) são detectados e bloqueados |
| **Recomenda avaliação profissional** | Obrigatório em `risk_level == alto` |
| **Linguagem cautelosa** | Limitações sempre presentes na resposta Markdown |
| **Fallbacks explícitos** | Alterações de paciente, inferência ou RAG documentadas na resposta, em `fallback_info`, logs e summary de auditoria — **sem** fallback silencioso |
| **MVP acadêmico** | Demonstração educacional; **não** uso clínico real nem diagnóstico automático |

Camadas de segurança:

1. Tool — linguagem de triagem, não de diagnóstico
2. `validate_medical_response` — guardrail textual
3. `etapa_etica` — filtro ético legado + auditoria SQLite

---

## 10. Modelo da Fase 2 na tool (`breast_cancer_phase2_tool.py`)

| Etapa | Comportamento |
| --- | --- |
| Carregamento | Tenta `models/breast_cancer_rf_ga_pipeline.joblib` (e caminhos alternativos documentados no código) |
| Inferência principal | Com modelo + **30 features** do exame sintético → `inference_method="phase2_joblib_model"` |
| Nomes de features | CSV da Fase 3 usa **underscore** (`concave_points_mean`); o pipeline da Fase 2 pode esperar **espaço** (`concave points_mean`) — a tool adapta automaticamente via `feature_names_in_` |
| Fallback | Se o joblib falhar, faltar dependência (ex.: scikit-learn), faltar features ou `predict_proba` não funcionar → **`rule_based_fallback`** explícito (6 features Wisconsin); aviso amigável na resposta |

O score exibido na resposta é **indicador acadêmico de triagem**, não probabilidade clínica confirmatória. Métricas históricas (recall, specificity) permanecem no model card — **não** são expostas como diagnóstico na UI.

---

## 11. Exemplo resumido de saída

**Entrada:**

```text
Analisar exame de câncer de mama da paciente P002 com histórico familiar.
```

**Saída (resumida):**

```markdown
### 🎀 Breast Cancer — Apoio à Triagem (FemCare AI)

**Paciente sintética:** P002
**Nível de atenção clínica:** alto
**Classificação do modelo:** maior atenção clínica
**Probabilidade estimada:** 92.00%

#### Explicação
Caso de paciente sintética 'P002', idade 28 anos, histórico familiar de câncer de mama informado.
[...] Este resultado é apoio à triagem e não confirma a presença de câncer.

#### Recomendação
Encaminhar para avaliação profissional e exames confirmatórios.

#### Trechos aplicáveis do protocolo recuperado
- Documento recuperado: breast_cancer_screening.md
- Seção/trecho aplicável: 6.4 Alterações que devem chamar atenção
- Síntese aplicável: [bullets do protocolo em linguagem cautelosa]

#### Limitações
Este resultado é apoio à triagem acadêmica e **não representa diagnóstico definitivo**.

#### Fontes
- phase2_breast_cancer_model_card.md
- breast_cancer_screening.md
- MedQuAD/FAISS
```

> A saída **não** deve exibir metadados crus do FAISS (`Category:`, `SourceType:`, `Protocol:`, `Content:`). Com `default_demo_case`, inclui observação sobre P002 após o resumo da paciente.

---

## 12. Limitações do fluxo

1. **MVP educacional** — desenvolvido para demonstração acadêmica, não para uso clínico real.
2. **Dados sintéticos** — pacientes e exames são fictícios; não representam prontuários reais.
3. **Roteamento keyword-based** — relatos ambíguos podem ser direcionados a outros nós.
4. **Fallback rule-based** — se o joblib não carregar (ex.: ambiente sem scikit-learn) ou faltarem features, a inferência cai em `rule_based_fallback`; o RAG e a tool coexistem, mas não substituem avaliação médica.
5. **RAG dependente do índice** — é necessário reconstruir `data/faiss_index/` após mudanças em `medquad.csv` ou `data/protocols/`.
6. **Sem validação clínica** — métricas da Fase 2 são históricas; não garantem desempenho em dados reais.
7. **Dupla auditoria** — JSONL + SQLite coexistem por design; não substituem sistemas hospitalares.
8. **Não substitui médico** — classificação de atenção clínica ≠ diagnóstico.

---

## 13. Como testar

### No terminal (módulos isolados)

```bash
# RAG — verificar recuperação de breast_cancer_screening.md no FAISS
python test_rag_protocols.py

# Etapa Breast Cancer (simula state do grafo + fake_search_func)
python -m src.engine.etapa_breast_cancer

# Tool de classificação
python -m src.tools.breast_cancer_phase2_tool

# Validador de segurança
python -m src.safety.response_validator

# Audit logger
python -m src.logging_audit.audit_logger
```

> Antes do primeiro teste após alterar protocolos: `Remove-Item -Recurse -Force data\faiss_index` (PowerShell) ou apagar a pasta manualmente.

### No Streamlit (fluxo integrado)

```bash
python -m streamlit run main.py
```

Prompts sugeridos no chat:

```text
Analisar exame de câncer de mama da paciente P002 com histórico familiar.
```

```text
Classificar exame de mama da paciente P003.
```

**Critério de sucesso:**

- Resposta Markdown com título *Breast Cancer — Apoio à Triagem*
- Seção **Trechos aplicáveis do protocolo recuperado** com síntese (bullets quando disponíveis)
- **Sem** exibir metadados técnicos na resposta: `Category:`, `SourceType:`, `Protocol:`, `Content:`
- Fontes incluindo **MedQuAD/FAISS**
- `inference_method`: **`phase2_joblib_model`** (com scikit-learn + joblib) ou **`rule_based_fallback`**
- `patient_id` correto (P002 explícito, P005 por nome, P002 + aviso sem identificação; **P999** sem cair em P002)
- `python -m src.engine.etapa_breast_cancer` — cenários A–D no `__main__` do módulo
- Recomendação de avaliação profissional quando risco alto
- Nova linha em `outputs/audit_logs.jsonl`
- Novo registro na sidebar **Audit Log (SQLite)**
- Log `Roteador → breast_cancer` em `logs/consultas_medica.log`

---

## Referências

| Artefato | Caminho |
| --- | --- |
| Model card (Fase 2) | `docs/phase2_breast_cancer_model_card.md` |
| Grafo LangGraph | `src/engine/grafo_clinico.py` |
| Nó Breast Cancer | `src/engine/etapa_breast_cancer.py` |
| Tool ML | `src/tools/breast_cancer_phase2_tool.py` |
| Indexação RAG | `src/rag/busca_medquad.py` |
| Protocolo curado (FAISS) | `data/protocols/breast_cancer_screening.md` |
| Síntese RAG | `src/engine/etapa_breast_cancer.py` (`_build_rag_synthesis`, `_build_protocol_reference`) |
| Seleção de paciente | `src/engine/etapa_breast_cancer.py` (`_select_case`, `_find_patient_id_by_name`) |
| Transparência / fallbacks | `src/engine/etapa_breast_cancer.py` (`_build_fallback_info`, `_build_transparency_notice`) |
| Teste RAG | `test_rag_protocols.py` |
