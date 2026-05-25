# Diagrama do Fluxo LangGraph — Consultas Medica

```mermaid
flowchart TD
    START([__start__]) -->|classificar_relato| roteador

    roteador{{"🔀 classificar_relato\nsangramento / dor forte / aguda\nmarido / violencia / medo\ngrávida / parto / gestação\nprontuário / histórico de\ndemais casos"}}

    roteador -->|sangramento / dor forte / aguda| urgencia
    roteador -->|marido / violencia / medo / briga| violencia
    roteador -->|grávida / parto / gestação| obstetricia
    roteador -->|prontuário / histórico de| prontuario
    roteador -->|demais casos| prevencao

    urgencia["🚑 etapa_urgencia\nProtocolo FEBRASGO — emergência\nnivel_risco = VERMELHO"]
    violencia["⚖️ etapa_violencia\nProtocolo Lei Maria da Penha\nnivel_risco = VERMELHO"]
    obstetricia["🤰 etapa_obstetricia\nProtocolo Pré-Natal OMS/MS\nnivel_risco = AMARELO"]
    prontuario["📋 etapa_prontuario\nConsulta SQLite prontuarios.db\nnivel_risco = VERDE"]
    prevencao["🏥 etapa_prevencao\nDiretrizes INCA/MS + RAG MedQuAD\nnivel_risco = VERDE"]

    urgencia --> seguranca_etica
    violencia --> seguranca_etica
    obstetricia --> seguranca_etica
    prontuario --> seguranca_etica
    prevencao --> seguranca_etica

    seguranca_etica["🛡️ etapa_etica\nFiltro ético — bloqueia prescrições\nAudit → SQLite prontuarios.db"]
    seguranca_etica --> END([__end__])

    style urgencia fill:#ff4b4b,color:#fff
    style violencia fill:#9b59b6,color:#fff
    style obstetricia fill:#3498db,color:#fff
    style prontuario fill:#1abc9c,color:#fff
    style prevencao fill:#27ae60,color:#fff
    style seguranca_etica fill:#f39c12,color:#fff
    style roteador fill:#ecf0f1,color:#333
```

## Descrição dos nós

| Função Python | Nó no grafo | Responsabilidade | Fonte de dados |
| --- | --- | --- | --- |
| `etapa_urgencia` | `urgencia` | Alerta de emergência — encaminha para pronto-socorro | Hardcoded |
| `etapa_violencia` | `violencia` | Protocolo de acolhimento — Ligue 180, CREAS | Hardcoded |
| `etapa_obstetricia` | `obstetricia` | Orienta pré-natal (Ácido Fólico, consultas, exames) | RAG (MedQuAD) + hardcoded + prontuário |
| `etapa_prontuario` | `prontuario` | Consulta e exibe prontuário da paciente por nome | SQLite `prontuarios.db` |
| `etapa_prevencao` | `prevencao` | Responde dúvidas clínicas preventivas (mama, papanicolau, etc.) | RAG (FAISS/MedQuAD) + LLM + prontuário |
| `etapa_etica` | `seguranca_etica` | Filtra prescrições/dosagens e grava auditoria no SQLite | Regras fixas |

## Estado do grafo (`EstadoAtendimento`)

```python
class EstadoAtendimento(TypedDict):
    relato: str               # pergunta/relato da paciente
    protocolo_seguranca: bool # True quando violência doméstica detectada
    nivel_risco: str          # VERDE | AMARELO | VERMELHO
    resposta_final: str       # resposta gerada pelo pipeline
```
