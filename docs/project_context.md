# PROJECT_CONTEXT.md

# FemCare AI — MVP Acadêmico para Tech Challenge FIAP Fase 3

## 1. Visão Geral

O **FemCare AI** é um MVP acadêmico desenvolvido para o **Tech Challenge FIAP Fase 3**, no contexto do desafio “Secretaria” voltado à saúde da mulher.

O projeto propõe um assistente de apoio à prevenção e triagem em saúde da mulher, utilizando:

- LLM;
- RAG com LangChain;
- fluxos automatizados com LangGraph;
- dados sintéticos;
- logs e auditoria;
- safety validator;
- pipeline demonstrativo de fine-tuning;
- reaproveitamento do trabalho da Fase 2 com Breast Cancer.

O sistema deve ser entendido como uma ferramenta acadêmica de apoio à decisão e triagem, não como sistema médico real, não como substituto de profissional de saúde e não como ferramenta de diagnóstico definitivo.

---

## 2. Objetivo do Projeto

Criar um assistente virtual de apoio à prevenção e triagem em saúde da mulher, capaz de:

1. responder dúvidas clínicas contextualizadas com apoio de RAG;
2. consultar dados sintéticos de pacientes;
3. executar fluxos automatizados com LangGraph;
4. interpretar dados sintéticos de exames de câncer de mama com apoio do modelo da Fase 2;
5. classificar nível de atenção clínica sem diagnóstico definitivo;
6. aplicar regras de segurança médica;
7. registrar logs e auditoria;
8. demonstrar um pipeline de fine-tuning de LLM.

O objetivo acadêmico é demonstrar domínio de arquitetura de IA aplicada, LLMs, RAG, LangChain, LangGraph, dados sintéticos, segurança, explainability e documentação técnica.

---

## 3. Contexto do Tech Challenge

O desafio da Fase 3 solicita a criação de um assistente médico treinado com dados próprios ou simulados, capaz de auxiliar condutas clínicas, responder dúvidas e sugerir procedimentos com base em protocolos internos. Também exige uso de LangChain, fluxos com LangGraph, logging, segurança, explainability, dataset anonimizado ou sintético, pipeline de fine-tuning e relatório técnico detalhado.

O documento “Secretaria” amplia o contexto para saúde da mulher, incluindo prevenção, triagem ginecológica, câncer de mama, câncer de colo do útero, obstetrícia, violência doméstica, saúde mental e fluxos seguros de atendimento. O MVP não implementa um hospital completo, mas cobre esses temas em versão acadêmica e controlada, com Breast Cancer como fluxo principal.

---

## 4. Escopo Principal

O fluxo principal do projeto será:

## Breast Cancer / Prevenção Oncológica

Esse fluxo reaproveita a Fase 2, que utilizou:

- Breast Cancer Wisconsin Dataset;
- Random Forest;
- Algoritmo Genético para otimização;
- foco em recall e redução de falsos negativos;
- explicação de resultado com LLM.

Na Fase 3, esse módulo será usado como ferramenta de apoio à interpretação de risco/atenção clínica, sempre com linguagem cautelosa.

O sistema deverá:

1. carregar dados sintéticos de uma paciente;
2. carregar dados simulados de exame compatíveis com o contexto Breast Cancer;
3. acionar uma tool Python de apoio à classificação;
4. gerar uma explicação em linguagem natural;
5. consultar protocolo ou documento RAG sobre câncer de mama;
6. validar a resposta com safety validator;
7. registrar log/auditoria;
8. retornar resposta com:
   - nível de atenção clínica;
   - justificativa;
   - fonte;
   - limitação;
   - recomendação de avaliação profissional.

O sistema não deve afirmar que a paciente tem câncer. Deve usar termos como:

- “maior atenção clínica”;
- “risco elevado pelo modelo”;
- “necessidade de avaliação especializada”;
- “resultado de apoio à triagem”.

---

## 5. Fluxos Complementares

Além do fluxo principal de Breast Cancer, o MVP deve conter fluxos complementares simplificados para demonstrar aderência ao desafio “Secretaria”.

### 5.1 Fluxo de Prevenção

Objetivo:

- identificar exames preventivos pendentes;
- verificar histórico sintético da paciente;
- orientar busca por avaliação profissional;
- consultar documentos RAG de prevenção.

Exemplo de entrada:

```json
{
  "patient_id": "P001",
  "idade": 45,
  "ultima_mamografia": "2021-03-10",
  "ultimo_papanicolau": "2020-08-15",
  "historico_familiar_cancer_mama": true
}

Saída esperada:

exames pendentes;
nível de prioridade;
orientação preventiva;
limitação clínica;
fonte consultada.
5.2 Fluxo de Triagem Ginecológica

Objetivo:

analisar sintomas ginecológicos relatados;
identificar sinais de alerta;
classificar risco como baixo, moderado ou alto;
recomendar avaliação presencial quando necessário.

Exemplos de sintomas:

dor pélvica intensa;
sangramento intenso;
febre;
corrimento com odor forte;
dor súbita;
sintomas persistentes.

Saída esperada:

classificação de risco;
justificativa;
recomendação de atendimento;
limitação clínica;
fonte ou protocolo usado.
5.3 Fluxo Obstétrico

Objetivo:

analisar relatos de gestantes;
detectar sinais de alerta obstétrico;
recomendar atendimento presencial em situações de risco.

Exemplos de sinais de alerta:

sangramento na gestação;
dor intensa;
visão turva;
cefaleia intensa;
febre;
redução de movimentos fetais;
perda de líquido.

Saída esperada:

risco gestacional;
alerta de atendimento;
recomendação de avaliação médica;
limitação clínica;
fonte consultada.
5.4 Fluxo de Violência Doméstica

Objetivo:

detectar sinais diretos ou indiretos de possível violência doméstica;
responder com linguagem segura, cautelosa e confidencial;
recomendar encaminhamento para equipe qualificada ou serviço especializado;
não expor dados sensíveis desnecessariamente.

Exemplos de sinais:

medo do parceiro;
agressão relatada;
ameaças;
controle excessivo;
lesões recorrentes;
relato contraditório associado a medo.

Saída esperada:

sinalização de possível risco;
resposta sensível;
recomendação de apoio especializado;
preservação de confidencialidade;
log com tratamento de dado sensível.
6. Fora de Escopo

O MVP não tem como objetivo:

substituir médicos, enfermeiros, psicólogos ou assistentes sociais;
emitir diagnóstico definitivo;
prescrever medicamentos;
informar dosagens;
recomendar tratamento individualizado;
operar com dados reais de pacientes;
integrar sistemas hospitalares reais;
implementar LGPD completa em ambiente produtivo;
executar deploy hospitalar;
criar prontuário eletrônico real;
treinar um LLM grande de produção;
validar clinicamente o modelo com especialistas reais.

O sistema é acadêmico, demonstrativo e voltado à aprendizagem.

7. Arquitetura Atual

O repositório compartilhado utiliza uma arquitetura monolítica baseada em Streamlit, com módulos internos de IA e dados.

Componentes principais:

Streamlit para interface;
LangGraph para fluxos automatizados;
LangChain/RAG para recuperação de contexto;
MedQuAD como base de perguntas e respostas médicas;
SQLite para prontuários sintéticos e registros;
Ollama para execução local de LLM;
scripts demonstrativos de fine-tuning;
logs/auditoria;
dados sintéticos.

Não será criado backend FastAPI nesta fase, para evitar aumento de complexidade e risco de integração.

Arquitetura lógica:

Usuário / Profissional de saúde
        ↓
Interface Streamlit
        ↓
LangGraph Clinical Orchestrator
        ↓
Intent Router
        ↓
Fluxos:
  - Breast Cancer
  - Prevenção
  - Triagem Ginecológica
  - Obstétrico
  - Violência Doméstica
        ↓
LangChain / RAG
        ↓
Base vetorial / documentos / MedQuAD
        ↓
LLM via Ollama
        ↓
Safety Validator
        ↓
Logs / Auditoria
        ↓
Resposta final

8. Módulos Esperados
8.1 Interface

Responsável por:

receber perguntas;
permitir escolha ou simulação de pacientes;
exibir respostas;
exibir histórico/logs;
demonstrar fluxos no vídeo.

Arquivo provável:

main.py

8.2 LangGraph

Responsável por:

classificar intenção;
direcionar para o fluxo adequado;
coordenar etapas clínicas;
acionar módulos de RAG, tool, safety e logs.

Arquivos prováveis:

src/engine/grafo_clinico.py
src/engine/etapas_clinicas.py
src/engine/etapa_breast_cancer.py

8.3 RAG / LangChain

Responsável por:

carregar documentos;
criar ou consultar base vetorial;
recuperar trechos relevantes;
fornecer contexto para a LLM;
melhorar explainability.

Arquivos prováveis:

src/rag/
data/protocols/
data/medquad.csv
8.4 Breast Cancer Tool

Responsável por:

receber dados sintéticos de paciente e exame;
aplicar lógica de apoio à classificação;
retornar nível de atenção clínica;
trazer métricas da Fase 2;
gerar saída padronizada para o LangGraph.

Arquivo esperado:

src/tools/breast_cancer_phase2_tool.py

Função principal esperada:

def analyze_breast_cancer_case(patient_data: dict, exam_data: dict) -> dict:
    ...

Retorno esperado:

{
  "flow": "breast_cancer",
  "risk_level": "alto",
  "prediction_label": "maior atenção clínica",
  "probability": 0.91,
  "model_metrics": {
    "recall": 0.952381,
    "specificity": 0.972222
  },
  "explanation": "O modelo indicou maior atenção com base nos atributos informados.",
  "limitations": "A saída não representa diagnóstico definitivo.",
  "recommended_action": "Encaminhar para avaliação profissional e exames confirmatórios.",
  "sources": [
    "phase2_breast_cancer_model_card.md",
    "breast_cancer_screening.md"
  ]
}
8.5 Safety Validator

Responsável por verificar se a resposta:

não diagnostica definitivamente;
não prescreve;
não informa dose;
recomenda avaliação profissional em risco alto;
preserva confidencialidade em casos sensíveis;
informa limitação clínica;
inclui fonte quando aplicável.

Arquivo esperado:

src/safety/response_validator.py

Retorno esperado:

{
  "approved": true,
  "warnings": [],
  "corrected_response": null
}
8.6 Logs e Auditoria

Responsável por registrar interações com estrutura mínima.

Arquivo esperado:

src/logging_audit/audit_logger.py

Destino sugerido:

outputs/audit_logs.jsonl

Campos mínimos:

{
  "timestamp": "2026-05-23T10:30:00",
  "flow": "breast_cancer",
  "risk_level": "alto",
  "sources": ["breast_cancer_screening.md"],
  "safety_status": "approved",
  "sensitive_case": false,
  "summary": "Caso sintético analisado com recomendação de avaliação especializada."
}
8.7 Fine-tuning Demonstrativo

Responsável por demonstrar:

preparação de dataset;
anonimização/sintetização;
curadoria;
formato de treinamento;
exemplo de treino ou pipeline;
avaliação por cenários.

O MVP pode usar LLM via Ollama no runtime, desde que o relatório deixe claro que o fine-tuning é entregue como pipeline demonstrativo.

Arquivos prováveis:

fine_tuning/
data/finetuning/

Narrativa correta:

O MVP executável utiliza LLM local via Ollama com RAG e guardrails. O fine-tuning é entregue como pipeline técnico demonstrativo, com dados sintéticos/curados, para mostrar o processo de especialização do modelo.

9. Fontes de Dados
9.1 Dados da Fase 2

Usados no fluxo Breast Cancer:

Breast Cancer Wisconsin Dataset;
métricas do modelo Random Forest otimizado por Algoritmo Genético;
explicações geradas por LLM;
relatório da Fase 2.

Uso:

apoiar o módulo Breast Cancer;
construir model card;
demonstrar continuidade entre Fase 2 e Fase 3.
9.2 Dados Sintéticos

Arquivos esperados:

data/synthetic/synthetic_patients.csv
data/synthetic/synthetic_breast_exam_results.csv

Esses dados devem conter exemplos fictícios de pacientes e exames.

Não devem conter dados reais.

9.3 Documentos RAG

Arquivos sugeridos:

data/protocols/assistant_safety_policy.md
data/protocols/breast_cancer_screening.md
data/protocols/cervical_cancer_screening.md
data/protocols/gynecological_triage.md
data/protocols/obstetric_warning_signs.md
data/protocols/domestic_violence_safety.md

Esses documentos devem ser curtos, rastreáveis e escritos com linguagem cautelosa.

9.4 MedQuAD

O repositório pode usar MedQuAD como base complementar de perguntas e respostas médicas.

O uso de MedQuAD é adequado para o desafio simplificado e pode apoiar o desafio Secretaria, desde que complementado por documentos curados de saúde da mulher.

10. Padrões de Segurança

Todas as respostas médicas devem obedecer às seguintes regras:

Não diagnosticar definitivamente.
Não prescrever medicamentos.
Não informar dosagem.
Não substituir avaliação profissional.
Recomendar atendimento presencial em risco alto.
Recomendar equipe qualificada em casos de violência doméstica.
Preservar confidencialidade em casos sensíveis.
Indicar fonte ou protocolo quando usar RAG.
Explicar limitações do assistente.
Usar linguagem não alarmista.
Registrar log da interação.
Tratar dados como sintéticos ou anonimizados.

Frases recomendadas:

“Este resultado não representa diagnóstico definitivo.”
“O assistente atua apenas como apoio à triagem.”
“Recomenda-se avaliação por profissional de saúde.”
“Em caso de sintomas intensos ou agravamento, procure atendimento presencial.”
“Em situações de risco ou violência, a segurança e a confidencialidade devem ser priorizadas.”

Frases proibidas:

“Você tem câncer.”
“O diagnóstico é...”
“Tome o medicamento...”
“Use X mg...”
“Não precisa procurar médico.”
“Este resultado confirma a doença.”
11. Divisão de Trabalho
11.1 Responsabilidades do Nelson

Nelson será responsável por:

módulo Breast Cancer da Fase 2;
dados sintéticos de pacientes e exames;
safety validator;
logs/auditoria estruturados;
model card da Fase 2;
documentação técnica do fluxo Breast Cancer;
apoio na narrativa de segurança e limitações;
seção do relatório sobre ML, Breast Cancer, safety e auditoria.

Arquivos/pastas sob responsabilidade preferencial:

src/tools/
src/safety/
src/logging_audit/
data/synthetic/
data/protocols/breast_cancer_screening.md
docs/phase2_breast_cancer_model_card.md
docs/breast_cancer_flow.md
outputs/audit_logs.jsonl
11.2 Responsabilidades do Colega

O colega será responsável por:

manter app Streamlit funcionando;
ajustar LangGraph existente;
manter RAG/MedQuAD/protocolos;
ajustar fine-tuning demonstrativo;
README e execução do projeto;
dependências;
relatório técnico base;
fluxo geral da aplicação.

Arquivos/pastas sob responsabilidade preferencial:

main.py
src/rag/
src/db/
src/engine/
fine_tuning/
README.md
requirements.txt
docs/relatorio_tecnico.md

Atenção: src/engine/grafo_clinico.py e src/engine/etapas_clinicas.py são arquivos de integração e devem ser alterados com cuidado. Preferencialmente, apenas uma pessoa deve aplicar mudanças nesses arquivos no final.

12. Regras de Desenvolvimento
Fazer alterações pequenas.
Fazer commits frequentes.
Não reescrever o projeto inteiro.
Não separar backend FastAPI nesta fase.
Não alterar o funcionamento atual sem necessidade.
Criar módulos novos em arquivos separados sempre que possível.
Evitar conflitos nos arquivos centrais do LangGraph.
Manter código didático, com comentários claros.
Priorizar funcionamento demonstrável.
Atualizar documentação junto com o código.
Não prometer funcionalidades que não existem.
Diferenciar claramente runtime, RAG e fine-tuning demonstrativo.
13. Entregáveis Esperados
13.1 Repositório Git

Deve conter:

código-fonte modular;
aplicação executável;
pipeline demonstrativo de fine-tuning;
integração com LangChain/RAG;
fluxos com LangGraph;
dados sintéticos;
módulos de segurança;
logs/auditoria;
README completo.
13.2 Dataset Sintético

Deve conter exemplos de:

pacientes sintéticas;
exames simulados;
perguntas e respostas;
casos de triagem;
exemplos para fine-tuning.
13.3 Relatório Técnico

Deve explicar:

objetivo do projeto;
arquitetura;
fontes de dados;
RAG e vetorização;
LangGraph e fluxos;
módulo Breast Cancer da Fase 2;
fine-tuning demonstrativo;
safety validator;
logs/auditoria;
avaliação;
limitações;
próximos passos.
13.4 Diagrama dos Fluxos

Deve mostrar:

fluxo geral do assistente;
fluxo Breast Cancer;
fluxo de prevenção;
fluxo de triagem;
fluxo obstétrico;
fluxo de violência doméstica;
validação de segurança;
logging/auditoria.
13.5 Vídeo de até 15 minutos

Deve demonstrar:

apresentação do objetivo;
arquitetura geral;
funcionamento do chat;
pergunta respondida com RAG;
fluxo Breast Cancer;
fluxo de prevenção;
exemplo de triagem ginecológica ou obstétrica;
exemplo de violência doméstica com resposta segura;
logs/auditoria;
explicação do fine-tuning demonstrativo;
limitações e próximos passos.
14. Estado Atual

O projeto possui um repositório compartilhado com:

Streamlit;
LangGraph;
RAG;
MedQuAD;
SQLite;
Ollama;
scripts de fine-tuning demonstrativo;
estrutura inicial de logs;
documentação inicial.

Pontos fortes:

base funcional já existente;
uso de LangGraph;
uso de RAG;
app visual simples;
dados sintéticos/prontuários;
aderência parcial ao desafio.

Pontos a melhorar:

integrar Breast Cancer da Fase 2;
criar ou melhorar safety validator;
estruturar logs/auditoria;
corrigir narrativa do fine-tuning;
adicionar documentos RAG curados;
atualizar README e relatório;
alinhar os fluxos ao desafio Secretaria;
garantir que a resposta não prometa diagnóstico ou prescrição.
15. Próximas Tarefas
Prioridade 1 — Congelar baseline funcional
Criar branch de trabalho.
Rodar projeto atual.
Confirmar que Streamlit abre.
Registrar baseline com commit.

Critério de aceite:

App executa sem erro crítico.
Prioridade 2 — Criar módulo Breast Cancer

Criar:

src/tools/breast_cancer_phase2_tool.py

Critério de aceite:

função recebe dados de paciente e exame;
retorna JSON padronizado;
não faz diagnóstico definitivo;
inclui limitação clínica;
inclui fontes;
inclui métricas da Fase 2.
Prioridade 3 — Criar dados sintéticos

Criar:

data/synthetic/synthetic_patients.csv
data/synthetic/synthetic_breast_exam_results.csv

Critério de aceite:

pelo menos 5 pacientes sintéticas;
pelo menos 5 exames simulados;
dados fictícios;
compatíveis com o fluxo Breast Cancer.
Prioridade 4 — Criar safety validator

Criar:

src/safety/response_validator.py

Critério de aceite:

detecta diagnóstico definitivo;
detecta prescrição/dosagem;
exige encaminhamento em risco alto;
retorna status de aprovação e warnings.
Prioridade 5 — Criar audit logger

Criar:

src/logging_audit/audit_logger.py

Critério de aceite:

salva logs JSONL;
cria pasta outputs se necessário;
registra flow, risk_level, sources, safety_status, sensitive_case e summary.
Prioridade 6 — Integrar Breast Cancer ao LangGraph

Critério de aceite:

mensagens sobre câncer de mama acionam fluxo Breast Cancer;
tool é chamada;
safety validator é aplicado;
log é salvo;
resposta aparece no app.
Prioridade 7 — Criar documentos RAG curados

Criar:

data/protocols/assistant_safety_policy.md
data/protocols/breast_cancer_screening.md
data/protocols/gynecological_triage.md
data/protocols/obstetric_warning_signs.md
data/protocols/domestic_violence_safety.md

Critério de aceite:

documentos curtos;
linguagem cautelosa;
fontes indicadas;
sem prescrição;
sem diagnóstico definitivo.
Prioridade 8 — Atualizar relatório

Atualizar:

docs/relatorio_tecnico.md

Critério de aceite:

explica arquitetura;
explica Breast Cancer;
explica RAG;
explica LangGraph;
explica fine-tuning demonstrativo;
explica safety;
explica logs;
declara limitações.
Prioridade 9 — Atualizar README

Critério de aceite:

explica instalação;
explica execução;
explica dependências;
explica fluxos;
explica limitações;
explica demonstração.
Prioridade 10 — Preparar vídeo

Critério de aceite:

roteiro pronto;
casos de teste definidos;
logs funcionando;
demonstração de até 15 minutos.
16. Critérios de Aceite Final

O projeto será considerado pronto quando:

A aplicação abrir.
O chat responder.
O RAG funcionar.
O LangGraph executar pelo menos os fluxos principais.
O fluxo Breast Cancer funcionar.
Dados sintéticos estiverem disponíveis.
Safety validator estiver aplicado.
Logs/auditoria forem gerados.
Fine-tuning demonstrativo estiver documentado.
README explicar como rodar.
Relatório técnico estiver completo.
Vídeo demonstrar o sistema.
O sistema não diagnosticar.
O sistema não prescrever.
O sistema recomendar avaliação profissional quando necessário.
17. Critério de Comunicação do Projeto

A narrativa oficial deve ser:

O FemCare AI é um MVP acadêmico de assistente de apoio à prevenção e triagem em saúde da mulher. O sistema usa RAG, LangGraph, dados sintéticos, guardrails e logs para responder perguntas contextualizadas e executar fluxos clínicos simplificados. O fluxo principal reaproveita o projeto da Fase 2 sobre Breast Cancer como uma ferramenta de apoio à classificação de atenção clínica. O assistente não diagnostica, não prescreve e não substitui profissionais de saúde.

Essa narrativa deve aparecer no README, relatório e vídeo.

