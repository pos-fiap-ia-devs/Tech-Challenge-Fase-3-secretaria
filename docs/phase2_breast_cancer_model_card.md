# Model Card — Breast Cancer RF-GA

## 1. Identificação do modelo

**Nome do modelo:** `breast_cancer_rf_ga`  
**Tipo:** Random Forest Classifier  
**Origem:** Tech Challenge FIAP — Fase 2  
**Projeto:** Otimização de modelos de diagnóstico para saúde da mulher com Algoritmo Genético e integração com LLM  
**Uso na Fase 3:** ferramenta de apoio ao fluxo Breast Cancer do FemCare AI  
**Artefatos relacionados:**

- `models/breast_cancer_rf_ga_pipeline.joblib`
- `models/breast_cancer_rf_ga_metadata.json`

---

## 2. Objetivo do modelo

Este modelo foi desenvolvido para apoiar a classificação de atenção clínica em casos relacionados à saúde da mulher, usando atributos estruturados do Breast Cancer Wisconsin Diagnostic Dataset.

O objetivo principal é identificar casos que possam exigir **maior atenção clínica**, especialmente reduzindo falsos negativos em situações onde um caso maligno poderia passar despercebido.

O modelo não realiza diagnóstico definitivo. Ele deve ser utilizado apenas como componente auxiliar de triagem, priorização e demonstração acadêmica.

---

## 3. Contexto de uso na Fase 3

Na Fase 3, o modelo será reaproveitado dentro do MVP **FemCare AI** como uma ferramenta estruturada do fluxo Breast Cancer.

O modelo será utilizado para:

- receber dados sintéticos de paciente e exame;
- calcular probabilidade associada à classe de maior atenção clínica;
- retornar nível de atenção;
- fornecer métricas globais do modelo;
- gerar uma explicação técnica resumida;
- apoiar a resposta da LLM com contexto estruturado;
- alimentar logs/auditoria;
- reforçar recomendações de avaliação profissional quando necessário.

Fluxo na Fase 3 (MVP FemCare AI):

```text
dados sintéticos de paciente e exame (CSVs em data/synthetic/)
→ etapa_breast_cancer (seleção por ID, nome, caso demo P002 ou erro se ID inexistente)
→ breast_cancer_phase2_tool
→ models/breast_cancer_rf_ga_pipeline.joblib (ou rule_based_fallback)
→ resultado estruturado
→ RAG FAISS (breast_cancer_screening.md) + síntese sem metadados na UI
→ safety validator
→ audit logger
→ resposta no Streamlit
```

### 3.1 Integração operacional na tool (`breast_cancer_phase2_tool.py`)

| Item | Detalhe |
| --- | --- |
| Artefato principal | `models/breast_cancer_rf_ga_pipeline.joblib` |
| Inferência preferencial | `inference_method="phase2_joblib_model"` quando o joblib carrega e as **30 features** do exame sintético estão disponíveis |
| Features Fase 3 | CSV com nomes em **underscore** (`concave_points_mean`, `radius_mean`, …) |
| Adaptação de nomes | Mapeamento automático para nomes com **espaço** esperados pelo pipeline (`concave points_mean`, conforme `feature_names_in_`) |
| Fallback explícito | `rule_based_fallback` se o modelo falhar, faltar dependência (ex.: scikit-learn), faltarem features ou `predict_proba` não funcionar — regras em 6 features Wisconsin |
| Transparência na UI | A etapa exibe *Método da análise:* em linguagem amigável; não expõe `joblib`, `model_path` nem listas de features na resposta |

A saída numérica é apresentada na interface como **score acadêmico de triagem**, não como diagnóstico ou probabilidade clínica confirmatória.

### 3.2 Rastreabilidade com `fallback_info` (`etapa_breast_cancer.py`)

A orquestração da Fase 3 agrega na resposta do grafo um dicionário **`fallback_info`**, incluindo `model_inference_method`, `model_loaded`, `model_inference_failed` e `model_missing_features` (estes últimos para **logs**, não para o usuário final).

Quando `inference_method` é `rule_based_fallback`, o usuário vê **regra acadêmica de contingência** — o identificador interno permanece em logs e auditoria (`summary` com sufixo `Fallback: rule_based_fallback` quando aplicável). Isso evita fallback silencioso sem prejudicar a clareza da interface.

---

## 4. Dataset utilizado

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

O dataset contém atributos numéricos extraídos de exames citológicos e morfométricos relacionados a tumores de mama.

A variável alvo foi tratada como binária:

| Valor | Classe original | Interpretação no projeto |
|---|---|---|
| 0 | Benigno | Menor atenção clínica |
| 1 | Maligno | Maior atenção clínica |

As features utilizadas incluem medidas como:

- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`
- `compactness_mean`
- `concavity_mean`
- `concave points_mean`
- `radius_worst`
- `texture_worst`
- `perimeter_worst`
- `area_worst`
- entre outras variáveis numéricas do dataset original.

A pipeline exportada espera que os dados de entrada contenham as mesmas features usadas no treinamento.

---

## 5. Preparação dos dados

As principais etapas de preparação na Fase 2 foram:

- remoção de colunas irrelevantes, como identificador;
- remoção de coluna vazia gerada na importação do CSV;
- conversão do alvo textual para valor binário;
- separação dos dados em treino, validação e teste;
- uso de pipeline com imputação e padronização;
- avaliação de modelos baseline;
- otimização da Random Forest por Algoritmo Genético.

A separação utilizada no estudo foi:

    60% treino
    20% validação
    20% teste

A validação foi usada durante a otimização dos hiperparâmetros. O conjunto de teste foi preservado para avaliação final.

---

## 6. Modelo baseline

A Random Forest foi escolhida como modelo principal para otimização por ser adequada a dados tabulares, robusta e compatível com busca de hiperparâmetros.

O modelo baseline serviu como referência para avaliar o ganho obtido após a otimização por Algoritmo Genético.

Resultados do modelo baseline no conjunto de teste:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.947368 |
| Precision | 1.000000 |
| Recall | 0.857143 |
| F1-score | 0.923077 |
| F2-score | 0.882353 |
| Specificity | 1.000000 |
| ROC-AUC | 0.996693 |
| PR-AUC | 0.994637 |

Matriz de confusão do baseline:

| Real \ Predito | Benigno | Maligno |
|---|---:|---:|
| Benigno | 72 | 0 |
| Maligno | 6 | 36 |

Interpretação:

- O baseline apresentou alta specificity.
- Não gerou falsos positivos no teste.
- Porém, deixou passar 6 casos malignos como benignos.
- Em contexto de triagem oncológica, esse número de falsos negativos é clinicamente relevante.

---

## 7. Otimização por Algoritmo Genético

O Algoritmo Genético foi usado para buscar uma melhor configuração de hiperparâmetros da Random Forest.

Cada indivíduo da população representou uma combinação de hiperparâmetros.

Genes considerados:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `class_weight`
- `threshold`

Os operadores genéticos utilizados foram:

- geração de população inicial;
- avaliação de aptidão;
- seleção por torneio;
- crossover por gene;
- mutação probabilística;
- elitismo;
- substituição da população;
- repetição por gerações.

---

## 8. Função de fitness

A função de fitness usada na otimização foi:

    fitness = 0.8 × F2-score + 0.2 × specificity

Justificativa:

- O **F2-score** dá mais peso ao recall do que à precision.
- O recall é prioritário porque falsos negativos representam casos malignos não detectados.
- A **specificity** foi mantida na função para evitar que o modelo se tornasse agressivo demais, classificando muitos casos benignos como de maior atenção.
- Essa composição busca equilibrar sensibilidade clínica com controle de falsos positivos.

O uso de F2-score foi preferido em relação ao F1-score porque o F1 trata precision e recall de forma equilibrada, enquanto o problema de triagem oncológica exige maior atenção à redução de falsos negativos.

---

## 9. Resultados do modelo otimizado

Resultados da Random Forest otimizada por Algoritmo Genético no conjunto de teste:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.964912 |
| Precision | 0.952381 |
| Recall | 0.952381 |
| F1-score | 0.952381 |
| F2-score | 0.952381 |
| Specificity | 0.972222 |
| ROC-AUC | 0.987103 |
| PR-AUC | 0.976539 |

Matriz de confusão do modelo otimizado:

| Real \ Predito | Benigno | Maligno |
|---|---:|---:|
| Benigno | 70 | 2 |
| Maligno | 2 | 40 |

Comparação principal:

| Indicador | Baseline | Otimizado |
|---|---:|---:|
| Falsos negativos | 6 | 2 |
| Verdadeiros positivos | 36 | 40 |
| Falsos positivos | 0 | 2 |
| Verdadeiros negativos | 72 | 70 |
| Recall | 0.857143 | 0.952381 |
| Specificity | 1.000000 | 0.972222 |

Interpretação:

O modelo otimizado apresentou melhora relevante na detecção de casos malignos. Os falsos negativos foram reduzidos de 6 para 2, o que representa uma redução aproximada de 67%.

Houve aumento de falsos positivos de 0 para 2, mas esse trade-off foi considerado aceitável no contexto de triagem, pois é preferível encaminhar alguns casos adicionais para avaliação do que deixar de sinalizar possíveis casos de maior atenção clínica.

---

## 10. Hiperparâmetros finais

Os hiperparâmetros finais devem ser consultados no arquivo operacional:

    models/breast_cancer_rf_ga_metadata.json

Esse arquivo é considerado a fonte de verdade para a Fase 3.

Exemplo de campos esperados:

    {
      "model_name": "breast_cancer_rf_ga",
      "model_type": "RandomForestClassifier",
      "optimization": "Genetic Algorithm",
      "best_params": {
        "n_estimators": "...",
        "max_depth": "...",
        "min_samples_split": "...",
        "min_samples_leaf": "...",
        "max_features": "...",
        "class_weight": "...",
        "threshold": "..."
      }
    }

Observação:

O parâmetro `threshold` não é hiperparâmetro interno da Random Forest. Ele deve ser usado na etapa de inferência, convertendo a probabilidade prevista em classe final.

Exemplo:

    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= threshold)

---

## 11. Uso pretendido

Este modelo pode ser usado para:

- demonstração acadêmica;
- apoio à triagem em dados sintéticos;
- classificação de nível de atenção clínica;
- integração com LLM para explicação em linguagem natural;
- integração com RAG para contextualização;
- geração de logs e auditoria;
- priorização de casos para avaliação profissional em um MVP educacional.

---

## 12. Uso não pretendido

Este modelo não deve ser usado para:

- diagnóstico definitivo;
- prescrição médica;
- recomendação de dosagem;
- substituição de avaliação profissional;
- uso clínico real sem validação externa;
- tomada de decisão médica autônoma;
- inferência sobre pacientes reais sem consentimento, governança e validação regulatória.

A saída do modelo deve sempre ser interpretada como apoio à triagem e não como conclusão diagnóstica.

---

## 13. Limitações

Principais limitações:

1. **Dataset acadêmico**  
   O modelo foi treinado com dataset público e acadêmico, não com dados reais de produção hospitalar.

2. **Ausência de histórico clínico completo**  
   O modelo usa atributos morfométricos, mas não incorpora histórico clínico, exames de imagem, biópsia, sintomas completos ou avaliação médica.

3. **Ausência de variáveis demográficas robustas**  
   O dataset não possui atributos suficientes para análise séria de equidade entre subgrupos demográficos de mulheres.

4. **Não utiliza imagens reais**  
   O modelo não analisa mamografias, ultrassons ou imagens médicas.

5. **Generalização limitada**  
   O desempenho observado pode não se reproduzir em populações diferentes, equipamentos diferentes ou ambientes clínicos reais.

6. **Uso restrito a MVP acadêmico**  
   O modelo deve ser usado apenas como demonstração educacional no contexto do Tech Challenge.

---

## 14. Considerações sobre equidade

O desafio menciona equidade entre diferentes grupos demográficos de mulheres.

No entanto, o Breast Cancer Wisconsin Dataset utilizado neste projeto não contém variáveis demográficas adequadas para uma avaliação robusta de fairness entre subgrupos populacionais, como raça, renda, região, idade em faixas clinicamente contextualizadas ou acesso prévio a serviços de saúde.

Por esse motivo, a análise de equidade foi registrada como limitação metodológica.

Em uma evolução futura, seria necessário utilizar uma base com atributos demográficos e clínicos adequados para medir desempenho por subgrupo, incluindo métricas como:

- recall por grupo;
- false negative rate por grupo;
- false positive rate por grupo;
- calibration por grupo;
- comparação de erro entre subpopulações.

---

## 15. Interpretação da saída

O modelo retorna uma probabilidade associada à classe positiva.

No contexto deste projeto:

    classe 0 → menor atenção clínica
    classe 1 → maior atenção clínica

A Fase 3 não deve apresentar a saída como “diagnóstico maligno” ou “diagnóstico benigno”.

Linguagem recomendada:

- “maior atenção clínica”
- “menor atenção clínica”
- “resultado sugere priorização”
- “recomenda avaliação profissional”
- “não representa diagnóstico definitivo”

Linguagem a evitar:

- “você tem câncer”
- “diagnóstico confirmado”
- “caso maligno confirmado”
- “não precisa procurar médico”
- “tratamento recomendado”
- “dosagem recomendada”

---

## 16. Integração com LLM

Na Fase 2, foi implementada uma integração com LLM para transformar os resultados numéricos do modelo em explicações em linguagem natural.

A LLM recebeu informações como:

- classe prevista;
- probabilidade estimada;
- métricas globais do modelo;
- falsos negativos;
- falsos positivos;
- características do caso analisado;
- instruções de segurança;
- limitação de não diagnóstico.

Na Fase 3, essa lógica deve ser preservada, mas com ainda mais cautela.

A LLM deve:

- explicar o resultado de forma técnica e breve;
- evitar diagnóstico definitivo;
- recomendar avaliação profissional em caso de risco alto;
- mencionar limitações;
- citar fontes e documentos curados;
- usar linguagem sensível e não alarmista;
- respeitar privacidade e confidencialidade.

---

## 17. Integração com Safety Validator

Toda resposta gerada a partir deste modelo deve passar pelo safety validator da Fase 3.

Regras mínimas:

- bloquear diagnóstico definitivo;
- bloquear prescrição médica;
- bloquear dosagem;
- exigir recomendação profissional em risco alto;
- exigir menção às limitações do modelo;
- alertar sobre linguagem alarmista;
- preservar confidencialidade em casos sensíveis.

---

## 18. Integração com logs e auditoria

Cada uso do modelo no fluxo Breast Cancer deve gerar um evento de auditoria.

Campos recomendados:

    {
      "timestamp": "2026-05-24T10:30:00",
      "flow": "breast_cancer",
      "risk_level": "alto",
      "sources": [
        "phase2_breast_cancer_model_card.md",
        "breast_cancer_screening.md"
      ],
      "safety_status": "approved",
      "sensitive_case": false,
      "summary": "Caso sintético analisado com modelo Random Forest da Fase 2."
    }

O log não deve armazenar dados sensíveis desnecessários.

---

## 19. Exemplo de saída estruturada esperada

```json
{
  "flow": "breast_cancer",
  "patient_id": "P002",
  "risk_level": "alto",
  "prediction_label": "maior atenção clínica",
  "probability": 0.92,
  "inference_method": "phase2_joblib_model",
  "model_metrics": {
    "recall": 0.952381,
    "specificity": 0.972222
  },
  "explanation": "O módulo indicou maior atenção com base nos atributos numéricos do exame sintético informado.",
  "limitations": "Este modelo foi treinado em dataset acadêmico e não substitui avaliação médica. A saída não representa diagnóstico definitivo.",
  "recommended_action": "Encaminhar para avaliação profissional e exames confirmatórios.",
  "sources": [
    "phase2_breast_cancer_model_card.md",
    "breast_cancer_screening.md",
    "MedQuAD/FAISS"
  ]
}
```

Quando o joblib não está operacional, `inference_method` será `rule_based_fallback` e campos como `model_path` podem refletir a indisponibilidade — comportamento documentado para transparência acadêmica, não para uso clínico.

---

## 20. Responsabilidade e governança

Este modelo deve ser tratado como componente acadêmico de apoio à decisão, não como sistema médico validado.

Qualquer uso fora do contexto educacional exigiria:

- validação clínica;
- revisão por especialistas;
- análise de segurança;
- avaliação de viés;
- governança de dados;
- adequação regulatória;
- validação externa em bases independentes.

---

## 21. Resumo executivo

O modelo Random Forest otimizado por Algoritmo Genético melhorou a capacidade de detecção de casos malignos no dataset utilizado, reduzindo falsos negativos em relação ao baseline.

A melhoria veio acompanhada de pequeno aumento de falsos positivos, considerado aceitável para um cenário de triagem, onde deixar de sinalizar um possível caso de maior atenção é mais grave do que encaminhar alguns casos adicionais para avaliação.

Na Fase 3, o modelo será utilizado como ferramenta estruturada dentro do FemCare AI, conectado a LLM, RAG, safety validator e logs/auditoria, sempre com linguagem cautelosa e sem diagnóstico definitivo.