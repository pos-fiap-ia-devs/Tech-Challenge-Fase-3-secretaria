flowchart TD

    A["Usuário digita pergunta no Streamlit"] --> B["main.py recebe o prompt"]
    B --> C["graph_app.invoke com campo relato"]
    C --> D["LangGraph em grafo_clinico.py"]

    D --> E{"Classificação da intenção"}

    E -->|caso ou exame de câncer de mama| BC["Fluxo Breast Cancer"]
    E -->|pergunta educativa ou preventiva| PREV["Fluxo Prevenção"]
    E -->|urgência clínica| URG["Fluxo Urgência"]
    E -->|violência doméstica| VIO["Fluxo Violência"]
    E -->|gestação ou obstetrícia| OBS["Fluxo Obstétrico"]
    E -->|prontuário| PRO["Fluxo Prontuário"]
    E -->|outros casos| ETI["Fluxo Ética ou fallback"]

    subgraph INIT["Inicialização em main.py"]
        I1["carregar_motores_ia"]
        I2["OllamaLLM llama3.1 8b"]
        I3["OllamaEmbeddings nomic embed text"]
        I4["inicializar_rag"]
        I5["FAISS retriever"]
        I6["buscar_contexto"]
        I7["montar_grafo"]

        I1 --> I2
        I1 --> I3
        I3 --> I4
        I4 --> I5
        I5 --> I6
        I2 --> I7
        I6 --> I7
    end

    I7 --> D

    subgraph RAG["RAG com MedQuAD, Markdown e FAISS"]
        R1["inicializar_rag embeddings_model"]
        R2["carregar MedQuAD CSV"]
        R3["carregar protocolos Markdown"]
        R4["dividir Markdown por seções"]
        R5["quebrar seções longas em chunks"]
        R6["criar Documents LangChain"]
        R7["gerar embeddings com Ollama"]
        R8["criar ou carregar índice FAISS"]
        R9["retriever com similaridade"]
        R10["retriever.invoke query"]

        R1 --> R2
        R1 --> R3
        R3 --> R4
        R4 --> R5
        R2 --> R6
        R5 --> R6
        R6 --> R7
        R7 --> R8
        R8 --> R9
        R9 --> R10
    end

    I4 --> R1
    I6 --> R10

    subgraph PREVFLOW["Fluxos gerais de prevenção e obstetrícia"]
        PREV --> P1["etapa_prevencao"]
        OBS --> O1["etapa_obstetricia"]

        P1 --> P2["search_func com relato"]
        O1 --> O2["search_func com relato"]

        P2 --> P3{"RAG encontrou contexto"}
        O2 --> O3{"RAG encontrou contexto"}

        P3 -->|sim| P4["formatar_resposta_rag"]
        O3 -->|sim| O4["formatar_resposta_rag"]

        P3 -->|não| P5["invocar_llm"]
        O3 -->|não| O5["invocar_llm"]

        P5 --> P6["llm.invoke com Ollama"]
        O5 --> O6["llm.invoke com Ollama"]
    end

    P2 --> R10
    O2 --> R10

    subgraph BCFLOW["Fluxo Breast Cancer em etapa_breast_cancer.py"]
        BC --> B1["etapa_breast_cancer search_func e state"]
        B1 --> B2["obter contexto RAG"]
        B1 --> B3["selecionar paciente sintética"]
        B3 --> B4{"Paciente encontrada"}

        B4 -->|não| B5["resposta segura de paciente não encontrada"]
        B4 -->|sim| B6["montar patient_data e exam_data"]

        B2 --> B7["build_rag_query"]
        B7 --> B8["search_func query"]
        B8 --> R10
        R10 --> B9["clean_rag_context"]
        B9 --> B10["rag_info com fonte seção e conteúdo"]

        B6 --> B11["analyze_breast_cancer_case"]
        B10 --> B12["build_protocol_reference"]
        B11 --> B13["build_integrated_interpretation"]
        B12 --> B14["build_markdown_response"]
        B13 --> B14
        B5 --> B14
    end

    subgraph ML["Predição com modelo da Fase 2"]
        B11 --> M1["extrair features do exame"]
        M1 --> M2["carregar modelo joblib"]
        M2 --> M3{"Modelo e features disponíveis"}

        M3 -->|sim| M4["montar DataFrame na ordem do modelo"]
        M4 --> M5["model.predict_proba"]
        M5 --> M6["mapear score para nível de atenção"]

        M3 -->|não| M7["fallback baseado em regras"]
        M7 --> M6

        M6 --> M8["gerar explicação da tool"]
        M8 --> M9["retornar risk_level score ação e método"]
    end

    M9 --> B13

    subgraph SAFE["Segurança e auditoria"]
        B14 --> S1["apply_safety_validation"]
        S1 --> S2["validate_medical_response"]
        S2 --> S3["write_flow_audit"]
        S3 --> S4["write_audit_log em JSONL"]
        S3 --> S5["registro em SQLite"]
    end

    subgraph OUT["Retorno ao usuário"]
        S2 --> Z1["resposta_final"]
        P4 --> Z1
        P6 --> Z1
        O4 --> Z1
        O6 --> Z1
        URG --> Z1
        VIO --> Z1
        PRO --> Z1
        ETI --> Z1
        Z1 --> Z2["Streamlit exibe resposta com st.markdown"]
        S5 --> Z3["Sidebar mostra histórico e auditoria"]
    end