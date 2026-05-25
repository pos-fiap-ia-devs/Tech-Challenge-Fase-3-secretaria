# Instalação

python -m pip install --upgrade pip
python -m pip install streamlit
python -m pip install -e .

# Teste

python -m src.tools.breast_cancer_phase2_tool
python -m src.safety.response_validator
python -m src.logging_audit.audit_logger
python -m src.engine.etapa_breast_cancer

# Subir o app

python -m streamlit run main.py

# Acessar
http://192.168.123.70:8501

# Ollama
winget install --id Ollama.Ollama -e
ollama --version

ollama pull nomic-embed-text
ollama pull llama3.1:8b
ollama list



