"""
Gera diagrama do fluxo LangGraph do Consultas Medica.
Salva em docs/diagrama_langgraph.md (Mermaid) e tenta gerar PNG se dependências disponíveis.

Execução:
    uv run python generate_diagram.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

MERMAID_MANUAL = """flowchart TD
    START([__start__]) --> identificacao
    identificacao["👤 identificacao\\nExtrai paciente do prontuário"]

    identificacao -->|sangramento / dor forte / aguda| urgencia
    identificacao -->|violencia / briga / ameaça| violencia
    identificacao -->|grávida / gestação / parto| obstetricia
    identificacao -->|demais casos| prevencao

    urgencia["🚑 urgencia\\nProtocolo FEBRASGO — emergência"]
    violencia["⚖️ violencia\\nProtocolo Lei Maria da Penha"]
    obstetricia["🤰 obstetricia\\nProtocolo Pré-Natal OMS/MS"]
    prevencao["🏥 prevencao\\nDiretrizes INCA/MS + RAG"]

    urgencia --> seguranca_etica
    violencia --> seguranca_etica
    obstetricia --> seguranca_etica
    prevencao --> seguranca_etica

    seguranca_etica["🛡️ seguranca_etica\\nFiltro ético — bloqueia prescrições"]
    seguranca_etica --> END([__end__])

    style urgencia fill:#ff4b4b,color:#fff
    style violencia fill:#9b59b6,color:#fff
    style obstetricia fill:#3498db,color:#fff
    style prevencao fill:#27ae60,color:#fff
    style seguranca_etica fill:#f39c12,color:#fff
"""


def save_mermaid(content: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Diagrama do Fluxo LangGraph — Consultas Medica\n\n")
        f.write("```mermaid\n")
        f.write(content)
        f.write("```\n")
    print(f"Mermaid salvo: {path}")


def try_generate_png(_mermaid_text: str, png_path: str):
    try:
        from langchain_core.runnables.graph import MermaidDrawMethod
        from IPython.display import Image
        _ = MermaidDrawMethod  # just check import
    except ImportError:
        pass

    # Tenta via langgraph graph.draw_mermaid_png()
    try:
        from langchain_ollama import OllamaLLM, OllamaEmbeddings
        from src.engine.graph import compile_consultas_graph
        from src.rag.core import get_rag_engine

        dummy_llm = OllamaLLM(model="llama3.2:1b", temperature=0)
        dummy_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        _, _, _ = get_rag_engine(dummy_embeddings)

        def dummy_search(q):
            return ""

        graph = compile_consultas_graph(dummy_llm, dummy_search)
        png_data = graph.get_graph().draw_mermaid_png()
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        with open(png_path, "wb") as f:
            f.write(png_data)
        print(f"PNG salvo   : {png_path}")
        return True
    except Exception as exc:
        print(f"PNG não gerado ({exc}) — use o Mermaid .md no browser/editor.")
        return False


def main():
    md_path = os.path.join(BASE_DIR, "docs", "diagrama_langgraph.md")
    png_path = os.path.join(BASE_DIR, "docs", "diagrama_langgraph.png")

    save_mermaid(MERMAID_MANUAL, md_path)
    try_generate_png(MERMAID_MANUAL, png_path)

    print("\nPara visualizar o diagrama:")
    print("  - Abra docs/diagrama_langgraph.md no VS Code (extensão Mermaid Preview)")
    print("  - Ou cole o conteúdo em https://mermaid.live")


if __name__ == "__main__":
    main()
