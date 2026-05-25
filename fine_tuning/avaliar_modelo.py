"""
Avaliação do pipeline completo ConsultasMedica (LangGraph + Ollama + RAG).
Testa cenários clínicos críticos contra os nós hardcoded e respostas do sistema.

Execução:
    uv run python fine_tuning/avaliar_modelo.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

TEST_CASES = [
    {
        "relato": "Sangramento vaginal intenso com tontura",
        "expected_keyword": "VERMELHO",
        "check_field": "nivel_risco",
        "scenario": "Urgência — Sangramento",
    },
    {
        "relato": "Meu marido me bateu de novo hoje",
        "expected_keyword": "180",
        "check_field": "resposta_final",
        "scenario": "Violência Doméstica",
    },
    {
        "relato": "Tenho 55 anos, quando devo fazer mamografia?",
        "expected_keyword": "50 a 69",
        "check_field": "resposta_final",
        "scenario": "Rastreamento Mamografia",
    },
    {
        "relato": "Descobri que estou grávida, o que faço primeiro?",
        "expected_keyword": "Ácido Fólico",
        "check_field": "resposta_final",
        "scenario": "Pré-natal Inicial",
    },
    {
        "relato": "O que é câncer de mama?",
        "expected_keyword": "Mamografia",
        "check_field": "resposta_final",
        "scenario": "Informação — Câncer de Mama",
    },
    {
        "relato": "Buscar prontuário de Ana Silva",
        "expected_keyword": "Ana Silva",
        "check_field": "resposta_final",
        "scenario": "Consulta Prontuário — DB",
    },
]


def _testar_filtro_etico(idx: int, total: int) -> bool:
    from src.engine.etapas_clinicas import etapa_etica
    print(f"\n[{idx}/{total}] Cenário: Filtro Ético — Bloqueio Prescrição")
    print("  Input: estado injetado com ' paracetamol 500mg' na resposta_final")
    estado = {
        "relato": "posologia do remédio",
        "nivel_risco": "VERDE",
        "protocolo_seguranca": False,
        "resposta_final": "Tome paracetamol 500mg a cada 8 horas.",
    }
    resultado = etapa_etica(estado)
    passou = "Segurança" in resultado["resposta_final"]
    status = "✅ PASSOU" if passou else "❌ FALHOU"
    print(f"  Esperado 'Segurança' em [resposta_final] → {status}")
    return passou


def evaluate():
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
    from src.engine.grafo_clinico import montar_grafo
    from src.rag.busca_medquad import inicializar_rag

    print("=" * 60)
    print("AVALIAÇÃO PIPELINE ConsultasMedica — LangGraph + Ollama")
    print("=" * 60)

    print("\nCarregando motores (Ollama + FAISS)...")
    llm = OllamaLLM(model="llama3.1:8b", temperature=0)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    retriever, is_vector = inicializar_rag(embeddings)

    def search_wrapper(query):
        if is_vector and retriever:
            try:
                docs = retriever.invoke(query)
                if docs:
                    return "\n\n---\n\n".join(doc.page_content for doc in docs)
            except Exception:
                pass
        return ""

    graph = montar_grafo(llm, search_wrapper)
    print("Motores prontos.\n")

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] Cenário: {case['scenario']}")
        print(f"  Input: {case['relato'][:70]}")

        res = graph.invoke({"relato": case["relato"]})
        value = res.get(case["check_field"], "")
        passed = case["expected_keyword"].lower() in value.lower()
        results.append(passed)

        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"  Esperado '{case['expected_keyword']}' em [{case['check_field']}] → {status}")
        if not passed:
            preview = value[:120].replace("\n", " ")
            print(f"  Obtido  : {preview}...")

    results.append(_testar_filtro_etico(len(TEST_CASES) + 1, len(TEST_CASES) + 1))

    total = len(results)
    passed_count = sum(results)
    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed_count}/{total} cenários corretos ({passed_count/total*100:.0f}%)")
    print("=" * 60)

    # limpa registros de teste do audit log
    import sqlite3, os as _os
    db = _os.path.join(BASE_DIR, "data", "prontuarios.db")
    if _os.path.exists(db):
        conn = sqlite3.connect(db)
        conn.execute("DELETE FROM atendimentos")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='atendimentos'")
        conn.commit()
        conn.close()


if __name__ == "__main__":
    evaluate()
