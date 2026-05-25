import re

from langgraph.graph import StateGraph, START, END
from .estado_atendimento import EstadoAtendimento
from .etapas_clinicas import (
    etapa_prevencao,
    etapa_urgencia, etapa_violencia, etapa_obstetricia, etapa_etica, etapa_prontuario,
)
from .etapa_breast_cancer import etapa_breast_cancer
from src.logger import get_logger

log = get_logger("consultas_medica.graph")

# Contexto mínimo para ID P00X contar como análise de caso (evita falso positivo isolado).
_BREAST_CASE_CONTEXT_TERMS = (
    "câncer de mama",
    "cancer de mama",
    "exame de mama",
    "exame",
    "wisconsin",
    "triagem",
    "analisar",
    "avaliar",
    "classificar",
    "laudo",
    "radius_mean",
    "concavity",
)

_BREAST_CASE_ANALYSIS_PHRASES = (
    "analisar exame",
    "analisar dados",
    "avaliar exame",
    "avaliar resultados",
    "avaliar resultado",
    "classificar exame",
    "resultado de exame",
    "resultados do exame",
    "resultados de exame",
    "laudo de",
    " laudo ",
    "triagem da paciente",
    "triagem breast cancer",
    "dados wisconsin",
    "atributos wisconsin",
    "exame sintético",
    "exame sintetico",
    "radius_mean",
    "texture_mean",
    "concavity_mean",
    "concave_points_mean",
)

_BREAST_EDUCATIONAL_PHRASES = (
    "me fale sobre",
    "fale sobre",
    "me explique",
    "explique sobre",
    "como prevenir",
    "quando fazer",
    "quando uma mulher",
    "quais são",
    "quais sao",
    "o que é",
    "o que e",
    "substitui",
    "deve fazer",
    "histórico familiar",
    "historico familiar",
    "prevenção de",
    "prevencao de",
    "sinais de alerta",
    "rastreamento de",
    "como funciona",
    "é importante",
    "e importante",
)

_BREAST_TOPIC_TERMS = (
    "câncer de mama",
    "cancer de mama",
    "mamografia",
    "rastreamento",
    "autoexame",
    "prevenção",
    "prevencao",
    "sinais de alerta",
    "nódulo",
    "nodulo",
    "exame de mama",
)

_PATIENT_ID_PATTERN = re.compile(r"\bp\d{3}\b", re.IGNORECASE)
_NAMED_PATIENT_PATTERN = re.compile(
    r"\bpaciente\s+[a-záàâãéèêíïóôõöúüç]+(?:\s+[a-záàâãéèêíïóôõöúüç]+)+",
    re.IGNORECASE,
)


def _normalize_relato(relato: str) -> str:
    return (relato or "").lower().strip()


def _is_breast_cancer_case_analysis(relato: str) -> bool:
    """
    True quando o relato indica análise de exame/caso/paciente sintético (fluxo breast_cancer).
    """
    text = _normalize_relato(relato)
    if not text:
        return False

    if any(phrase in text for phrase in _BREAST_CASE_ANALYSIS_PHRASES):
        return True

    if "analisar" in text and any(
        term in text for term in ("câncer de mama", "cancer de mama", "exame de mama", "exame")
    ):
        return True

    if _PATIENT_ID_PATTERN.search(text) and any(term in text for term in _BREAST_CASE_CONTEXT_TERMS):
        return True

    if _NAMED_PATIENT_PATTERN.search(relato) and any(
        verb in text
        for verb in ("analisar", "avaliar", "classificar", "triagem", "exame", "laudo")
    ):
        return True

    return False


def _is_breast_cancer_educational_question(relato: str) -> bool:
    """
    True para perguntas educativas/preventivas sobre câncer de mama (fluxo prevencao + RAG).
    """
    text = _normalize_relato(relato)
    if not text:
        return False

    if _is_breast_cancer_case_analysis(relato):
        return False

    if not any(term in text for term in _BREAST_TOPIC_TERMS):
        return False

    if any(phrase in text for phrase in _BREAST_EDUCATIONAL_PHRASES):
        return True

    # Tópico de mama sem verbos de análise de caso → orientação preventiva (ex.: "quando fazer mamografia?").
    analysis_verbs = ("analisar", "avaliar", "classificar", "laudo de", "triagem da paciente")
    if not any(verb in text for verb in analysis_verbs):
        return True

    return False

def montar_grafo(llm, search_func):
    workflow = StateGraph(EstadoAtendimento)

    workflow.add_node("prevencao", lambda s: etapa_prevencao(llm, search_func, s))
    workflow.add_node("urgencia", lambda s: etapa_urgencia(llm, search_func, s))
    workflow.add_node("violencia", lambda s: etapa_violencia(llm, search_func, s))
    workflow.add_node("obstetricia", lambda s: etapa_obstetricia(llm, search_func, s))
    workflow.add_node("prontuario", etapa_prontuario)
    workflow.add_node("breast_cancer", lambda s: etapa_breast_cancer(search_func, s))
    workflow.add_node("seguranca_etica", etapa_etica)

    def classificar_relato(state: EstadoAtendimento):
        relato = state['relato'].lower()
        if any(t in relato for t in ["sangramento", "dor forte", "aguda"]):
            log.info("Roteador → urgencia")
            return "urgencia"
        if any(p in relato for p in ["marido", "agrediu", "medo", "violencia", "briga", "ameaça", "hematoma", "bater"]):
            log.info("Roteador → violencia")
            return "violencia"
        if any(o in relato for o in ["grávida", "parto", "bebê", "gestação"]):
            log.info("Roteador → obstetricia")
            return "obstetricia"
        relato_original = state["relato"]
        if _is_breast_cancer_case_analysis(relato_original):
            log.info("Roteador → breast_cancer")
            return "breast_cancer"
        if _is_breast_cancer_educational_question(relato_original):
            log.info("Roteador → prevencao (breast_cancer_educational_question)")
            return "prevencao"
        if any(p in relato for p in ["prontuário", "prontuario", "histórico de", "buscar paciente"]):
            log.info("Roteador → prontuario")
            return "prontuario"
        log.info("Roteador → prevencao")
        return "prevencao"

    workflow.add_conditional_edges(
        START,
        classificar_relato,
        {
            "urgencia": "urgencia",
            "violencia": "violencia",
            "obstetricia": "obstetricia",
            "breast_cancer": "breast_cancer",
            "prontuario": "prontuario",
            "prevencao": "prevencao",
        }
    )

    for node in ["urgencia", "violencia", "obstetricia", "breast_cancer", "prevencao", "prontuario"]:
        workflow.add_edge(node, "seguranca_etica")

    workflow.add_edge("seguranca_etica", END)

    return workflow.compile()
