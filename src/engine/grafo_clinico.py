from langgraph.graph import StateGraph, START, END
from .estado_atendimento import EstadoAtendimento
from .etapas_clinicas import (
    etapa_prevencao,
    etapa_urgencia, etapa_violencia, etapa_obstetricia, etapa_etica, etapa_prontuario,
)
from src.logger import get_logger

log = get_logger("consultas_medica.graph")

def montar_grafo(llm, search_func):
    workflow = StateGraph(EstadoAtendimento)

    workflow.add_node("prevencao", lambda s: etapa_prevencao(llm, search_func, s))
    workflow.add_node("urgencia", lambda s: etapa_urgencia(llm, search_func, s))
    workflow.add_node("violencia", lambda s: etapa_violencia(llm, search_func, s))
    workflow.add_node("obstetricia", lambda s: etapa_obstetricia(llm, search_func, s))
    workflow.add_node("prontuario", etapa_prontuario)
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
            "prontuario": "prontuario",
            "prevencao": "prevencao",
        }
    )

    for node in ["urgencia", "violencia", "obstetricia", "prevencao", "prontuario"]:
        workflow.add_edge(node, "seguranca_etica")

    workflow.add_edge("seguranca_etica", END)

    return workflow.compile()
