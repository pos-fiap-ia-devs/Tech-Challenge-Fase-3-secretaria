from typing import TypedDict

class EstadoAtendimento(TypedDict):
    relato: str
    protocolo_seguranca: bool
    nivel_risco: str
    resposta_final: str
