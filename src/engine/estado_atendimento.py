"""
Estado compartilhado do grafo LangGraph (ConsultasMedica / FemCare AI).

O TypedDict abaixo define os campos mínimos que circulam entre os nós do grafo.
Cada nó pode ler o state recebido e retornar um dicionário parcial com campos
atualizados — o LangGraph faz o merge automaticamente.
"""

from typing import TypedDict


class EstadoAtendimento(TypedDict):
    relato: str               # Pergunta ou relato digitado pelo usuário no Streamlit
    protocolo_seguranca: bool # True quando fluxos sensíveis (ex.: violência) são acionados
    nivel_risco: str          # VERDE | AMARELO | VERMELHO — classificação do atendimento
    resposta_final: str       # Texto Markdown exibido ao usuário no chat
