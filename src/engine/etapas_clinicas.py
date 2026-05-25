from .estado_atendimento import EstadoAtendimento
from src.logger import get_logger
from src.db import salvar_atendimento, consultar_paciente, obter_pacientes

log = get_logger("consultas_medica.nodes")

TERMOS_PRESCRICAO = [
    "posologia", "prescrevo", "prescrito", "prescrita", " mg", " ml ",
    "gotas", " receitas", "dose", " medicar com", " paracetamol", " dipirona", " ibuprofeno"
]


def invocar_llm(llm, prompt, state):
    relato = state['relato'].lower()

    if "sangramento" in relato or "dor forte" in relato:
        log.info("invocar_llm | fallback urgência ativado")
        return (
            "🚨 **EMERGÊNCIA:** Seus sintomas indicam a necessidade de avaliação imediata. "
            "Por favor, dirija-se ao pronto-socorro ginecológico mais próximo. "
            "Não utilize medicamentos sem prescrição."
        )

    if "gravida" in relato or "gravidez" in relato or "gestante" in relato:
        log.info("invocar_llm | fallback pré-natal ativado")
        return (
            "🤰 **ACOMPANHAMENTO PRÉ-NATAL:** É fundamental manter a regularidade das consultas. "
            "Procure um médico para iniciar o acompanhamento pré-natal."
        )

    if any(t in relato for t in ["violencia", "abuso", "briga", "ameaça", "bater"]):
        log.info("invocar_llm | fallback violência ativado")
        return (
            "💜 **ACOLHIMENTO E SEGURANÇA:** Você não está sozinha. Para apoio sigiloso e orientação "
            "jurídica, ligue para o **180** (Central de Atendimento à Mulher) ou procure o CREAS. "
            "Em caso de perigo imediato, ligue 190."
        )

    try:
        log.debug("invocar_llm | chamando LLM fine-tunada")
        resposta = llm.invoke(prompt)
        return f"{resposta}\n\n---\n📚 *Source: MedQuAD/NIH*"
    except Exception as exc:
        log.warning("invocar_llm | LLM falhou (%s) — usando fallback preventivo", exc)
        return (
            "✨ **Orientação Preventiva:** Recomendamos manter seus exames de rotina em dia. "
            "Consulte seu médico para orientação personalizada."
        )


def formatar_resposta_rag(contexto: str) -> str:
    parts = []
    for chunk in contexto.split("\n\n---\n\n"):
        lines = chunk.splitlines()
        filtered = []
        for line in lines:
            if line.startswith(("Category: ", "Question: ")):
                continue
            if line.startswith("Answer: "):
                filtered.append(line[8:])
            else:
                filtered.append(line)
        text = "\n".join(filtered).strip()
        if text:
            parts.append(text)
    if not parts:
        return contexto + "\n\n---\n📚 *Source: MedQuAD/NIH*"
    return "\n\n".join(parts) + "\n\n---\n📚 *Source: MedQuAD/NIH*"


def _enriquecer_com_prontuario(relato: str) -> str:
    """Retorna bloco de contexto do prontuário se nome de paciente for detectado no relato."""
    for nome in obter_pacientes():
        if nome.lower() in relato.lower():
            paciente = consultar_paciente(nome)
            if paciente:
                log.info("Contexto prontuário injetado para: %s", nome)
                return (
                    f"\nContexto do prontuário de {paciente['nome']}:\n"
                    f"- Último Papanicolau: {paciente['ultimo_papanicolau'] or 'não registrado'}\n"
                    f"- Última Mamografia: {paciente['ultima_mamografia'] or 'não registrado'}\n"
                    f"- Histórico Familiar: {paciente['historico_familiar'] or '—'}\n"
                    f"- Observações: {paciente['observacoes'] or '—'}"
                )
    return ""


def etapa_prevencao(llm, search_func, state: EstadoAtendimento):
    relato = state['relato']
    relato_lower = relato.lower()
    log.info("NÓ: PREVENÇÃO | relato='%.60s'", relato_lower)

    ctx_paciente = _enriquecer_com_prontuario(relato)

    if "papanicolau" in relato_lower or "colo do útero" in relato_lower:
        log.info("etapa_prevencao | papanicolau match")
        resposta = (
            "### 🏥 Papanicolau — Câncer de Colo de Útero\n\n"
            "*   **Público-Alvo:** Mulheres de 25 a 64 anos que já iniciaram vida sexual.\n"
            "*   **Frequência:** Os dois primeiros exames devem ser anuais. Se ambos estiverem "
            "normais, os próximos podem ser realizados **a cada 3 anos**.\n"
            "*   **Objetivo:** Detecção precoce de lesões precursoras.\n\n"
            "---\n📚 *Source: MedQuAD/NIH*"
        )
        if ctx_paciente:
            resposta += f"\n\n---\n**📋 Dados do prontuário:**{ctx_paciente}"
        return {"nivel_risco": "VERDE", "resposta_final": resposta}

    if "mamografia" in relato_lower or "mama" in relato_lower:
        log.info("etapa_prevencao | mamografia match")
        resposta = (
            "### 🎀 Mamografia — Câncer de Mama\n\n"
            "*   **Público-Alvo:** Mulheres de 50 a 69 anos.\n"
            "*   **Frequência:** Recomendada a realização **a cada 2 anos**.\n"
            "*   **Nota:** Fora dessa faixa etária ou com histórico familiar, "
            "a indicação deve ser avaliada individualmente pelo seu médico.\n\n"
            "---\n📚 *Source: MedQuAD/NIH*"
        )
        if ctx_paciente:
            resposta += f"\n\n---\n**📋 Dados do prontuário:**{ctx_paciente}"
        return {"nivel_risco": "VERDE", "resposta_final": resposta}

    log.info("etapa_prevencao | RAG MedQuAD")
    contexto_rag = search_func(relato_lower)
    if contexto_rag:
        resposta = formatar_resposta_rag(contexto_rag)
        if ctx_paciente:
            resposta += f"\n\n---\n**📋 Dados do prontuário:**{ctx_paciente}"
        log.info("etapa_prevencao | RAG retornou contexto — bypass LLM")
        return {"nivel_risco": "VERDE", "resposta_final": resposta}

    log.info("etapa_prevencao | sem contexto RAG — LLM fallback")
    prompt = (
        f"You are a women's health clinical assistant. Answer in the same language the patient used.\n\n"
        f"Patient question: {relato}"
        + (f"\n\n{ctx_paciente}" if ctx_paciente else "") +
        "\n\nProvide a helpful, accurate, and safe clinical response. Do not prescribe medications or dosages."
    )
    return {"nivel_risco": "VERDE", "resposta_final": invocar_llm(llm, prompt, state)}


def etapa_urgencia(llm, _search_func, state: EstadoAtendimento):
    log.warning("NÓ: URGÊNCIA | relato='%.60s'", state['relato'])
    prompt = f"🚨 URGÊNCIA: {state['relato']}. Gere um aviso de 2 frases mandando o paciente para o hospital imediatamente."
    return {"nivel_risco": "VERMELHO", "resposta_final": invocar_llm(llm, prompt, state)}


def etapa_violencia(_llm, _search_func, _state: EstadoAtendimento):
    log.warning("NÓ: VIOLÊNCIA — protocolo Lei Maria da Penha ativado")
    return {
        "nivel_risco": "VERMELHO",
        "protocolo_seguranca": True,
        "resposta_final": (
            "### ⚖️ Segurança e Acolhimento\n\n"
            "Sua segurança é nossa prioridade absoluta. Você não está sozinha.\n\n"
            "*   **Ligue 180:** Central de Atendimento à Mulher (Gratuito e sigiloso).\n"
            "*   **Delegacia da Mulher:** Você pode buscar proteção legal e medidas protetivas.\n"
            "*   **Apoio Social:** Procure o CREAS mais próximo.\n\n"
            "Este canal de atendimento é seguro e sigiloso."
        ),
    }


def etapa_obstetricia(llm, search_func, state: EstadoAtendimento):
    relato = state['relato']
    relato_lower = relato.lower()
    log.info("NÓ: OBSTETRÍCIA | relato='%.60s'", relato_lower)

    ctx_paciente = _enriquecer_com_prontuario(relato)

    if any(t in relato_lower for t in ["grávida", "gestação", "prenatal", "pré-natal"]):
        log.info("etapa_obstetricia | pré-natal match")
        resposta = (
            "### 👶 Pré-Natal\n\n"
            "Parabéns por este momento! Os primeiros passos essenciais são:\n\n"
            "1. **Suplementação:** Inicie o Ácido Fólico imediatamente.\n"
            "2. **Primeira Consulta:** Agende sua consulta de pré-natal o quanto antes.\n"
            "3. **Exames Iniciais:** Prepare-se para realizar exames de sangue e ultrassom "
            "de primeiro trimestre.\n\nComo posso te ajudar?\n\n"
            "---\n📚 *Source: MedQuAD/NIH*"
        )
        if ctx_paciente:
            resposta += f"\n\n---\n**📋 Dados do prontuário:**{ctx_paciente}"
        return {"nivel_risco": "AMARELO", "resposta_final": resposta}

    log.info("etapa_obstetricia | RAG MedQuAD")
    contexto_rag = search_func(relato_lower)
    if contexto_rag:
        resposta = formatar_resposta_rag(contexto_rag)
        if ctx_paciente:
            resposta += f"\n\n---\n**📋 Dados do prontuário:**{ctx_paciente}"
        return {"nivel_risco": "AMARELO", "resposta_final": resposta}

    prompt = (
        f"You are a women's health clinical assistant specializing in obstetrics. Answer in the same language the patient used.\n\n"
        f"Patient question: {relato}"
        + (f"\n\n{ctx_paciente}" if ctx_paciente else "") +
        "\n\nProvide a helpful, accurate, and safe obstetric response. Do not prescribe medications or dosages."
    )
    return {"nivel_risco": "AMARELO", "resposta_final": invocar_llm(llm, prompt, state)}


def etapa_prontuario(state: EstadoAtendimento):
    relato = state['relato']
    log.info("NÓ: PRONTUÁRIO | relato='%.60s'", relato)

    for nome in obter_pacientes():
        if nome.lower() in relato.lower():
            paciente = consultar_paciente(nome)
            if paciente:
                resposta = (
                    f"### 📋 Prontuário — {paciente['nome']}\n\n"
                    f"*   **Data de Nascimento:** {paciente['nascimento']}\n"
                    f"*   **Último Papanicolau:** {paciente['ultimo_papanicolau'] or 'Não registrado'}\n"
                    f"*   **Última Mamografia:** {paciente['ultima_mamografia'] or 'Não registrado'}\n"
                    f"*   **Histórico Familiar:** {paciente['historico_familiar'] or '—'}\n"
                    f"*   **Observações:** {paciente['observacoes'] or '—'}\n\n"
                    f"---\n📚 *Source: prontuarios.db*"
                )
                log.info("etapa_prontuario | paciente encontrado: %s", nome)
                return {"nivel_risco": "VERDE", "resposta_final": resposta}

    log.info("etapa_prontuario | paciente não encontrado no relato")
    return {
        "nivel_risco": "VERDE",
        "resposta_final": "Paciente não encontrado no banco de dados. Verifique o nome informado.",
    }


def etapa_etica(state: EstadoAtendimento):
    log.info("NÓ: FILTRO ÉTICO")

    if state.get("nivel_risco") == "VERMELHO":
        log.info("Filtro ético: bypass — resposta de protocolo de emergência (risco VERMELHO)")
        _registrar_audit(state, state["resposta_final"])
        return {"resposta_final": state["resposta_final"]}

    resp = state["resposta_final"].lower()
    if any(t in resp for t in TERMOS_PRESCRICAO):
        log.warning("FILTRO ÉTICO BLOQUEOU resposta com termos de prescrição/dosagem")
        resposta_bloqueada = (
            "### ⚠️ Aviso de Segurança Ética\n\n"
            "Identificamos uma solicitação ou termo relacionado a dosagens/prescrições.\n\n"
            "**Como assistente de IA, não tenho autorização para prescrever medicamentos ou dosagens.**\n\n"
            "A automedicação oferece riscos graves. Por favor, consulte o seu médico para obter "
            "uma receita adequada ao seu caso."
        )
        _registrar_audit(state, resposta_bloqueada)
        return {"resposta_final": resposta_bloqueada}

    log.info("Filtro ético: resposta aprovada")
    _registrar_audit(state, state["resposta_final"])
    return {"resposta_final": state["resposta_final"]}


def _registrar_audit(state: EstadoAtendimento, resposta: str) -> None:
    relato = state.get("relato", "")
    risco = state.get("nivel_risco", "VERDE")
    try:
        salvar_atendimento("anonymous", relato, risco, resposta)
    except Exception as exc:
        log.warning("Audit DB falhou: %s", exc)
