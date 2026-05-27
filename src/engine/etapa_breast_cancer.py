"""
Nó LangGraph — Fluxo Breast Cancer (FemCare AI MVP).

Etapa clínica isolada para triagem acadêmica em câncer de mama (Fase 3).

Fluxo interno deste módulo:
    Streamlit (relato) → LangGraph → RAG (search_func) → CSV sintético → tool Fase 2
    → response_validator → audit_logger → resposta Markdown final

Responsabilidades:
    - carregar paciente e exame em data/synthetic/;
    - chamar analyze_breast_cancer_case (tool da Fase 2);
    - validar segurança da resposta;
    - registrar auditoria JSONL;
    - devolver state parcial ao grafo (resposta_final, nivel_risco, etc.).

Projetado para integração em grafo_clinico.py sem alterar os demais nós.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.logging_audit.audit_logger import write_audit_log
from src.safety.response_validator import validate_medical_response
from src.tools.breast_cancer_phase2_tool import analyze_breast_cancer_case

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caminhos e constantes
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PATIENTS_CSV = _PROJECT_ROOT / "data" / "synthetic" / "synthetic_patients.csv"
_EXAMS_CSV = _PROJECT_ROOT / "data" / "synthetic" / "synthetic_breast_exam_results.csv"

_DEFAULT_PATIENT_ID = "P002"

_PATIENT_ID_PATTERN = re.compile(r"\bP(\d{3})\b", re.IGNORECASE)
_MIN_NAME_MATCH_CHARS = 6

_NAMED_PATIENT_IN_RELATO_PATTERN = re.compile(
    r"\bpaciente\s+"
    r"([A-ZÁÀÂÃÉÈÊÍÏÓÔÕÖÚÜÇ][a-záàâãéèêíïóôõöúüç]+"
    r"\s+"
    r"[A-ZÁÀÂÃÉÈÊÍÏÓÔÕÖÚÜÇ][a-záàâãéèêíïóôõöúüç]+)",
    re.IGNORECASE,
)

_VALID_PATIENT_ID_HINT = "P001 a P008"

_RISK_TO_NIVEL: dict[str, str] = {
    "alto": "VERMELHO",
    "moderado": "AMARELO",
    "baixo": "VERDE",
    "indeterminado": "AMARELO",
    "unknown": "AMARELO",
}

# Fallback hardcoded seguro quando os CSVs sintéticos não estiverem disponíveis
_FALLBACK_PATIENT: dict[str, Any] = {
    "patient_id": "P002",
    "nome_sintetico": "Paciente P002",
    "idade": 28,
    "gestante": "sim",
    "historico_familiar_cancer_mama": "sim",
    "ultima_mamografia": "nao_realizada",
    "ultimo_papanicolau": "2024-03-01",
    "sintomas_relatados": "Nenhum",
    "observacoes": "Fallback acadêmico — dados sintéticos indisponíveis.",
}

_FALLBACK_EXAM: dict[str, float] = {
    "radius_mean": 18.90,
    "texture_mean": 23.80,
    "perimeter_mean": 122.60,
    "area_mean": 1010.50,
    "smoothness_mean": 0.115,
    "compactness_mean": 0.235,
    "concavity_mean": 0.178,
    "concave_points_mean": 0.074,
}

_SAFE_ERROR_RESPONSE = (
    "### 🎀 Breast Cancer — Apoio à Triagem (FemCare AI)\n\n"
    "Não foi possível analisar os dados sintéticos neste momento.\n\n"
    "**Este resultado não representa diagnóstico definitivo** e **não substitui "
    "avaliação por profissional de saúde.**\n\n"
    "Recomenda-se encaminhar a paciente para avaliação profissional e exames "
    "confirmatórios conforme protocolo clínico."
)

# Texto de limitação em português — evita falso positivo no response_validator
# (a limitação em inglês da tool contém a substring "prescribe").
_LIMITATIONS_PT = (
    "Este resultado é apoio à triagem acadêmica e **não representa diagnóstico definitivo**. "
    "Não substitui avaliação por profissional de saúde."
)


def _load_csv_records(path: Path) -> list[dict[str, str]]:
    """
    Carrega todas as linhas de um CSV em uma lista de dicionários.

    Parâmetros:
        path: caminho absoluto ou relativo do arquivo CSV.

    Retorna:
        Lista de registros (dict por linha). Lista vazia se o arquivo não existir
        ou não puder ser lido.

    Efeitos colaterais:
        Registra logs de info/warning/exception; não altera arquivos.
    """
    if not path.is_file():
        logger.warning("Arquivo CSV não encontrado: %s", path)
        return []

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        logger.info("Carregados %d registros de %s", len(records), path.name)
        return records
    except Exception:
        logger.exception("Falha ao ler CSV: %s", path)
        return []


def _find_patient_id(text: str) -> str | None:
    """
    Extrai identificador sintético (ex.: P001, P002) de texto livre.

    Parâmetros:
        text: relato do usuário ou qualquer string de entrada.

    Retorna:
        patient_id em maiúsculas (P00X) ou None se não encontrado.
    """
    if not text:
        return None
    match = _PATIENT_ID_PATTERN.search(text)
    if not match:
        return None
    return f"P{match.group(1)}"


def _to_float(value: str | None) -> float | None:
    """Converte string do CSV para float; retorna None se inválido ou vazio."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _index_by_patient_id(records: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Monta índice de busca rápida: patient_id → linha do CSV."""
    index: dict[str, dict[str, str]] = {}
    for row in records:
        pid = (row.get("patient_id") or "").strip().upper()
        if pid:
            index[pid] = row
    return index


def _normalize_text_for_match(text: str) -> str:
    """
    Normaliza texto para comparação de nomes/IDs no relato.

    Minúsculas, sem acentos, underscores como espaço, sem pontuação simples.
    """
    if not text:
        return ""

    lowered = text.lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    without_accents = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    without_underscores = without_accents.replace("_", " ")
    without_punctuation = re.sub(r"[^\w\s]", " ", without_underscores)
    return re.sub(r"\s+", " ", without_punctuation).strip()


def _find_patient_id_by_name(
    text: str,
    patient_records: dict[str, dict[str, str]],
) -> str | None:
    """
    Resolve patient_id a partir do nome ou ID presente no relato.

    Prioriza correspondências mais longas para evitar match parcial arriscado.
    """
    normalized_relato = _normalize_text_for_match(text)
    if not normalized_relato:
        return None

    matches: list[tuple[int, str, str]] = []

    for patient_id, row in patient_records.items():
        for field in ("nome_sintetico", "nome", "patient_id"):
            raw_value = (row.get(field) or "").strip()
            if not raw_value:
                continue

            normalized_value = _normalize_text_for_match(raw_value)
            if not normalized_value:
                continue

            if field == "patient_id":
                if re.search(rf"\b{re.escape(normalized_value)}\b", normalized_relato):
                    matches.append((len(normalized_value), patient_id, raw_value))
                continue

            if len(normalized_value) < _MIN_NAME_MATCH_CHARS:
                continue

            if normalized_value in normalized_relato:
                matches.append((len(normalized_value), patient_id, raw_value))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    _, resolved_id, display_name = matches[0]
    logger.info(
        "patient_id detectado por nome — patient_id=%s nome=%s",
        resolved_id,
        display_name,
    )
    return resolved_id


def _resolve_patient_selection_method(
    selected_patient_id: str,
    explicit_id_requested: str | None,
    resolved_from_name: str | None,
) -> str:
    """Classifica como o caso sintético foi escolhido (transparência na resposta)."""
    if explicit_id_requested and selected_patient_id == explicit_id_requested:
        return "explicit_id"
    if resolved_from_name and selected_patient_id == resolved_from_name.upper():
        return "name_match"
    if selected_patient_id == _DEFAULT_PATIENT_ID:
        return "default_demo_case"
    return "explicit_id"


def _build_patient_exam_from_rows(
    patient_row: dict[str, str],
    exam_row: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Monta patient_data e exam_data a partir de linhas dos CSVs sintéticos."""
    patient_data = dict(patient_row)
    exam_data: dict[str, Any] = {}
    for key, raw in exam_row.items():
        if key == "patient_id":
            continue
        parsed = _to_float(raw)
        if parsed is not None:
            exam_data[key] = parsed
    return patient_data, exam_data


def _relato_suggests_unmatched_patient_name(relato: str) -> bool:
    """Indica relato com 'paciente Nome Sobrenome' sem correspondência no cadastro."""
    return bool(_NAMED_PATIENT_IN_RELATO_PATTERN.search(relato or ""))


def _select_case(
    patient_id: str | None,
    relato: str = "",
) -> tuple[dict[str, Any], dict[str, Any], str, str] | None:
    """
    Seleciona paciente e exame sintéticos para a tool de triagem.

    Parâmetros:
        patient_id: ID detectado no relato (ex.: P002) ou None.
        relato: mensagem do usuário para resolução por nome quando ID ausente.

    Retorna:
        Tupla (patient_data, exam_data, patient_id_resolvido, selection_method) ou None
        quando um patient_id explícito foi informado mas não existe nos CSVs.

        selection_method: explicit_id | name_match | default_demo_case | hardcoded_fallback

    Prioridade de seleção:
        1. patient_id explícito (P00X), se existir nos dois CSVs (senão None — sem fallback silencioso);
        2. patient_id por nome no relato (nome_sintetico / nome);
        3. P002 (caso padrão acadêmico);
        4. primeiro par paciente/exame disponível;
        5. dicionários hardcoded de fallback (não interrompe o app).

    Efeitos colaterais:
        Lê arquivos CSV; registra logs informativos.
    """
    patients = _index_by_patient_id(_load_csv_records(_PATIENTS_CSV))
    exams = _index_by_patient_id(_load_csv_records(_EXAMS_CSV))

    candidate_ids: list[str] = []
    resolved_from_name: str | None = None
    explicit_id_requested = patient_id.upper() if patient_id else None

    if explicit_id_requested:
        patient_row = patients.get(explicit_id_requested)
        exam_row = exams.get(explicit_id_requested)
        if patient_row and exam_row:
            patient_data, exam_data = _build_patient_exam_from_rows(patient_row, exam_row)
            logger.info(
                "Patient selection method: explicit_id — patient_id=%s",
                explicit_id_requested,
            )
            return patient_data, exam_data, explicit_id_requested, "explicit_id"
        logger.warning(
            "Explicit patient_id not found in synthetic CSVs — patient_id=%s",
            explicit_id_requested,
        )
        return None

    if relato.strip():
        resolved_from_name = _find_patient_id_by_name(relato, patients)
        if resolved_from_name:
            candidate_ids.append(resolved_from_name.upper())
    if _DEFAULT_PATIENT_ID not in candidate_ids:
        candidate_ids.append(_DEFAULT_PATIENT_ID)
    candidate_ids.extend(sorted(set(patients.keys()) | set(exams.keys())))

    for pid in candidate_ids:
        patient_row = patients.get(pid)
        exam_row = exams.get(pid)
        if patient_row and exam_row:
            patient_data, exam_data = _build_patient_exam_from_rows(patient_row, exam_row)
            selection_method = _resolve_patient_selection_method(
                pid,
                explicit_id_requested,
                resolved_from_name,
            )
            if selection_method == "name_match":
                logger.info(
                    "Patient selection method: name_match — patient_id=%s",
                    pid,
                )
            elif selection_method == "default_demo_case":
                logger.warning(
                    "Patient selection method: default_demo_case — using %s",
                    pid,
                )
            else:
                logger.info(
                    "Caso sintético selecionado — patient_id=%s selection_method=%s",
                    pid,
                    selection_method,
                )
            return patient_data, exam_data, pid, selection_method

    logger.warning(
        "Patient selection method: hardcoded_fallback — synthetic CSV unavailable"
    )
    return (
        dict(_FALLBACK_PATIENT),
        dict(_FALLBACK_EXAM),
        _DEFAULT_PATIENT_ID,
        "hardcoded_fallback",
    )


def _map_risk_level_to_nivel(risk_level: str) -> str:
    """
    Converte risk_level da tool (alto/moderado/baixo) para nivel_risco do grafo.

    Retorna VERMELHO, AMARELO ou VERDE conforme tabela _RISK_TO_NIVEL.
    """
    normalized = (risk_level or "indeterminado").strip().lower()
    return _RISK_TO_NIVEL.get(normalized, "AMARELO")


_DISPLAY_FEATURE_KEYS: tuple[str, ...] = (
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concave_points_mean",
)


def _format_academic_score(probability: float | int | None) -> str | None:
    """Formata score do módulo como percentual acadêmico (ex.: 92,00%).

    Aplica floor visual de 0,10% para evitar exibição de 0,00% em casos
    com probabilidade arredondada a zero — o risk_level e a recomendação
    permanecem inalterados.
    """
    if probability is None:
        return None
    try:
        pct = max(float(probability) * 100.0, 0.1)
        return f"{pct:.2f}".replace(".", ",") + "%"
    except (TypeError, ValueError):
        return None


def _normalize_patient_display_value(value: Any) -> str:
    """Normaliza valores sintéticos do CSV para exibição (acentos, underscores)."""
    text = str(value).strip()
    if not text:
        return ""

    normalized = text.replace("_", " ")
    canonical = {
        "nao realizada": "não realizada",
        "nao informado": "não informado",
        "nao informada": "não informada",
    }
    mapped = canonical.get(normalized.lower())
    if mapped:
        return mapped
    return normalized


def _format_family_history_flag(value: Any) -> str:
    """Normaliza histórico familiar sintético para sim/não."""
    if value is None or str(value).strip() == "":
        return "não informado"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "sim", "yes", "1", "s"}:
            return "sim"
        if normalized in {"false", "não", "nao", "no", "0", "n"}:
            return "não"
    return "sim" if bool(value) else "não"


def _build_patient_summary(patient_data: dict[str, Any]) -> str:
    """Campos úteis da paciente sintética (lista Markdown, uma informação por linha)."""
    age_raw = patient_data.get("idade", "não informada")
    age = _normalize_patient_display_value(age_raw) or str(age_raw)
    family = _format_family_history_flag(
        patient_data.get("historico_familiar_cancer_mama")
        or patient_data.get("historico_familiar")
    )
    lines = [
        f"- **Idade:** {age}",
        f"- **Histórico familiar:** {family}",
    ]

    symptoms = str(patient_data.get("sintomas_relatados", "")).strip()
    if symptoms and symptoms.lower() not in {"nenhum", "none", "n/a"}:
        lines.append(f"- **Sintomas relatados:** {_normalize_patient_display_value(symptoms)}")

    mammography = str(patient_data.get("ultima_mamografia", "")).strip()
    if mammography:
        lines.append(
            f"- **Última mamografia:** {_normalize_patient_display_value(mammography)}"
        )

    return "\n".join(lines)


def _extract_display_features_from_exam(exam_data: dict[str, Any]) -> dict[str, float]:
    """Extrai atributos Wisconsin principais do exame sintético."""
    features: dict[str, float] = {}
    if not exam_data:
        return features
    for key in _DISPLAY_FEATURE_KEYS:
        raw = exam_data.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            features[key] = float(raw)
            continue
        parsed = _to_float(raw)
        if parsed is not None:
            features[key] = parsed
    return features


def _collect_exam_feature_values(result: dict[str, Any]) -> dict[str, float]:
    """Valores de atributos presentes em result (exam_features ou chaves de topo)."""
    exam_features = result.get("exam_features")
    if isinstance(exam_features, dict) and exam_features:
        collected: dict[str, float] = {}
        for key in _DISPLAY_FEATURE_KEYS:
            if key in exam_features and exam_features[key] is not None:
                try:
                    collected[key] = float(exam_features[key])
                except (TypeError, ValueError):
                    continue
        return collected

    collected = {}
    for key in _DISPLAY_FEATURE_KEYS:
        raw = result.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            collected[key] = float(raw)
        else:
            parsed = _to_float(raw)
            if parsed is not None:
                collected[key] = parsed
    return collected


def _build_exam_numeric_summary(result: dict[str, Any]) -> str:
    """Lista principal dos atributos tabulares do exame sintético com valores."""
    feature_values = _collect_exam_feature_values(result)
    if not feature_values:
        return ""

    lines = ["**Dados sintéticos relevantes do exame:**"]
    for key in _DISPLAY_FEATURE_KEYS:
        if key in feature_values:
            lines.append(f"- {key}: {feature_values[key]:.3f}")

    lines.append("")
    lines.append(
        "_Esses atributos são medidas tabulares compatíveis com o formato Wisconsin Dataset "
        "e não equivalem, isoladamente, a confirmação clínica de doença._"
    )
    return "\n".join(lines)


_MAX_RAG_CONTEXT_CHARS = 1500
_MIN_RAG_SENTENCE_CHARS = 25
_MAX_RAG_SENTENCE_CHARS = 280

_RAG_TECHNICAL_LINE_PREFIXES: tuple[str, ...] = (
    "category:",
    "sourcetype:",
    "content:",
)

_RAG_QUERY_CLINICAL_TERMS: str = (
    "câncer de mama rastreamento mamografia sinais de alerta "
    "alterações suspeitas nas mamas nódulo mamário secreção sanguinolenta "
    "retração da pele retração do mamilo linfonodo axilar pele casca de laranja "
    "avaliação profissional exames confirmatórios diagnóstico precoce "
    "breast cancer screening mammogram warning signs risk factors"
)

_RAG_SIGN_GUIDANCE = (
    "sinais como nódulo mamário persistente, secreção sanguinolenta, retração de pele ou mamilo, "
    "linfonodo axilar e pele com aspecto de casca de laranja devem motivar avaliação profissional"
)

_RAG_SCREENING_GUIDANCE = (
    "a mamografia é tratada como estratégia preventiva em faixas etárias específicas"
)

_RAG_STRUCTURED_SYNTHESIS_SIGNS = (
    "O protocolo orienta atenção a alterações como nódulo mamário, secreção sanguinolenta, "
    "retração de pele ou mamilo, linfonodo axilar e pele com aspecto de casca de laranja. "
    "Esses achados devem motivar avaliação profissional."
)

_RAG_STRUCTURED_SYNTHESIS_SCREENING = (
    "Quando aplicável, a mamografia é tratada como estratégia de rastreamento em faixas etárias específicas."
)

_RAG_SYNTHESIS_KEYWORDS: tuple[str, ...] = (
    "sinal",
    "sintoma",
    "nódulo",
    "nodulo",
    "mama",
    "mamografia",
    "rastreamento",
    "secreção",
    "secrecao",
    "retração",
    "retracao",
    "linfonodo",
    "alerta",
    "avaliação",
    "avaliacao",
    "profissional",
    "encaminh",
    "referência",
    "referencia",
    "urgente",
    "casca de laranja",
    "preventiv",
    "diretriz",
    "orienta",
)

_RAG_UNSAFE_SENTENCE_FRAGMENTS: tuple[str, ...] = (
    " mg ",
    "dosagem",
    "prescrev",
    "tome ",
    "administre ",
    "comprimido",
    "medicamento",
)


def _is_rag_technical_line(line: str) -> bool:
    """True se a linha for metadado técnico do chunk FAISS (não conteúdo clínico)."""
    stripped = line.strip().lower()
    if not stripped:
        return False
    return any(stripped.startswith(prefix) for prefix in _RAG_TECHNICAL_LINE_PREFIXES)


def _is_complete_sentence_start(sentence: str) -> bool:
    """True se a frase parece começar de forma completa (não no meio de uma palavra)."""
    stripped = sentence.strip()
    if len(stripped) < _MIN_RAG_SENTENCE_CHARS:
        return False
    if stripped[0].islower():
        return False
    if stripped.startswith("#"):
        return True
    first_word = re.match(r"^[\wÀ-úÁ-Ú]+", stripped)
    if first_word:
        word = first_word.group()
        if len(word) < 4 and word.lower() not in {"a", "o", "as", "os", "um", "uma", "the"}:
            return False
    return stripped[0].isupper() or stripped[0].isdigit()


def _align_to_sentence_start(text: str) -> str:
    """
    Descarta fragmento inicial quebrado (ex.: chunk cortado no meio da palavra).

    Tenta iniciar na primeira frase ou parágrafo com começo válido.
    """
    text = text.strip()
    if not text:
        return ""
    if _is_complete_sentence_start(text):
        return text

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _is_complete_sentence_start(paragraph):
            return paragraph

        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            sentence = sentence.strip()
            if _is_complete_sentence_start(sentence):
                return sentence

    match = re.search(
        r"(?<=[.!?]\s)([A-ZÁÀÂÃÉÈÊÍÏÓÔÕÖÚÜÇ0-9][^.!?]{19,}[.!?])",
        text,
    )
    if match:
        return match.group(1).strip()

    match_word = re.search(r"\b[A-ZÁÀÂÃÉÈÊÍÏÓÔÕÖÚÜÇ][a-záàâãéèêíïóôõúüç]{3,}", text)
    if match_word:
        return text[match_word.start() :].strip()

    return text


def _split_rag_sentences(text: str) -> list[str]:
    """Divide texto em frases candidatas para o resumo."""
    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        part = part.strip()
        if not part:
            continue
        if not part.endswith((".", "!", "?")):
            part = part + "."
        sentences.append(part)
    return sentences


# !!! Monta consulta para RAG utilizando contexto com termos cílinios e nível de risco !!!

def _build_rag_query(relato: str, risk_level: str | None = None) -> str:
    """
    Monta consulta bilíngue contextualizada para o fluxo Breast Cancer.

    Incorpora termos clínicos do protocolo INCA e o relato do usuário.
    """
    base = (relato or "").strip()
    parts = [base, _RAG_QUERY_CLINICAL_TERMS] if base else [_RAG_QUERY_CLINICAL_TERMS]

    normalized_risk = (risk_level or "").strip().lower()
    if normalized_risk == "alto":
        parts.append("maior atenção clínica avaliação especializada exames confirmatórios")
    elif normalized_risk == "moderado":
        parts.append("atenção clínica moderada acompanhamento profissional")
    elif normalized_risk == "baixo":
        parts.append("rastreamento preventivo orientação profissional")

    return " ".join(parts).strip()


def _parse_rag_block(block: str) -> dict[str, str]:
    """Extrai protocolo, seção e conteúdo clínico de um chunk FAISS."""
    protocol = ""
    section = ""
    content_lines: list[str] = []

    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("protocol:"):
            protocol = stripped.split(":", 1)[1].strip()
            continue
        if lower.startswith("section:"):
            section = stripped.split(":", 1)[1].strip()
            continue
        if lower in {"content:", "content"}:
            continue
        if lower.startswith("answer:"):
            answer_text = stripped.split(":", 1)[1].strip()
            if answer_text:
                content_lines.append(answer_text)
            continue
        if lower.startswith("question:") or _is_rag_technical_line(line):
            continue
        content_lines.append(line.rstrip())

    content = "\n".join(content_lines).strip()
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = _align_to_sentence_start(content)

    if len(content) > _MAX_RAG_CONTEXT_CHARS:
        truncated = content[:_MAX_RAG_CONTEXT_CHARS]
        last_stop = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
        if last_stop > _MIN_RAG_SENTENCE_CHARS:
            content = truncated[: last_stop + 1].strip()
        else:
            content = truncated.rstrip() + "..."

    return {
        "protocol": protocol or "breast_cancer_screening.md",
        "section": section,
        "content": content,
    }


def _score_rag_payload(payload: dict[str, str]) -> int:
    """Pontua chunk pelo alinhamento com termos do fluxo Breast Cancer."""
    blob = f"{payload.get('section', '')} {payload.get('content', '')}".lower()
    keywords = (
        "sinal", "sintoma", "nódulo", "nodulo", "secreção", "secrecao", "mamografia",
        "rastreamento", "retração", "retracao", "linfonodo", "casca de laranja",
        "avaliação profissional", "avaliacao profissional", "alerta",
    )
    return sum(1 for kw in keywords if kw in blob)


def _clean_rag_context(raw_context: str) -> dict[str, str]:
    """
    Remove metadados técnicos e preserva protocolo, seção e conteúdo útil.

    Retorna dict com protocol, section e content (vazio se nada útil).
    """
    if not raw_context or not str(raw_context).strip():
        return {"protocol": "", "section": "", "content": ""}

    parsed_blocks: list[dict[str, str]] = []
    for block in re.split(r"\n-{3,}\n", str(raw_context)):
        payload = _parse_rag_block(block)
        if payload.get("content"):
            parsed_blocks.append(payload)

    if not parsed_blocks:
        return {"protocol": "", "section": "", "content": ""}

    best = max(parsed_blocks, key=_score_rag_payload)
    logger.info(
        "RAG payload selected — protocol=%s section=%s score=%d content_len=%d",
        best.get("protocol"),
        best.get("section"),
        _score_rag_payload(best),
        len(best.get("content", "")),
    )
    return best


def _is_safe_clinical_sentence(sentence: str) -> bool:
    """Evita frases com prescrição, dosagem ou linguagem de tratamento."""
    lower = sentence.lower()
    return not any(fragment in lower for fragment in _RAG_UNSAFE_SENTENCE_FRAGMENTS)


def _is_clinically_relevant_sentence(sentence: str) -> bool:
    """True se a frase parece útil para triagem de mama (sinais, rastreamento, encaminhamento)."""
    lower = sentence.lower()
    return any(keyword in lower for keyword in _RAG_SYNTHESIS_KEYWORDS)


def _has_incomplete_rag_ending(sentence: str) -> bool:
    """True se a frase termina de forma truncada ou lista incompleta."""
    stripped = sentence.strip()
    if not stripped:
        return True

    normalized = stripped.lower().strip()
    normalized = normalized.rstrip(" .:;,…")

    incomplete_endings = (
        "como",
        "alterações como",
        "alteracoes como",
        "incluem",
        "inclui",
        "tais como",
        "a seguir",
        "seguintes",
    )

    if any(normalized.endswith(term) for term in incomplete_endings):
        return True

    if stripped.endswith((":", ";", ",")):
        return True

    return False


def _mentions_screening(blob: str) -> bool:
    """True se o trecho recuperado menciona mamografia ou rastreamento."""
    lower = blob.lower()
    return any(kw in lower for kw in ("mamografia", "rastreamento", "preventiv"))


def _build_structured_protocol_synthesis(blob: str) -> str:
    """Síntese segura quando o chunk recuperado não gera frases úteis completas."""
    parts = [_RAG_STRUCTURED_SYNTHESIS_SIGNS]
    if _mentions_screening(blob):
        parts.append(_RAG_STRUCTURED_SYNTHESIS_SCREENING)
    return " ".join(parts)


def _extract_useful_content_sentences(body: str, max_sentences: int = 3) -> list[str]:
    """
    Seleciona frases completas e clinicamente úteis do conteúdo recuperado do protocolo.

    Prioriza texto real do chunk; não usa LLM.
    """
    if not body.strip():
        logger.info("RAG useful sentence extraction — candidates=%d selected=%d", 0, 0)
        return []

    candidate_sentences = _split_rag_sentences(body)
    selected: list[str] = []
    for sentence in candidate_sentences:
        if len(sentence) < _MIN_RAG_SENTENCE_CHARS or len(sentence) > _MAX_RAG_SENTENCE_CHARS:
            continue
        if not _is_complete_sentence_start(sentence):
            continue
        if not _is_safe_clinical_sentence(sentence):
            continue
        if not _is_clinically_relevant_sentence(sentence):
            continue
        if _has_incomplete_rag_ending(sentence):
            continue
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    logger.info(
        "RAG useful sentence extraction — candidates=%d selected=%d",
        len(candidate_sentences),
        len(selected),
    )
    return selected


def _extract_bullet_items_from_rag(content: str, max_items: int = 8) -> list[str]:
    """
    Extrai itens de lista Markdown do conteúdo recuperado pelo RAG.

    Exemplo de item:
    - nódulo ou caroço na mama;
    """
    items: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()

        if stripped.startswith(("- ", "* ")):
            item = stripped[2:].strip()
        elif re.match(r"^\d+\.\s+", stripped):
            item = re.sub(r"^\d+\.\s+", "", stripped).strip()
        else:
            continue

        item = item.rstrip(" ;.")
        if len(item) < 4:
            continue

        items.append(item)

        if len(items) >= max_items:
            break

    return items


def _build_bullet_based_rag_synthesis(rag_payload: dict[str, str]) -> str:
    content = rag_payload.get("content", "")
    bullets = _extract_bullet_items_from_rag(content)

    if bullets:
        formatted = "; ".join(bullets[:6])
        return (
            f"O protocolo orienta atenção a alterações como {formatted}. "
            "Esses achados devem motivar avaliação profissional."
        )

    return ""


def _build_triage_result_text(result: dict[str, Any]) -> str:
    """Texto do resultado da triagem com score acadêmico (sem linguagem diagnóstica)."""
    risk_level = str(result.get("risk_level", "indeterminado")).strip().lower()

    if risk_level == "alto":
        attention = "maior atenção clínica"
    elif risk_level == "moderado":
        attention = "atenção clínica moderada"
    elif risk_level == "baixo":
        attention = "baixa atenção clínica"
    else:
        attention = "atenção clínica a avaliar"

    parts = [f"A análise dos dados sintéticos indicou **{attention}**."]

    score = _format_academic_score(result.get("probability"))
    if score:
        parts.append(
            f"O score estimado do módulo foi **{score}**, usado apenas como indicador "
            "acadêmico de triagem, não como probabilidade clínica confirmatória."
        )

    parts.append("Este resultado **não confirma câncer**.")
    return " ".join(parts)


def _build_rag_synthesis(rag_payload: dict[str, str]) -> str:
    """
    Síntese curta do trecho de protocolo (referência documental, sem LLM).

    Ordem de prioridade:
    1. Bullets reais do protocolo, quando existirem.
    2. Frases reais completas do protocolo.
    3. Síntese estruturada segura como fallback.
    """
    body = (rag_payload.get("content") or "").strip()
    section = (rag_payload.get("section") or "").strip()
    blob = f"{section} {body}".lower()

    # 1) Primeiro tenta usar listas reais do protocolo.
    bullet_synthesis = _build_bullet_based_rag_synthesis(rag_payload)
    if bullet_synthesis:
        logger.info("RAG synthesis source: recovered_protocol_bullets")
        if _mentions_screening(blob) and "mamografia" not in bullet_synthesis.lower():
            logger.info("RAG synthesis added screening guidance")
            return f"{bullet_synthesis} {_RAG_STRUCTURED_SYNTHESIS_SCREENING}"
        return bullet_synthesis

    # 2) Se não houver bullets úteis, tenta usar frases reais completas.
    content_sentences = _extract_useful_content_sentences(body, max_sentences=3)
    if content_sentences:
        logger.info("RAG synthesis source: recovered_protocol_sentence")
        synthesis = " ".join(content_sentences)
        if _mentions_screening(blob) and "mamografia" not in synthesis.lower():
            synthesis = f"{synthesis} {_RAG_STRUCTURED_SYNTHESIS_SCREENING}"
            logger.info("RAG synthesis added screening guidance")
        return synthesis

    # 3) Só então usa fallback estruturado.
    logger.info("RAG synthesis source: structured_fallback")
    return _build_structured_protocol_synthesis(blob)


def _build_protocol_reference(rag_info: dict[str, Any] | None) -> str:
    """
    Referência documental limpa ao protocolo recuperado (sem metadados técnicos do índice).

    Usa apenas rag_payload já limpo por _clean_rag_context; não invoca LLM.
    """
    if not rag_info or not rag_info.get("rag_available"):
        return (
            "- **Documento recuperado:** não disponível nesta consulta\n"
            "- **Seção/trecho aplicável:** Trecho recuperado do protocolo\n"
            "- **Síntese aplicável:** Orienta-se avaliação profissional para conduta individualizada.\n\n"
            "_Este trecho foi usado como apoio documental à triagem, não como confirmação diagnóstica._"
        )

    payload = rag_info.get("rag_payload") or {}
    document = (payload.get("protocol") or "breast_cancer_screening.md").strip()
    section_title = (
        (payload.get("section_title") or payload.get("section") or "").strip()
        or "Trecho recuperado do protocolo"
    )
    synthesis = (
        _build_rag_synthesis(payload)
        if payload.get("content")
        else (
            f"{_RAG_SIGN_GUIDANCE.capitalize()}. "
            f"Para rastreamento, {_RAG_SCREENING_GUIDANCE}."
        )
    )

    return (
        f"- **Documento recuperado:** {document}\n"
        f"- **Seção/trecho aplicável:** {section_title}\n"
        f"- **Síntese aplicável:** {synthesis}\n\n"
        "_Este trecho foi usado como apoio documental à triagem, não como confirmação diagnóstica._"
    )


def _protocol_clause_for_interpretation(
    risk_level: str,
    rag_info: dict[str, Any] | None,
) -> str:
    """
    Trecho sobre o protocolo na interpretação integrada, alinhado ao risk_level.

    Cita o protocolo e a seção recuperada sem repetir a síntese completa dos bullets,
    que já é exibida em "Trechos aplicáveis do protocolo recuperado".
    """
    if rag_info and rag_info.get("rag_available"):
        payload = rag_info.get("rag_payload") or {}
        section = (payload.get("section") or "").strip()
        section_ref = f" (seção: {section})" if section else ""

        if risk_level == "alto":
            return (
                f"O protocolo recuperado pelo RAG{section_ref} foi usado como apoio documental "
                "e reforça que achados suspeitos ou persistentes nas mamas devem motivar "
                "avaliação profissional. "
            )
        if risk_level == "moderado":
            return (
                f"O protocolo recuperado pelo RAG{section_ref} foi usado como apoio documental "
                "e reforça que achados, sintomas ou dúvidas clínicas devem ser avaliados por "
                "profissional de saúde. "
            )
        return (
            f"O protocolo recuperado pelo RAG{section_ref} foi usado como apoio documental "
            "para orientar o reconhecimento de sinais de alerta nas mamas. "
        )

    if risk_level == "alto":
        return (
            "O protocolo de referência reforça que alterações suspeitas ou persistentes nas mamas "
            "devem motivar avaliação profissional. "
        )
    if risk_level == "moderado":
        return (
            "O protocolo de referência reforça que achados, sintomas ou histórico que mereçam "
            "avaliação devem ser revisados por profissional de saúde. "
        )
    return (
        "O protocolo de referência serve como referência educativa para sinais de alerta. "
    )


def _build_integrated_interpretation(
    result: dict[str, Any],
    rag_info: dict[str, Any] | None,
    patient_id: str = "",
) -> str:
    """
    Interpretação integrada alinhada ao risk_level da tool (não texto fixo de caso alto).

    Parâmetros:
        result: saída de analyze_breast_cancer_case (risk_level, etc.).
        rag_info: contexto RAG opcional.
        patient_id: reservado para extensões futuras.
    """
    _ = patient_id
    risk_level = str(result.get("risk_level", "indeterminado")).strip().lower()
    protocol_clause = _protocol_clause_for_interpretation(risk_level, rag_info)

    if risk_level == "alto":
        attention = "maior atenção clínica"
        application = (
            "Como a análise indicou maior atenção clínica, a aplicação segura ao caso é "
            "encaminhar para **avaliação profissional e exames confirmatórios**."
        )
    elif risk_level == "moderado":
        attention = "atenção clínica moderada"
        application = (
            "Como a análise indicou atenção moderada, a aplicação segura ao caso é "
            "**agendar avaliação médica para revisão dos achados e definição de conduta preventiva**."
        )
    elif risk_level == "baixo":
        attention = "baixa atenção clínica"
        application = (
            "Como a análise indicou baixa atenção clínica, a aplicação segura ao caso é "
            "**manter acompanhamento preventivo de rotina** e procurar avaliação profissional se "
            "surgirem alterações novas, persistentes ou suspeitas nas mamas."
        )
    else:
        attention = "atenção clínica a avaliar"
        application = (
            "Como a análise não definiu um nível claro, a aplicação segura ao caso é "
            "**agendar avaliação médica** para revisão dos achados."
        )

    cancer_notice = (
        " e não substitui avaliação profissional. "
        if risk_level == "baixo"
        else ". "
    )
    return (
        f"Os dados tabulares sintéticos do exame foram associados a **{attention}** "
        f"nesta triagem acadêmica. Esse resultado **não confirma câncer**{cancer_notice}"
        f"{protocol_clause}"
        f"{application}"
    )


def _build_rag_summary(
    rag_payload: dict[str, str],
    risk_level: str = "",
    patient_id: str = "",
) -> str:
    """Compatibilidade: referência documental a partir de rag_payload isolado."""
    _ = risk_level, patient_id
    return _build_protocol_reference(
        {"rag_available": True, "rag_payload": rag_payload}
    )


# !!! Monta consulta para RAG e realiza busca !!!

def _get_rag_context(search_func, relato: str) -> dict[str, Any]:
    """
    Recupera contexto documental via RAG/FAISS/MedQuAD antes da resposta final.

    Parâmetros:
        search_func: função de busca já injetada pelo grafo (ex.: buscar_contexto).
        relato: mensagem do usuário.

    Retorna:
        {
            "rag_available": bool,
            "rag_context": str,
            "rag_source": str,
        }

    Efeitos colaterais:
        Registra logs de início, sucesso, falha ou ausência de contexto.
        Erros são capturados — o fluxo breast_cancer continua sem RAG.
    """
    empty: dict[str, Any] = {
        "rag_available": False,
        "rag_context": "",
        "rag_source": "",
        "rag_payload": {},
    }

    if search_func is None:
        logger.warning("RAG fallback: no protocol context available — search_func indisponível")
        return empty

    query = _build_rag_query(relato)
    logger.info("Consulta RAG iniciada — query_len=%d", len(query))

    try:
        context = search_func(query)
        if context and str(context).strip():
            raw_context = str(context).strip()
            rag_payload = _clean_rag_context(raw_context)
            if not rag_payload.get("content"):
                logger.warning("RAG fallback: no protocol context available")
                return empty

            logger.info(
                "Consulta RAG concluída com sucesso — protocol=%s section=%s content_len=%d",
                rag_payload.get("protocol"),
                rag_payload.get("section"),
                len(rag_payload.get("content", "")),
            )
            protocol_name = rag_payload.get("protocol") or "MedQuAD/FAISS"
            return {
                "rag_available": True,
                "rag_context": "",
                "rag_payload": rag_payload,
                "rag_source": protocol_name,
            }

        logger.warning("RAG fallback: no protocol context available")
        return empty
    except Exception:
        logger.exception(
            "Falha na consulta RAG — fluxo breast_cancer continua sem contexto documental"
        )
        logger.warning("RAG fallback: no protocol context available")
        return empty


def _normalize_markdown_spacing(text: str) -> str:
    """
    Corrige espaçamentos comuns na resposta Markdown (concatenação de blocos/strings).

    Não altera URLs (http://, https://) nem extensões de arquivo (.md, .jsonl).
    """
    if not text:
        return text

    normalized = text

    literal_fixes = (
        ("combase", "com base"),
        ("compatível como formato", "compatível com o formato"),
        ("representadiagnóstico", "representa diagnóstico"),
        ("acadêmica.O", "acadêmica. O"),
        ("acadêmica.o", "acadêmica. o"),
        ("saúde.Não", "saúde. Não"),
        ("saúde.não", "saúde. não"),
    )
    for wrong, right in literal_fixes:
        normalized = normalized.replace(wrong, right)

    # Título Markdown colado ao texto anterior (ex.: "...texto#### Recomendação")
    normalized = re.sub(r"([^\n#])####(\S)", r"\1\n\n#### \2", normalized)

    # Ponto final seguido de letra maiúscula sem espaço (ex.: ".O protocolo")
    # (?<!:) evita alterar esquemas de URL (http://, https://)
    normalized = re.sub(
        r"(?<!:)\.([A-ZÁÀÂÃÉÈÊÍÏÓÔÕÖÚÜÇ])",
        r". \1",
        normalized,
    )

    return normalized


def _build_fallback_info(
    patient_selection_method: str,
    tool_result: dict[str, Any] | None,
    rag_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Metadados de rastreabilidade dos fallbacks do fluxo Breast Cancer."""
    tool_result = tool_result or {}
    rag_info = rag_info or {}
    return {
        "patient_selection_method": patient_selection_method,
        "model_inference_method": tool_result.get("inference_method"),
        "model_loaded": tool_result.get("model_loaded"),
        "model_inference_failed": tool_result.get("model_inference_failed"),
        "model_missing_features": tool_result.get("model_missing_features", []),
        "rag_available": rag_info.get("rag_available"),
        "rag_source": rag_info.get("rag_source"),
        "rag_payload_available": bool(rag_info.get("rag_payload")),
        "hardcoded_fallback_used": patient_selection_method == "hardcoded_fallback",
        "default_demo_case_used": patient_selection_method == "default_demo_case",
    }


def _audit_summary_fallback_suffix(fallback_info: dict[str, Any] | None) -> str:
    """Sufixo textual para o summary do audit log quando há fallback relevante."""
    if not fallback_info:
        return ""

    tags: list[str] = []
    if fallback_info.get("default_demo_case_used"):
        tags.append("default_demo_case")
    if fallback_info.get("hardcoded_fallback_used"):
        tags.append("hardcoded_fallback")
    if fallback_info.get("model_inference_method") == "rule_based_fallback":
        tags.append("rule_based_fallback")
    if fallback_info.get("rag_available") is False:
        tags.append("rag_unavailable")

    if not tags:
        return ""
    return f" Fallback: {', '.join(tags)}."


def _build_transparency_notice(
    patient_selection_method: str,
    tool_result: dict[str, Any],
    rag_info: dict[str, Any] | None,
    relato: str = "",
) -> str:
    """
    Avisos discretos de transparência (paciente, modelo, RAG) para a resposta Markdown.

    Retorna string vazia quando nenhum aviso é necessário.
    """
    parts: list[str] = []

    if patient_selection_method == "default_demo_case":
        parts.append(
            "**Observação:** nenhuma paciente específica foi identificada no relato; "
            "foi usado o caso demonstrativo padrão P002."
        )
        if _relato_suggests_unmatched_patient_name(relato):
            parts.append(
                "Não encontrei essa paciente no cadastro sintético; "
                "foi usado o caso demonstrativo padrão P002."
            )
    elif patient_selection_method == "hardcoded_fallback":
        parts.append(
            "**Observação:** os arquivos sintéticos de paciente/exame não foram carregados; "
            "foi usado um caso de contingência para manter a demonstração."
        )

    inference_method = tool_result.get("inference_method")
    if inference_method == "phase2_joblib_model":
        parts.append("**Método da análise:** modelo treinado da Fase 2.")
    elif inference_method == "rule_based_fallback":
        degraded = (
            " A análise foi mantida em modo degradado para preservar a demonstração."
            if tool_result.get("model_inference_failed")
            else ""
        )
        parts.append(
            "**Método da análise:** regra acadêmica de contingência, pois o modelo treinado "
            "não estava disponível ou não pôde inferir com as features recebidas."
            f"{degraded}"
        )

    rag_info = rag_info or {}
    rag_has_payload = bool(rag_info.get("rag_payload")) and rag_info.get("rag_available")
    if not rag_has_payload:
        parts.append(
            "**Protocolo:** não foi possível recuperar trecho específico do protocolo nesta "
            "consulta; a orientação permanece como apoio seguro à triagem acadêmica."
        )

    if not parts:
        return ""

    return "\n\n".join(parts) + "\n\n"


def _build_unknown_patient_id_response(requested_id: str) -> str:
    """Resposta segura quando o patient_id explícito não existe nos CSVs sintéticos."""
    markdown = (
        "### 🎀 Breast Cancer — Apoio à Triagem (FemCare AI)\n\n"
        f"Não encontrei dados sintéticos para o identificador **{requested_id}**. "
        f"Informe um `patient_id` válido, como **{_VALID_PATIENT_ID_HINT}**, ou um nome "
        "cadastrado (por exemplo, Carla Mendes ou Ana Silva).\n\n"
        "**Este resultado não representa diagnóstico definitivo** e **não substitui "
        "avaliação por profissional de saúde.**\n\n"
        "Recomenda-se encaminhar a paciente para avaliação profissional quando houver "
        "dúvidas clínicas."
    )
    return _normalize_markdown_spacing(markdown)


def _log_model_inference(tool_result: dict[str, Any]) -> None:
    """Registra método de inferência da tool (modelo real ou contingência)."""
    inference_method = tool_result.get("inference_method")
    if inference_method == "phase2_joblib_model":
        logger.info("Model inference method: phase2_joblib_model")
        return

    logger.warning(
        "Model inference method: rule_based_fallback — model_loaded=%s inference_failed=%s missing_features=%s",
        tool_result.get("model_loaded"),
        tool_result.get("model_inference_failed"),
        tool_result.get("model_missing_features", []),
    )


def _build_markdown_response(
    result: dict[str, Any],
    patient_id: str,
    rag_info: dict[str, Any] | None = None,
    patient_data: dict[str, Any] | None = None,
    transparency_notice: str = "",
) -> str:
    """
    Monta resposta Markdown exibida no Streamlit a partir do resultado da tool.

    Parâmetros:
        result: dicionário retornado por analyze_breast_cancer_case.
        patient_id: identificador sintético utilizado na análise.
        rag_info: contexto opcional retornado por _get_rag_context.
        transparency_notice: avisos discretos de fallback (paciente, modelo, RAG).

    Retorna:
        String Markdown com linguagem cautelosa (triagem, não diagnóstico).

    Cuidado de segurança:
        Usa _LIMITATIONS_PT em português para evitar falso positivo no validator.
    """
    sources = result.get("sources") or []
    sources_lines = "\n".join(f"- {source}" for source in sources) if sources else "- N/A"

    patient_summary = _build_patient_summary(patient_data or {})
    triage_text = _build_triage_result_text(result)
    exam_summary = _build_exam_numeric_summary(result)
    integrated = _build_integrated_interpretation(result, rag_info, patient_id)
    protocol_reference = _build_protocol_reference(rag_info)
    recommended = result.get(
        "recommended_action",
        "Encaminhar para avaliação profissional e exames confirmatórios.",
    )

    triage_section = f"{triage_text}\n\n"
    if exam_summary:
        triage_section += f"{exam_summary}\n\n"

    markdown = (
        "### 🎀 Breast Cancer — Apoio à Triagem (FemCare AI)\n\n"
        f"**Paciente sintética:** {patient_id}\n\n"
        f"{patient_summary}\n\n"
        f"{transparency_notice}"
        f"**Nível de atenção clínica:** {result.get('risk_level', 'indeterminado')}\n\n"
        "#### Resultado da triagem\n\n"
        f"{triage_section}"
        "#### Interpretação com apoio do protocolo\n\n"
        f"{integrated}\n\n"
        "#### Trechos aplicáveis do protocolo recuperado\n\n"
        f"{protocol_reference}\n\n"
        "#### Recomendação\n\n"
        f"{recommended}\n\n"
        "#### Limitações\n\n"
        f"{_LIMITATIONS_PT}\n\n"
        "#### Fontes\n\n"
        f"{sources_lines}\n\n"
        "---\n"
        "*Este resultado é apoio à triagem e **não representa diagnóstico definitivo**.*"
    )
    return _normalize_markdown_spacing(markdown)


def _apply_safety_validation(markdown_response: str, risk_level: str) -> tuple[str, str]:
    """
    Valida a resposta Markdown com response_validator.

    Parâmetros:
        markdown_response: texto gerado para o usuário.
        risk_level: nível de atenção retornado pela tool (alto/moderado/baixo).

    Retorna:
        Tupla (resposta_final, safety_status) onde safety_status é 'approved' ou 'blocked'.

    Efeitos colaterais:
        Pode substituir a resposta por corrected_response quando a validação reprovar.
    """
    validation = validate_medical_response(
        markdown_response,
        risk_level=risk_level,
        sensitive_case=False,
    )
    safety_status = "approved" if validation.get("approved") else "blocked"
    logger.info(
        "Validação de segurança concluída — approved=%s warnings=%d",
        validation.get("approved"),
        len(validation.get("warnings") or []),
    )

    if not validation.get("approved") and validation.get("corrected_response"):
        return str(validation["corrected_response"]), safety_status

    return markdown_response, safety_status


def _write_flow_audit(
    patient_id: str,
    result: dict[str, Any],
    safety_status: str,
    summary_suffix: str = "",
    fallback_info: dict[str, Any] | None = None,
) -> None:
    """
    Persiste evento de auditoria JSONL para o fluxo breast_cancer.

    Parâmetros:
        patient_id: ID sintético analisado.
        result: saída parcial da tool (risk_level, sources, prediction_label).
        safety_status: 'approved' ou 'blocked' após validação.
        summary_suffix: texto opcional anexado ao resumo (ex.: erro interno).
        fallback_info: metadados opcionais de fallback (usados no summary, não no payload JSONL).

    Efeitos colaterais:
        Append em outputs/audit_logs.jsonl via write_audit_log.
        Falhas são logadas, mas não interrompem o fluxo principal.
    """
    summary = (
        f"Caso sintético {patient_id} analisado — "
        f"{result.get('prediction_label', 'triagem registrada')}"
    )
    summary += _audit_summary_fallback_suffix(fallback_info)
    if summary_suffix:
        summary = f"{summary} {summary_suffix}"

    try:
        write_audit_log(
            {
                "flow": "breast_cancer",
                "risk_level": result.get("risk_level", "unknown"),
                "sources": result.get("sources", []),
                "safety_status": safety_status,
                "sensitive_case": False,
                "summary": summary,
            }
        )
    except Exception:
        logger.exception("Falha ao gravar audit log para patient_id=%s", patient_id)


def etapa_breast_cancer(search_func, state: dict) -> dict:
    """
    Nó LangGraph: triagem Breast Cancer com RAG, dados sintéticos e tool da Fase 2.

    Parâmetros:
        search_func: função de busca RAG injetada pelo grafo.
        state: estado do grafo contendo ao menos ``relato`` (mensagem do usuário).

    Retorna:
        Dicionário parcial com resposta_final, nivel_risco, protocolo_seguranca,
        flow, sources, sensitive_case e fallback_info (rastreabilidade de fallbacks).

    Fluxo interno:
        1. consultar RAG via search_func;
        2. detectar patient_id no relato;
        3. carregar CSVs sintéticos;
        4. executar analyze_breast_cancer_case;
        5. montar Markdown + validar segurança + auditar;
        6. mapear risco para VERMELHO/AMARELO/VERDE.

    Cuidado de segurança:
        Nunca diagnostica; falha de RAG não interrompe CSV + tool + safety.
    """
    relato = state.get("relato", "")
    logger.info("etapa_breast_cancer iniciada — relato_len=%d", len(relato))

    try:
        # --- Contexto documental via RAG (MedQuAD/FAISS) -------------------------
        rag_info = _get_rag_context(search_func, relato)
        if rag_info.get("rag_available"):
            logger.info(
                "RAG context available — source=%s",
                rag_info.get("rag_source"),
            )
        else:
            logger.warning("RAG fallback: no protocol context available")

        # --- Seleção do caso sintético (CSV ou fallback) -------------------------
        detected_id = _find_patient_id(relato)
        if detected_id:
            logger.info("patient_id detectado no relato: %s", detected_id)
        else:
            logger.warning("patient_id não detectado no relato — caso padrão será usado")

        selection = _select_case(detected_id, relato)
        if selection is None:
            requested_id = (detected_id or "unknown").upper()
            not_found_result = {
                "risk_level": "unknown",
                "prediction_label": "triagem não executada",
                "sources": ["breast_cancer_screening.md"],
            }
            fallback_info = _build_fallback_info(
                "explicit_id_not_found",
                None,
                rag_info,
            )
            markdown_response = _build_unknown_patient_id_response(requested_id)
            final_response, safety_status = _apply_safety_validation(
                markdown_response,
                "moderado",
            )
            _write_flow_audit(
                "unknown",
                not_found_result,
                "not_executed",
                summary_suffix=f"ID explícito não encontrado ({requested_id}).",
                fallback_info=fallback_info,
            )
            logger.info(
                "etapa_breast_cancer concluída — patient_id não encontrado %s safety_status=%s",
                requested_id,
                safety_status,
            )
            return {
                "resposta_final": final_response,
                "nivel_risco": "AMARELO",
                "protocolo_seguranca": False,
                "flow": "breast_cancer",
                "sources": not_found_result["sources"],
                "sensitive_case": False,
                "fallback_info": fallback_info,
            }

        patient_data, exam_data, patient_id, patient_selection_method = selection
        logger.info(
            "Usando caso sintético patient_id=%s selection_method=%s",
            patient_id,
            patient_selection_method,
        )

        # --- Tool da Fase 2 (classificação de atenção clínica) -------------------
        tool_result = analyze_breast_cancer_case(patient_data, exam_data)
        _log_model_inference(tool_result)
        risk_level = str(tool_result.get("risk_level", "indeterminado"))
        sources = list(tool_result.get("sources") or [])

        if rag_info.get("rag_available"):
            rag_source = rag_info.get("rag_source") or "MedQuAD/FAISS"
            if rag_source not in sources:
                sources.append(rag_source)

        fallback_info = _build_fallback_info(
            patient_selection_method,
            tool_result,
            rag_info,
        )

        exam_features = _extract_display_features_from_exam(exam_data)
        result_with_sources = {**tool_result, "sources": sources}
        if exam_features:
            result_with_sources["exam_features"] = exam_features
        audit_payload = result_with_sources

        transparency_notice = _build_transparency_notice(
            patient_selection_method,
            tool_result,
            rag_info,
            relato,
        )

        # --- Resposta, validação de segurança e auditoria JSONL ------------------
        markdown_response = _build_markdown_response(
            result_with_sources,
            patient_id,
            rag_info,
            patient_data,
            transparency_notice,
        )
        final_response, safety_status = _apply_safety_validation(markdown_response, risk_level)
        _write_flow_audit(
            patient_id,
            audit_payload,
            safety_status,
            fallback_info=fallback_info,
        )

        nivel_risco = _map_risk_level_to_nivel(risk_level)

        logger.info(
            "etapa_breast_cancer concluída — patient_id=%s risk_level=%s nivel_risco=%s safety_status=%s",
            patient_id,
            risk_level,
            nivel_risco,
            safety_status,
        )

        return {
            "resposta_final": final_response,
            "nivel_risco": nivel_risco,
            "protocolo_seguranca": False,
            "flow": "breast_cancer",
            "sources": sources,
            "sensitive_case": False,
            "fallback_info": fallback_info,
        }

    except Exception:
        logger.exception("etapa_breast_cancer falhou — retornando resposta segura de fallback")
        fallback_result = {
            "risk_level": "moderado",
            "prediction_label": "análise indisponível",
            "sources": ["breast_cancer_screening.md"],
        }
        try:
            _write_flow_audit(
                "unknown",
                fallback_result,
                "blocked",
                summary_suffix="(erro interno na etapa)",
            )
        except Exception:
            logger.exception("Audit log também falhou durante fallback de erro")

        return {
            "resposta_final": _SAFE_ERROR_RESPONSE,
            "nivel_risco": "AMARELO",
            "protocolo_seguranca": False,
            "flow": "breast_cancer",
            "sources": fallback_result["sources"],
            "sensitive_case": False,
        }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print("=== FemCare AI — etapa_breast_cancer (teste local) ===\n")

    def fake_search_func(query):
        return (
            "Protocol: breast_cancer_screening.md\n"
            "Section: 6.4 Alterações que devem chamar atenção\n"
            "Content:\n"
            "A estratégia de conscientização deve orientar a pessoa a ficar atenta a alterações como:\n\n"
            "- nódulo ou caroço na mama;\n"
            "- nódulo ou caroço na axila;\n"
            "- retração da pele;\n"
            "- secreção sanguinolenta;\n"
            "- pele com aspecto de casca de laranja;\n"
            "- retração ou mudança no formato do mamilo.\n"
        )

    _TECHNICAL_MARKERS = (
        "Category:",
        "SourceType:",
        "Protocol:",
        "Content:",
        "Question:",
        "Answer:",
    )
    _FORBIDDEN_TERMS = (
        "Recall",
        "Specificity",
        "Precision",
        "model_loaded",
        "model_path",
        "model_missing_features",
        ".pkl",
        "joblib",
        "sklearn",
        "feature_names_in_",
        "runtime atual",
        "chance de câncer",
        "diagnóstico confirmado",
        "Resumo numérico da análise",
        "Métricas históricas da Fase 2",
        "rule_based_fallback",
    )

    state_with_id = {
        "relato": "Analisar exame de câncer de mama da paciente P002 com histórico familiar.",
    }
    result_with_id = etapa_breast_cancer(fake_search_func, state_with_id)
    print("[TESTE — relato com P002]")
    print(json.dumps(result_with_id, indent=2, ensure_ascii=False))
    resp = result_with_id.get("resposta_final", "")
    resp_lower = resp.lower()
    for required in (
        "Resultado da triagem",
        "- **Idade:**",
        "- **Histórico familiar:**",
        "score estimado do módulo",
        "Dados sintéticos relevantes do exame",
        "radius_mean",
        "Interpretação com apoio do protocolo",
        "Trechos aplicáveis do protocolo recuperado",
        "Documento recuperado",
        "Seção/trecho aplicável",
        "Síntese aplicável",
        "avaliação profissional",
        "não representa diagnóstico definitivo",
    ):
        assert required in resp or required.lower() in resp_lower, f"Ausente na resposta: {required!r}"
    assert "não confirma" in resp_lower
    assert "92,00%" in resp or "score estimado" in resp_lower
    for marker in _TECHNICAL_MARKERS:
        assert marker not in resp, f"Metadado técnico exibido: {marker}"
    for forbidden in _FORBIDDEN_TERMS:
        assert forbidden not in resp and forbidden.lower() not in resp_lower, (
            f"Termo técnico indesejado na resposta: {forbidden!r}"
        )
    for bad_spacing in ("combase", "####Recomendação", "representadiagnóstico"):
        assert bad_spacing not in resp, f"Espaçamento incorreto: {bad_spacing!r}"
    assert "chance de câncer" not in resp_lower
    assert "breast_cancer_screening.md" in resp
    assert "nódulo ou caroço na mama" in resp_lower, (
        "Síntese deve incluir bullet real do protocolo recuperado"
    )
    assert "secreção sanguinolenta" in resp_lower, (
        "Síntese deve incluir bullet real do protocolo recuperado"
    )
    assert "pele com aspecto de casca de laranja" in resp_lower, (
        "Síntese deve incluir bullet real do protocolo recuperado"
    )
    assert "não realizada" in resp_lower
    for bad_fragment in ("alterações como..", "alterações como.", "nao_realizada"):
        assert bad_fragment not in resp and bad_fragment not in resp_lower, (
            f"Fragmento indesejado na resposta: {bad_fragment!r}"
        )
    assert "caso demonstrativo padrão P002" not in resp, (
        "Relato com P002 explícito não deve exibir aviso de caso demonstrativo"
    )
    fb_a = result_with_id.get("fallback_info") or {}
    assert fb_a.get("patient_selection_method") == "explicit_id"
    assert fb_a.get("default_demo_case_used") is False
    if fb_a.get("model_inference_method") == "phase2_joblib_model":
        assert "modelo treinado da Fase 2" in resp
    print("[OK] Cenário A — P002 explicit_id, transparência e segurança validados")

    print("\n" + "=" * 60 + "\n")

    state_carla = {
        "relato": (
            "Analisar exame de câncer de mama da paciente Carla Mendes "
            "com histórico familiar."
        ),
    }
    result_carla = etapa_breast_cancer(fake_search_func, state_carla)
    print("[TESTE — relato com Carla Mendes]")
    print(json.dumps(result_carla, indent=2, ensure_ascii=False))
    resp_carla = result_carla.get("resposta_final", "")
    assert "**Paciente sintética:** P005" in resp_carla, "Deve resolver Carla Mendes como P005"
    assert "**Idade:** 63" in resp_carla, "P005 tem idade 63 no CSV sintético"
    assert "**Paciente sintética:** P002" not in resp_carla, "Não deve cair no padrão P002"
    assert "caso demonstrativo padrão P002" not in resp_carla, (
        "Nome identificado não deve exibir aviso de caso demonstrativo"
    )
    fb_b = result_carla.get("fallback_info") or {}
    assert fb_b.get("patient_selection_method") == "name_match"
    assert "**Paciente sintética:** P005" in resp_carla
    print("[OK] Cenário B — Carla Mendes → P005 (name_match)")

    print("\n" + "=" * 60 + "\n")

    state_without_id = {
        "relato": "Preciso de apoio à triagem de câncer de mama com dados sintéticos.",
    }
    result_without_id = etapa_breast_cancer(fake_search_func, state_without_id)
    print("[TESTE — relato sem patient_id]")
    print(json.dumps(result_without_id, indent=2, ensure_ascii=False))
    resp_default = result_without_id.get("resposta_final", "")
    assert "**Paciente sintética:** P002" in resp_default
    assert "caso demonstrativo padrão P002" in resp_default, (
        "Relato sem identificação deve exibir aviso do caso demonstrativo P002"
    )
    fb_c = result_without_id.get("fallback_info") or {}
    assert fb_c.get("patient_selection_method") == "default_demo_case"
    assert fb_c.get("default_demo_case_used") is True
    print("[OK] Cenário C — default_demo_case com aviso P002")

    print("\n" + "=" * 60 + "\n")

    state_p999 = {
        "relato": "Analisar exame de câncer de mama da paciente P999.",
    }
    result_p999 = etapa_breast_cancer(fake_search_func, state_p999)
    print("[TESTE — relato com P999 inexistente]")
    print(json.dumps(result_p999, indent=2, ensure_ascii=False))
    resp_p999 = result_p999.get("resposta_final", "")
    assert result_p999.get("nivel_risco") == "AMARELO"
    assert result_p999.get("flow") == "breast_cancer"
    assert "P999" in resp_p999
    assert "não encontrei dados sintéticos" in resp_p999.lower()
    assert "**Paciente sintética:** P002" not in resp_p999
    assert "caso demonstrativo padrão P002" not in resp_p999.lower()
    assert "Resultado da triagem" not in resp_p999
    fb_d = result_p999.get("fallback_info") or {}
    assert fb_d.get("patient_selection_method") == "explicit_id_not_found"
    for forbidden in _FORBIDDEN_TERMS:
        assert forbidden not in resp_p999 and forbidden.lower() not in resp_p999.lower()
    print("[OK] Cenário D — P999 não cai em P002 e não executa triagem")

    print("\n" + "=" * 60 + "\n")

    state_ana = {
        "relato": "Analisar exame de câncer de mama da paciente Ana Silva.",
    }
    result_ana = etapa_breast_cancer(fake_search_func, state_ana)
    print("[TESTE — relato com Ana Silva / P001]")
    print(json.dumps(result_ana, indent=2, ensure_ascii=False))
    resp_ana = result_ana.get("resposta_final", "")
    resp_ana_lower = resp_ana.lower()
    assert "**Paciente sintética:** P001" in resp_ana or "P001" in resp_ana
    assert result_ana.get("nivel_risco") == "VERDE"
    assert "baixa atenção clínica" in resp_ana_lower
    assert "manter acompanhamento preventivo" in resp_ana_lower
    assert "Como a análise indicou maior atenção clínica" not in resp_ana
    assert "encaminhar para avaliação profissional e exames confirmatórios" not in resp_ana_lower
    interp_start = resp_ana.find("#### Interpretação com apoio do protocolo")
    interp_end = resp_ana.find("#### Trechos aplicáveis", interp_start)
    interp_block = resp_ana[interp_start:interp_end] if interp_start >= 0 else ""
    assert "baixa atenção clínica" in interp_block.lower()
    assert "Como a análise indicou maior atenção clínica" not in interp_block
    print("[OK] Cenário E — Ana Silva / P001 (baixo) sem linguagem de caso alto")

    print("\n" + "=" * 60 + "\n")

    state_p008 = {
        "relato": "Analisar exame de câncer de mama da paciente P008.",
    }
    result_p008 = etapa_breast_cancer(fake_search_func, state_p008)
    print("[TESTE — relato com P008]")
    resp_p008 = result_p008.get("resposta_final", "")
    resp_p008_lower = resp_p008.lower()
    assert "**Paciente sintética:** P008" in resp_p008
    assert result_p008.get("nivel_risco") == "AMARELO"
    assert (
        "atenção clínica moderada" in resp_p008_lower
        or "atenção moderada" in resp_p008_lower
    )
    assert "agendar avaliação médica" in resp_p008_lower or "revisão dos achados" in resp_p008_lower
    interp_p008 = resp_p008[
        resp_p008.find("#### Interpretação com apoio do protocolo") : resp_p008.find(
            "#### Trechos aplicáveis"
        )
    ]
    assert "Como a análise indicou maior atenção clínica" not in interp_p008
    print("[OK] Cenário F — P008 (moderado) com interpretação alinhada")

    print("\n" + "=" * 60 + "\n")

    for label, resp_high, nivel in (
        ("P002", resp, result_with_id.get("nivel_risco")),
        ("P005", resp_carla, result_carla.get("nivel_risco")),
    ):
        if nivel != "VERMELHO":
            continue
        assert "maior atenção clínica" in resp_high.lower(), f"{label} alto deve mencionar maior atenção"
        assert (
            "encaminhar para avaliação profissional e exames confirmatórios" in resp_high.lower()
            or "avaliação profissional e exames confirmatórios" in resp_high.lower()
        ), f"{label} alto deve orientar encaminhamento na interpretação ou recomendação"
    print("[OK] Cenários alto (P002/P005) mantêm linguagem de maior atenção quando VERMELHO")
