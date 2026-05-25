"""
Validador de segurança de respostas médicas — FemCare AI MVP.

Guardrail rule-based para saídas do assistente acadêmico de triagem.
Verifica diagnóstico definitivo, prescrição, dosagem, encaminhamento profissional,
confidencialidade e presença de limitação clínica.

Este módulo **não substitui** revisão humana; apoia conformidade de segurança do MVP.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Padrões bloqueados / exigidos (busca por substring, case-insensitive)
# ---------------------------------------------------------------------------

# Grupo 1 — Diagnóstico definitivo: frases que afirmam doença confirmada
_DEFINITIVE_DIAGNOSIS_PATTERNS: tuple[str, ...] = (
    "você tem câncer",
    "voce tem cancer",
    "you have cancer",
    "o diagnóstico é",
    "o diagnostico e",
    "the diagnosis is",
    "diagnóstico definitivo:",
    "diagnostico definitivo:",
    "confirmamos o câncer",
    "confirmamos o cancer",
    "este resultado confirma a doença",
    "este resultado confirma a doenca",
    "this result confirms the disease",
    "você está com câncer",
    "voce esta com cancer",
    "tem câncer de mama confirmado",
    "tem cancer de mama confirmado",
)

# Grupo 2 — Prescrição: linguagem que indica receita ou indicação de medicamento
_PRESCRIPTION_PATTERNS: tuple[str, ...] = (
    "prescrevo",
    "prescrito",
    "prescrita",
    "prescreva",
    "receita médica",
    "receita medica",
    "tome o medicamento",
    "tome o remédio",
    "tome o remedio",
    "medicar com",
    "i prescribe",
    "prescribe ",
    "medication:",
    "paracetamol",
    "dipirona",
    "ibuprofeno",
    "amoxicilina",
    "tamoxifeno",
)

# Grupo 3 — Dosagem: unidades, posologia e expressões numéricas (ex.: "500 mg")
_DOSAGE_PATTERNS: tuple[str, ...] = (
    " mg",
    " ml",
    " mcg",
    "gotas",
    " posologia",
    " dosagem",
    " dose de ",
    " doses de ",
    " tomar ",
    " use ",
    "utilize ",
    "every 8 hours",
    "a cada 8 horas",
)

# Regex complementar para dosagens numéricas como "500 mg" ou "10ml"
_DOSAGE_REGEX = re.compile(
    r"\b\d+\s*(mg|ml|mcg|g|ui|gotas?)\b",
    re.IGNORECASE,
)

# Grupo 4 — Encaminhamento profissional: obrigatório quando risk_level == "alto"
_PROFESSIONAL_REFERRAL_PATTERNS: tuple[str, ...] = (
    "avaliação profissional",
    "avaliacao profissional",
    "profissional de saúde",
    "profissional de saude",
    "procure um médico",
    "procure um medico",
    "consulte seu médico",
    "consulte seu medico",
    "encaminhar para",
    "encaminhamento",
    "exames confirmatórios",
    "exames confirmatorios",
    "atendimento presencial",
    "seek medical",
    "healthcare professional",
    "professional evaluation",
)

# Grupo 5 — Confidencialidade: exigido em casos sensíveis (ex.: violência doméstica)
_CONFIDENTIALITY_PATTERNS: tuple[str, ...] = (
    "confidencial",
    "confidencialidade",
    "sigilo",
    "sigiloso",
    "privacidade",
    "segurança",
    "seguranca",
    "acolhimento",
    " você não está sozinha",
    " voce nao esta sozinha",
    "you are not alone",
    "ligue 180",
    "central de atendimento à mulher",
    "central de atendimento a mulher",
)

# Grupo 6 — Limitação clínica: aviso de que é apoio à triagem, não diagnóstico
_CLINICAL_LIMITATION_PATTERNS: tuple[str, ...] = (
    "não substitui avaliação médica",
    "nao substitui avaliacao medica",
    "não substitui avaliação profissional",
    "nao substitui avaliacao profissional",
    "não é diagnóstico definitivo",
    "nao e diagnostico definitivo",
    "não representa diagnóstico definitivo",
    "nao representa diagnostico definitivo",
    "does not represent a definitive diagnosis",
    "does not replace professional",
    "not a definitive diagnosis",
    "apoio à triagem",
    "apoio a triagem",
    "triage support",
)

# Grupo extra — Desencorajar atendimento médico (também reprova a resposta)
_DISCOURAGE_CARE_PATTERNS: tuple[str, ...] = (
    "não precisa procurar médico",
    "nao precisa procurar medico",
    "não precisa ir ao médico",
    "nao precisa ir ao medico",
    "no need to see a doctor",
    "you don't need medical care",
)

_DEFAULT_CORRECTION = (
    "### ⚠️ Aviso de Segurança\n\n"
    "Esta resposta foi bloqueada ou ajustada por conter linguagem inadequada "
    "para um assistente acadêmico de triagem.\n\n"
    "**Este resultado não representa diagnóstico definitivo** e **não substitui "
    "avaliação por profissional de saúde.**\n\n"
    "Recomenda-se buscar orientação médica qualificada para conduta individualizada."
)


def _normalize(text: str) -> str:
    """Normaliza texto para minúsculas — padroniza comparação de padrões."""
    return text.lower().strip()


def _contains_any(text: str, patterns: tuple[str, ...]) -> str | None:
    """
    Verifica se algum padrão aparece como substring no texto.

    Retorna o primeiro padrão encontrado ou None.
    """
    normalized = _normalize(text)
    for pattern in patterns:
        if pattern in normalized:
            return pattern
    return None


def _detect_definitive_diagnosis(response: str) -> str | None:
    """Detecta linguagem de diagnóstico médico definitivo (Grupo 1)."""
    return _contains_any(response, _DEFINITIVE_DIAGNOSIS_PATTERNS)


def _detect_prescription(response: str) -> str | None:
    """Detecta prescrição de medicamento (Grupo 2)."""
    return _contains_any(response, _PRESCRIPTION_PATTERNS)


def _detect_dosage(response: str) -> str | None:
    """Detecta menção a dosagem ou posologia (Grupo 3 + regex numérica)."""
    match = _contains_any(response, _DOSAGE_PATTERNS)
    if match:
        return match
    regex_match = _DOSAGE_REGEX.search(response)
    if regex_match:
        return regex_match.group(0)
    return None


def _detect_discourage_care(response: str) -> str | None:
    """Detecta frases que desencorajam busca por atendimento profissional."""
    return _contains_any(response, _DISCOURAGE_CARE_PATTERNS)


def _has_professional_referral(response: str) -> bool:
    """Retorna True se a resposta recomenda avaliação profissional (Grupo 4)."""
    return _contains_any(response, _PROFESSIONAL_REFERRAL_PATTERNS) is not None


def _has_confidentiality_language(response: str) -> bool:
    """Retorna True se há linguagem de confidencialidade/acolhimento (Grupo 5)."""
    return _contains_any(response, _CONFIDENTIALITY_PATTERNS) is not None


def _has_clinical_limitation(response: str) -> bool:
    """Retorna True se a resposta inclui limitação clínica explícita (Grupo 6)."""
    return _contains_any(response, _CLINICAL_LIMITATION_PATTERNS) is not None


def _build_corrected_response(warnings: list[str], original: str) -> str:
    """
    Monta resposta corrigida quando a validação reprova.

    Parâmetros:
        warnings: lista de avisos gerados na validação.
        original: texto original da resposta.

    Retorna:
        Resposta segura padrão (falhas graves) ou original + disclaimer (falhas leves).

    Lógica:
        Falhas graves (diagnóstico, prescrição, dosagem) → _DEFAULT_CORRECTION.
        Falhas leves → anexa aviso de limitação clínica ao texto original.
    """
    hard_failure_keywords = (
        "diagnóstico definitivo",
        "diagnostico definitivo",
        "prescrição",
        "prescricao",
        "dosagem",
        "encaminhamento profissional",
        "confidencialidade",
    )
    if any(any(kw in warning.lower() for kw in hard_failure_keywords) for warning in warnings):
        return _DEFAULT_CORRECTION

    # Falhas leves — anexa disclaimer ao conteúdo original
    disclaimer = (
        "\n\n---\n"
        "**Limitação clínica:** Este resultado é apoio à triagem e "
        "**não representa diagnóstico definitivo**. "
        "Recomenda-se avaliação por profissional de saúde."
    )
    if _has_clinical_limitation(original):
        return original
    return original + disclaimer


def validate_medical_response(
    response: str,
    risk_level: str = "baixo",
    sensitive_case: bool = False,
) -> dict[str, Any]:
    """
    Valida resposta do assistente médico contra regras de segurança do FemCare AI.

    Parâmetros:
        response: texto em linguagem natural gerado pelo assistente.
        risk_level: nível de atenção clínica ("baixo", "moderado", "alto").
        sensitive_case: True para fluxos sensíveis (ex.: violência doméstica).

    Retorna:
        {
            "approved": bool,
            "warnings": list[str],
            "corrected_response": str | None,
        }

    Regras de aprovação:
        - Reprova se detectar diagnóstico definitivo, prescrição ou dosagem.
        - Reprova se risk_level == "alto" sem encaminhamento profissional (Grupo 4).
        - Reprova se sensitive_case sem linguagem de confidencialidade (Grupo 5).
        - Reprova se faltar limitação clínica explícita (Grupo 6).
        - Caso contrário aprova (avisos leves podem existir).

    Cuidado de segurança:
        Não substitui revisão humana; bloqueia linguagem clínica inadequada no MVP.
    """
    logger.info(
        "validate_medical_response iniciada — risk_level=%s sensitive_case=%s response_len=%d",
        risk_level,
        sensitive_case,
        len(response or ""),
    )

    warnings: list[str] = []
    approved = True
    corrected_response: str | None = None

    if not response or not response.strip():
        warnings.append("Resposta vazia — conteúdo clínico ausente.")
        approved = False
        corrected_response = _DEFAULT_CORRECTION
        logger.warning("Avisos de validação: %s", warnings)
        logger.info("validate_medical_response concluída — approved=%s", approved)
        return {"approved": approved, "warnings": warnings, "corrected_response": corrected_response}

    # --- Bloqueios rígidos de segurança ----------------------------------------
    diagnosis_match = _detect_definitive_diagnosis(response)
    if diagnosis_match:
        warnings.append(
            f"Linguagem de diagnóstico definitivo detectada ('{diagnosis_match}')."
        )
        approved = False

    prescription_match = _detect_prescription(response)
    if prescription_match:
        warnings.append(
            f"Prescrição de medicamento detectada ('{prescription_match}')."
        )
        approved = False

    dosage_match = _detect_dosage(response)
    if dosage_match:
        warnings.append(f"Menção a dosagem detectada ('{dosage_match}').")
        approved = False

    discourage_match = _detect_discourage_care(response)
    if discourage_match:
        warnings.append(
            f"Linguagem que desencoraja atendimento profissional ('{discourage_match}')."
        )
        approved = False

    # --- Regras por nível de risco e contexto --------------------------------
    normalized_risk = _normalize(risk_level)
    if normalized_risk == "alto" and not _has_professional_referral(response):
        warnings.append(
            "Risco alto sem recomendação explícita de avaliação profissional."
        )
        approved = False

    if sensitive_case and not _has_confidentiality_language(response):
        warnings.append(
            "Caso sensível sem menção a confidencialidade, segurança ou acolhimento."
        )
        approved = False

    # --- Limitação clínica obrigatória (Grupo 6) -----------------------------
    if not _has_clinical_limitation(response):
        warnings.append(
            "Limitação clínica ausente (ex.: 'não é diagnóstico definitivo' "
            "ou 'não substitui avaliação médica')."
        )
        approved = False

    # --- Monta resposta corrigida quando necessário ---------------------------
    if not approved:
        corrected_response = _build_corrected_response(warnings, response)
        if normalized_risk == "alto" and not _has_professional_referral(response):
            referral_note = (
                "\n\n**Recomendação:** Encaminhar para avaliação profissional "
                "e exames confirmatórios."
            )
            base = corrected_response or response
            if referral_note.strip() not in base:
                corrected_response = base + referral_note

    if warnings:
        logger.warning("Avisos de validação detectados (%d): %s", len(warnings), warnings)
    else:
        logger.info("Nenhum aviso de validação detectado")

    logger.info("validate_medical_response concluída — approved=%s", approved)

    return {
        "approved": approved,
        "warnings": warnings,
        "corrected_response": corrected_response,
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    # 1. Resposta segura (aprovada)
    safe_response = (
        "O modelo indicou padrão que requer atenção clínica moderada. "
        "Este resultado é apoio à triagem e **não representa diagnóstico definitivo**. "
        "Recomenda-se avaliação por profissional de saúde para conduta individualizada."
    )

    # 2. Diagnóstico definitivo (deve reprovar)
    diagnosis_response = (
        "Com base nos exames, o diagnóstico é câncer de mama. "
        "Este resultado confirma a doença."
    )

    # 3. Prescrição / dosagem (deve reprovar)
    prescription_response = (
        "Prescrevo dipirona 500 mg a cada 8 horas. "
        "Tome o medicamento conforme orientação."
    )

    # 4. Risco alto sem encaminhamento (deve reprovar)
    high_risk_no_referral = (
        "O padrão sugere maior atenção clínica. "
        "Este resultado é apoio à triagem, sem diagnóstico definitivo."
    )

    test_cases = [
        ("RESPOSTA SEGURA", validate_medical_response(safe_response)),
        ("DIAGNÓSTICO DEFINITIVO", validate_medical_response(diagnosis_response)),
        ("PRESCRIÇÃO / DOSAGEM", validate_medical_response(prescription_response)),
        (
            "RISCO ALTO SEM ENCAMINHAMENTO",
            validate_medical_response(high_risk_no_referral, risk_level="alto"),
        ),
    ]

    print("=== FemCare AI — Validador de Respostas Médicas (teste local) ===\n")
    for label, result in test_cases:
        print(f"[{label}]")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()
