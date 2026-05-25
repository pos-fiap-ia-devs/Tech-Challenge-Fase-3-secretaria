"""
Logger de auditoria estruturada — FemCare AI MVP.

Grava eventos de auditoria em JSONL (outputs/audit_logs.jsonl) para rastreabilidade
acadêmica do fluxo Breast Cancer e demais etapas que o utilizem.

Por que JSONL **complementa** o SQLite existente (src/db/prontuarios.py):
    - SQLite: prontuário relacional legado, integrado à etapa_etica do grafo.
    - JSONL: trilha append-only, legível por linha, ideal para exportar/analisar
      eventos de fluxo (flow, risk_level, safety_status) sem alterar o schema SQL.
    - Os dois coexistem: JSONL não substitui o banco; cada um serve a um propósito.

Conteúdo sensível é resumido ou mascarado antes da persistência.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caminhos e campos padrão do evento de auditoria
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_DIR = _PROJECT_ROOT / "outputs"
_AUDIT_FILE = _AUDIT_DIR / "audit_logs.jsonl"

_STANDARD_FIELDS: tuple[str, ...] = (
    "timestamp",
    "flow",
    "risk_level",
    "sources",
    "safety_status",
    "sensitive_case",
    "summary",
)

_DEFAULTS: dict[str, Any] = {
    "flow": "unknown",
    "risk_level": "unknown",
    "sources": [],
    "safety_status": "unknown",
    "sensitive_case": False,
    "summary": "Audit event recorded.",
}

_MAX_NORMAL_SUMMARY_LEN = 500
_MAX_SENSITIVE_SUMMARY_LEN = 80

# Padrões que podem indicar PII — substituídos por [REDACTED] antes do log
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),  # CPF
    re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),  # e-mail
    re.compile(r"\b\d{10,11}\b"),  # telefone
)


def _ensure_outputs_directory() -> Path:
    """
    Garante que o diretório outputs/ existe.

    Retorna o caminho do diretório (criado se necessário).
    """
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return _AUDIT_DIR


def get_audit_log_path() -> Path:
    """Retorna o caminho absoluto do arquivo JSONL de auditoria."""
    return _AUDIT_FILE


def _truncate_text(text: str, max_len: int) -> str:
    """Trunca texto em max_len caracteres, com reticências se necessário."""
    cleaned = text.strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _redact_pii(text: str) -> str:
    """
    Substitui padrões comuns de PII (CPF, e-mail, telefone) por [REDACTED].

    Cuidado de segurança: reduz vazamento acidental em logs acadêmicos.
    """
    redacted = text
    for pattern in _PII_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _mask_sensitive_summary(summary: str, sensitive_case: bool) -> str:
    """
    Mascara ou encurta resumos em fluxos sensíveis (ex.: violência doméstica).

    Parâmetros:
        summary: texto bruto do resumo do evento.
        sensitive_case: True quando o fluxo exige sigilo extra.

    Retorna:
        Resumo truncado e com PII redigida; em casos sensíveis, apenas contexto mínimo.

    Cuidado de segurança:
        Nunca persiste narrativa completa de casos sensíveis no JSONL.
    """
    if not summary or not summary.strip():
        if sensitive_case:
            return "Sensitive case recorded — detailed content withheld for confidentiality."
        return _DEFAULTS["summary"]

    cleaned = _redact_pii(summary.strip())

    if not sensitive_case:
        return _truncate_text(cleaned, _MAX_NORMAL_SUMMARY_LEN)

    # Caso sensível — mantém contexto mínimo, nunca a narrativa completa
    brief = _truncate_text(cleaned, _MAX_SENSITIVE_SUMMARY_LEN)
    return f"{brief} [conteúdo sensível omitido]"


def _coerce_sources(value: Any) -> list[str]:
    """Normaliza sources para lista de strings curtas."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _coerce_bool(value: Any) -> bool:
    """Converte entradas diversas (str, int, bool) para booleano."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "sim"}
    return bool(value)


def _sanitize_event(event: dict) -> dict:
    """
    Normaliza e sanitiza um evento de auditoria antes da gravação.

    Parâmetros:
        event: dicionário bruto recebido por write_audit_log.

    Retorna:
        Dicionário apenas com campos padronizados (_STANDARD_FIELDS).

    Efeitos:
        - Preenche campos ausentes com defaults seguros.
        - Mascara summary quando sensitive_case é True.
        - Descarta chaves extras para não persistir payloads inesperados.
    """
    if not isinstance(event, dict):
        logger.warning("Evento de auditoria não é dict — usando payload vazio com defaults")
        event = {}

    missing_fields: list[str] = []

    # --- sensitive_case primeiro (define mascaramento do summary) ------------
    if "sensitive_case" not in event:
        missing_fields.append("sensitive_case")
    sensitive_case = _coerce_bool(event.get("sensitive_case", _DEFAULTS["sensitive_case"]))

    # --- Timestamp UTC/local em ISO ------------------------------------------
    timestamp = event.get("timestamp")
    if not timestamp:
        missing_fields.append("timestamp")
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    # --- Campos escalares com defaults ---------------------------------------
    flow = event.get("flow") or _DEFAULTS["flow"]
    if "flow" not in event:
        missing_fields.append("flow")

    risk_level = event.get("risk_level") or _DEFAULTS["risk_level"]
    if "risk_level" not in event:
        missing_fields.append("risk_level")

    safety_status = event.get("safety_status") or _DEFAULTS["safety_status"]
    if "safety_status" not in event:
        missing_fields.append("safety_status")

    # --- Lista de fontes (sources) -------------------------------------------
    sources = _coerce_sources(event.get("sources"))
    if "sources" not in event:
        missing_fields.append("sources")

    # --- Summary (mascarado se sensitive_case) --------------------------------
    raw_summary = event.get("summary", "")
    if "summary" not in event:
        missing_fields.append("summary")
    summary = _mask_sensitive_summary(str(raw_summary) if raw_summary else "", sensitive_case)

    if missing_fields:
        logger.warning("Evento de auditoria com campos ausentes — defaults aplicados: %s", missing_fields)

    return {
        "timestamp": str(timestamp),
        "flow": str(flow),
        "risk_level": str(risk_level),
        "sources": sources,
        "safety_status": str(safety_status),
        "sensitive_case": sensitive_case,
        "summary": summary,
    }


def write_audit_log(event: dict) -> None:
    """
    Grava um evento sanitizado como uma linha JSON em outputs/audit_logs.jsonl.

    Parâmetros:
        event: payload bruto; chaves padrão são normalizadas, extras ignoradas.

    Efeitos colaterais:
        Append no arquivo JSONL (modo "a" — logs anteriores nunca são sobrescritos).
        Cria outputs/ se não existir.

    Cuidado de segurança:
        Summary passa por mascaramento de PII e truncamento em casos sensíveis.
    """
    logger.info("write_audit_log iniciada — flow=%s", event.get("flow", "unknown"))

    try:
        _ensure_outputs_directory()
        sanitized = _sanitize_event(event)

        line = json.dumps(sanitized, ensure_ascii=False)

        with _AUDIT_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        logger.info(
            "Evento de auditoria salvo — file=%s flow=%s risk_level=%s safety_status=%s sensitive_case=%s",
            _AUDIT_FILE,
            sanitized["flow"],
            sanitized["risk_level"],
            sanitized["safety_status"],
            sanitized["sensitive_case"],
        )
    except Exception:
        logger.exception("Falha ao gravar audit log em %s", _AUDIT_FILE)
        raise


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print("=== FemCare AI — Audit Logger (teste local) ===\n")

    # 1. Evento normal do fluxo breast_cancer
    write_audit_log(
        {
            "flow": "breast_cancer",
            "risk_level": "alto",
            "sources": ["phase2_breast_cancer_model_card.md", "breast_cancer_screening.md"],
            "safety_status": "approved",
            "sensitive_case": False,
            "summary": "Caso sintético P002 analisado com recomendação de avaliação especializada.",
        }
    )
    print("[1] Evento breast_cancer normal gravado.")

    # 2. Evento com campos ausentes (defaults aplicados)
    write_audit_log({})
    print("[2] Evento mínimo (campos ausentes) gravado.")

    # 3. Caso sensível — summary deve ser mascarado
    write_audit_log(
        {
            "flow": "violencia",
            "risk_level": "alto",
            "sources": ["domestic_violence_safety.md"],
            "safety_status": "approved",
            "sensitive_case": True,
            "summary": (
                "Usuária relatou medo do parceiro, ameaças frequentes e lesões recorrentes. "
                "Contato: maria.silva@email.com CPF 123.456.789-00. Pediu sigilo absoluto."
            ),
        }
    )
    print("[3] Evento de caso sensível gravado (summary mascarado).")

    audit_path = get_audit_log_path()
    print(f"\nArquivo de audit log: {audit_path}")
    print(f"Arquivo existe: {audit_path.is_file()}")

    if audit_path.is_file():
        print("\nÚltimas 3 linhas gravadas:")
        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines[-3:]:
            print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))
            print()
