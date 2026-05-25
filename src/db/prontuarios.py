"""
Camada de acesso ao banco de dados SQLite de prontuários e auditoria.
Satisfaz requisito: "consultas em base de dados estruturadas (prontuários e registros)" — Tech Challenge Fase 3.
"""

import csv
import sqlite3
import os
from contextlib import contextmanager
from src.logger import get_logger

log = get_logger("consultas_medica.db")

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "prontuarios.db",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nome            TEXT    NOT NULL UNIQUE,
    nascimento      TEXT    NOT NULL,
    cpf_anonimizado TEXT,
    ultimo_papanicolau INTEGER DEFAULT 0,
    ultima_mamografia  INTEGER DEFAULT 0,
    historico_familiar TEXT,
    observacoes        TEXT
);

CREATE TABLE IF NOT EXISTS atendimentos (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    data_hora TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    relato    TEXT,
    risco     TEXT,
    resposta  TEXT
);
"""


_CSV_PATH = os.path.join(os.path.dirname(_DB_PATH), "pacientes_sinteticos.csv")


def _importar_csv(conn):
    """Importa pacientes_sinteticos.csv para a tabela pacientes (apenas se vazia)."""
    count = conn.execute("SELECT COUNT(*) FROM pacientes").fetchone()[0]
    if count > 0 or not os.path.exists(_CSV_PATH):
        return
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            conn.execute(
                """INSERT OR IGNORE INTO pacientes
                   (nome, nascimento, cpf_anonimizado, ultimo_papanicolau,
                    ultima_mamografia, historico_familiar, observacoes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["nome"], row["nascimento"], row["cpf_anonimizado"],
                    int(row["ultimo_papanicolau"] or 0),
                    int(row["ultima_mamografia"] or 0),
                    row["historico_familiar"], row["observacoes"],
                ),
            )
    conn.commit()
    log.info("DB | pacientes importados de %s", _CSV_PATH)


@contextmanager
def get_db():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        _importar_csv(conn)
        yield conn
    finally:
        conn.close()


def consultar_paciente(nome: str) -> dict | None:
    """Retorna prontuário completo pelo nome ou None se não encontrado."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM pacientes WHERE nome = ?", (nome,)
        ).fetchone()
    if row:
        log.info("DB | paciente encontrado: %s", nome)
        return dict(row)
    log.info("DB | paciente não encontrado: %s", nome)
    return None


def obter_pacientes() -> list[str]:
    """Retorna lista de nomes de todos os pacientes cadastrados."""
    with get_db() as conn:
        rows = conn.execute("SELECT nome FROM pacientes ORDER BY nome").fetchall()
    return [r["nome"] for r in rows]


def salvar_atendimento(_nome: str, relato: str, risco: str, resposta: str) -> None:
    """Grava registro de atendimento para auditoria."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO atendimentos (relato, risco, resposta) VALUES (?, ?, ?)",
            (relato, risco, resposta),
        )
        conn.commit()
    log.info("DB | atendimento registrado (risco=%s)", risco)


def buscar_historico(limit: int = 20) -> list[dict]:
    """Retorna os atendimentos mais recentes para exibição no dashboard."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, data_hora, relato, risco, resposta FROM atendimentos ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
