"""
sql_tool — executor determinístico para consultas SQL.

Recebe uma query SQL (gerada pelo Query Generator) e a executa
via SQLAlchemy. Sem LLM aqui: entrada → banco → saída.

Segurança:
- Aceita apenas SELECT (validação simples antes de executar)
- Credentials nunca passam pelo LLM — ficam em variáveis de ambiente
"""
from __future__ import annotations

import os
import re
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


# ---------------------------------------------------------------------------
# Pool de conexões (uma engine por connection_key)
# ---------------------------------------------------------------------------

_engines: dict[str, Engine] = {}


def _get_engine(connection_key: str = "default") -> Engine:
    """Retorna (e cacheia) a engine SQLAlchemy para a connection_key."""
    if connection_key not in _engines:
        url = os.getenv(
            f"DB_URL_{connection_key.upper()}",
            os.getenv("DB_URL", os.getenv("MYSQL_URI", "")),
        )
        if not url:
            raise EnvironmentError(
                f"Variável de ambiente DB_URL_{connection_key.upper()}, "
                f"DB_URL ou MYSQL_URI não definida."
            )
        _engines[connection_key] = create_engine(url, pool_pre_ping=True)
    return _engines[connection_key]


# ---------------------------------------------------------------------------
# Validação de segurança
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def _validate_select_only(sql: str) -> None:
    """Levanta ValueError se a query contiver operações de escrita."""
    if _FORBIDDEN_PATTERNS.search(sql):
        raise ValueError(
            f"Query rejeitada: contém operação não permitida. "
            f"Apenas SELECT é aceito. Query: {sql[:200]}"
        )
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        raise ValueError(
            f"Query rejeitada: deve começar com SELECT ou WITH. "
            f"Query: {sql[:200]}"
        )


# ---------------------------------------------------------------------------
# Interface pública
# ---------------------------------------------------------------------------

def execute_sql(
    sql: str,
    connection_key: str = "default",
    max_rows: int = 1000,
) -> dict[str, Any]:
    """
    Executa uma query SQL e retorna os resultados.

    Args:
        sql: query SELECT a executar
        connection_key: chave da conexão no pool (default: "default")
        max_rows: limite de linhas retornadas (proteção contra dumps acidentais)

    Returns:
        {
            "status": "success" | "empty" | "error",
            "rows": [{"col": val, ...}, ...],
            "columns": ["col1", "col2", ...],
            "row_count": int,
            "error_message": str  # vazio se success/empty
        }
    """
    try:
        _validate_select_only(sql)
    except ValueError as e:
        return {
            "status": "error",
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error_message": str(e),
        }

    try:
        engine = _get_engine(connection_key)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchmany(max_rows)]

        if not rows:
            return {
                "status": "empty",
                "rows": [],
                "columns": columns,
                "row_count": 0,
                "error_message": "",
            }

        return {
            "status": "success",
            "rows": rows,
            "columns": columns,
            "row_count": len(rows),
            "error_message": "",
        }

    except SQLAlchemyError as e:
        return {
            "status": "error",
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error_message": str(e),
        }
    except Exception as e:
        return {
            "status": "error",
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error_message": f"Erro inesperado: {e}",
        }
