"""
Catálogo semântico de fontes — lê o catalog.json do SCB existente.

Não duplica os dados: este módulo apenas carrega e indexa
`catalog/output/catalog_v3.json` (gerado por catalog/build_catalog_v2.py
e usado em catalog/run_llm_benchmark.py para o artigo do SCB), expondo
a mesma interface (CATALOG, get_source, get_schema_for_prompt) que o
restante do pipeline (core/components.py, tools/sql_tool.py) já consome.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "catalog" / "output" / "catalog_v3.json"
)


@dataclass
class SourceSchema:
    """Metadados de uma fonte necessários para SCB, Planner e Query Generator."""
    source_id: str
    source_type: str          # "sql" | "api" | "spreadsheets"
    description: str
    embedding_hint: str
    # Para SQL: campos disponíveis com tipo e descrição curta
    fields: list[dict] = field(default_factory=list)
    # Exemplos de valores reais (mascarados) para reduzir alucinações
    sample_values: dict = field(default_factory=dict)
    # Para SQL: nome da tabela e conexão
    table_name: str = ""
    connection_key: str = "default"


def _columns_to_fields(columns: dict) -> list[dict]:
    return [
        {
            "name": col_name,
            "type": col.get("type", ""),
            "desc": col.get("description", ""),
        }
        for col_name, col in columns.items()
    ]


def _sample_rows_to_values(sample_rows: list[dict], columns: dict) -> dict:
    """Agrupa valores de exemplo (de sample_rows) por coluna."""
    samples: dict[str, list] = {col: [] for col in columns}
    for row in sample_rows:
        for col, val in row.items():
            if col in samples and val not in samples[col]:
                samples[col].append(val)
    return {col: vals for col, vals in samples.items() if vals}


def _load_catalog() -> dict[str, SourceSchema]:
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    sources = raw.get("sources", {})
    catalog: dict[str, SourceSchema] = {}

    for table_name, meta in sources.get("sql", {}).get("tables", {}).items():
        source_id = f"sql.{table_name}"
        columns = meta.get("columns", {})
        catalog[source_id] = SourceSchema(
            source_id=source_id,
            source_type="sql",
            description=meta.get("description", ""),
            embedding_hint=meta.get("embedding_hint", meta.get("description", "")),
            fields=_columns_to_fields(columns),
            sample_values=_sample_rows_to_values(meta.get("sample_rows", []), columns),
            table_name=table_name,
            connection_key="default",
        )

    for endpoint_name, meta in sources.get("api", {}).get("endpoints", {}).items():
        source_id = f"api.{endpoint_name}"
        catalog[source_id] = SourceSchema(
            source_id=source_id,
            source_type="api",
            description=meta.get("description", ""),
            embedding_hint=meta.get("embedding_hint", meta.get("description", "")),
        )

    for file_name, meta in sources.get("spreadsheets", {}).get("files", {}).items():
        source_id = f"spreadsheets.{file_name}"
        catalog[source_id] = SourceSchema(
            source_id=source_id,
            source_type="spreadsheets",
            description=meta.get("description", ""),
            embedding_hint=meta.get("embedding_hint", meta.get("description", "")),
        )

    return catalog


CATALOG: dict[str, SourceSchema] = _load_catalog()


def get_source(source_id: str) -> SourceSchema:
    """Retorna o schema de uma fonte pelo ID."""
    if source_id not in CATALOG:
        raise KeyError(f"Fonte '{source_id}' não encontrada no catálogo.")
    return CATALOG[source_id]


def get_schema_for_prompt(source_id: str) -> str:
    """
    Serializa o schema de uma fonte em formato compacto para injeção
    no prompt do Query Generator.
    """
    src = get_source(source_id)
    fields_text = "\n".join(
        f"  - {f['name']} ({f['type']}): {f['desc']}"
        for f in src.fields
    )
    samples_text = "\n".join(
        f"  - {col}: {vals}"
        for col, vals in src.sample_values.items()
    )
    return (
        f"FONTE: {src.source_id}\n"
        f"TABELA: {src.table_name}\n"
        f"DESCRIÇÃO: {src.description}\n\n"
        f"CAMPOS DISPONÍVEIS:\n{fields_text}\n\n"
        f"EXEMPLOS DE VALORES (dados mascarados):\n{samples_text}"
    )
