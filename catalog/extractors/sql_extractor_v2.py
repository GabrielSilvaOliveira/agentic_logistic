"""
catalog/extractors/sql_extractor.py
─────────────────────────────────────────────────────────────────────────────
Extrator automático de metadados MySQL → catálogo JSON.

O que este extrator faz automaticamente:
  ✅ Lista tabelas e colunas com tipos nativos MySQL
  ✅ Conta linhas por tabela
  ✅ Detecta chaves primárias e estrangeiras
  ✅ Calcula taxa de nulos por coluna
  ✅ Extrai N linhas de amostra (com dados já mascarados)
  ✅ Detecta colunas com valores únicos (candidatas a chave de busca)

O que você preenche manualmente depois:
  ✏️  description, business_role, when_to_use de cada tabela
  ✏️  context_hints no catálogo principal

Uso:
  python sql_extractor.py                        # usa config padrão
  python sql_extractor.py --host localhost \
         --port 3306 --db meu_banco \
         --user root --output ../output/catalog_sql.json
"""

import argparse
import json
import getpass
from datetime import datetime, date
from decimal import Decimal
from typing import Any

try:
    import sqlalchemy as sa
    import pandas as pd
except ImportError:
    raise SystemExit("Execute: pip install sqlalchemy pymysql pandas")


# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZADOR — converte tipos MySQL para JSON
# ─────────────────────────────────────────────────────────────────────────────

def json_safe(obj: Any) -> Any:
    """Converte tipos não serializáveis para JSON."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRATOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class MySQLCatalogExtractor:

    def __init__(self, connection_url: str, sample_rows: int = 3):
        """
        connection_url: SQLAlchemy URL
            Ex: mysql+pymysql://user:senha@localhost:3306/banco
        sample_rows: quantas linhas de amostra extrair por tabela
        """
        self.engine = sa.create_engine(connection_url, pool_pre_ping=True)
        self.sample_rows = sample_rows
        self.inspector = sa.inspect(self.engine)

    def extract(self, tables: list[str] | None = None) -> dict:
        """
        Extrai metadados de todas as tabelas (ou das listadas em `tables`).
        Retorna dicionário no formato do catalog_schema_template.json.
        """
        target_tables = tables or self.inspector.get_table_names()
        print(f"[SQL Extractor] Encontradas {len(target_tables)} tabela(s): "
              f"{', '.join(target_tables)}")

        extracted = {}
        for table in target_tables:
            print(f"  → Extraindo: {table} ...", end=" ", flush=True)
            try:
                extracted[table] = self._extract_table(table)
                print(f"✅ ({extracted[table]['row_count']:,} linhas)")
            except Exception as e:
                print(f"❌ Erro: {e}")
                extracted[table] = {"error": str(e)}

        return extracted

    def _extract_table(self, table: str) -> dict:
        """Extrai metadados completos de uma tabela."""

        # ── Colunas e tipos ───────────────────────────────────────────────────
        columns_meta = self.inspector.get_columns(table)
        pk_columns   = set(self.inspector.get_pk_constraint(table).get("constrained_columns", []))
        fk_list      = self.inspector.get_foreign_keys(table)

        # ── Contagem de linhas ────────────────────────────────────────────────
        with self.engine.connect() as conn:
            row_count = conn.execute(
                sa.text(f"SELECT COUNT(*) FROM `{table}`")
            ).scalar()

        # ── Amostra de dados ──────────────────────────────────────────────────
        sample_df = pd.read_sql(
            f"SELECT * FROM `{table}` LIMIT {self.sample_rows}",
            self.engine
        )

        # ── Taxa de nulos por coluna ──────────────────────────────────────────
        null_rates = self._compute_null_rates(table, [c["name"] for c in columns_meta])

        # ── Cardinalidade de colunas-chave (valores únicos) ───────────────────
        cardinality = self._compute_cardinality(table, [c["name"] for c in columns_meta])

        # ── Monta estrutura de colunas ────────────────────────────────────────
        columns = {}
        for col in columns_meta:
            col_name = col["name"]
            columns[col_name] = {
                "type":        str(col["type"]),
                "nullable":    col.get("nullable", True),
                "is_pk":       col_name in pk_columns,
                "default":     str(col.get("default", "")) if col.get("default") is not None else None,
                "null_rate":   null_rates.get(col_name, 0.0),
                "cardinality": cardinality.get(col_name),
                # Placeholder para anotação manual
                "description": f"[MANUAL] Descreva o significado de '{col_name}' no contexto logístico",
            }

        # ── Chaves estrangeiras ───────────────────────────────────────────────
        foreign_keys = [
            {
                "columns":            fk["constrained_columns"],
                "references_table":   fk["referred_table"],
                "references_columns": fk["referred_columns"],
            }
            for fk in fk_list
        ]

        # ── Amostra serializada ───────────────────────────────────────────────
        sample = json.loads(
            sample_df.to_json(orient="records", default_handler=str, force_ascii=False)
        )

        return {
            "description":  f"[MANUAL] Descreva o papel da tabela '{table}' no negócio logístico",
            "business_role": "[MANUAL] Ex: O que esta tabela representa operacionalmente?",
            "when_to_use":  "[MANUAL] Ex: Quando o agente deve consultar esta tabela?",
            "row_count":    row_count,
            "column_count": len(columns),
            "columns":      columns,
            "foreign_keys": foreign_keys,
            "sample_rows":  sample,
            "data_quality": {
                "null_rates":    null_rates,
                "last_analyzed": datetime.utcnow().isoformat() + "Z",
            },
        }

    def _compute_null_rates(self, table: str, columns: list[str]) -> dict[str, float]:
        """Calcula taxa de nulos para cada coluna em uma única query."""
        # Monta: SUM(col IS NULL)/COUNT(*) para cada coluna
        parts = [
            f"ROUND(SUM(`{col}` IS NULL) / COUNT(*), 4) AS `{col}`"
            for col in columns
        ]
        query = f"SELECT {', '.join(parts)} FROM `{table}`"
        try:
            with self.engine.connect() as conn:
                row = conn.execute(sa.text(query)).fetchone()
            return {col: float(row[i] or 0) for i, col in enumerate(columns)}
        except Exception:
            return {col: 0.0 for col in columns}

    def _compute_cardinality(self, table: str, columns: list[str]) -> dict[str, int | None]:
        """Conta valores únicos por coluna (útil para identificar colunas-chave)."""
        cardinality = {}
        # Só calcula para tabelas com menos de 100k linhas (performance)
        with self.engine.connect() as conn:
            count = conn.execute(sa.text(f"SELECT COUNT(*) FROM `{table}`")).scalar()

        if count > 100_000:
            return {col: None for col in columns}  # None = "não calculado"

        for col in columns:
            try:
                with self.engine.connect() as conn:
                    unique = conn.execute(
                        sa.text(f"SELECT COUNT(DISTINCT `{col}`) FROM `{table}`")
                    ).scalar()
                cardinality[col] = int(unique)
            except Exception:
                cardinality[col] = None
        return cardinality


# ─────────────────────────────────────────────────────────────────────────────
# MERGE COM O TEMPLATE EXISTENTE
# ─────────────────────────────────────────────────────────────────────────────

def merge_into_catalog(catalog_path: str, extracted_tables: dict) -> dict:
    """
    Carrega o catalog_schema_template.json e preenche a seção sql.tables
    com os dados extraídos, preservando anotações manuais existentes.
    """
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    existing_tables = catalog["sources"]["sql"].get("tables", {})

    for table_name, table_data in extracted_tables.items():
        if table_name in existing_tables:
            existing = existing_tables[table_name]
            # Preserva anotações manuais (description, business_role, when_to_use)
            # se já foram preenchidas (não contêm [MANUAL])
            # Preserva TODOS os campos manuais, incluindo embedding_hint
            _MANUAL = ("description", "business_role", "when_to_use",
                       "when_NOT_to_use", "embedding_hint")
            for manual_field in _MANUAL:
                if manual_field in existing and "[MANUAL]" not in str(existing[manual_field]):
                    table_data[manual_field] = existing[manual_field]
            # Preserva descrições manuais de colunas
            if "columns" in existing:
                for col_name, col_existing in existing["columns"].items():
                    if col_name in table_data["columns"]:
                        if ("description" in col_existing and
                                "[MANUAL]" not in str(col_existing["description"])):
                            table_data["columns"][col_name]["description"] = \
                                col_existing["description"]

        existing_tables[table_name] = table_data

    catalog["sources"]["sql"]["tables"] = existing_tables
    catalog["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    return catalog


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrator SQL → Data Catalog")
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     default=3306, type=int)
    parser.add_argument("--db",       required=True, help="Nome do banco de dados")
    parser.add_argument("--user",     required=True, help="Usuário MySQL")
    parser.add_argument("--password", default=None,  help="Senha (ou deixe em branco para prompt)")
    parser.add_argument("--tables",   default=None,  help="Tabelas específicas, separadas por vírgula")
    parser.add_argument("--sample",   default=3, type=int, help="Linhas de amostra por tabela")
    parser.add_argument("--template", default="../catalog_schema_template.json",
                        help="Caminho do template JSON a ser preenchido")
    parser.add_argument("--output",   default="../output/catalog.json",
                        help="Caminho de saída do catálogo preenchido")
    args = parser.parse_args()

    password = args.password or getpass.getpass(f"Senha MySQL para {args.user}@{args.host}: ")

    url = f"mysql+pymysql://{args.user}:{password}@{args.host}:{args.port}/{args.db}?charset=utf8mb4"
    tables = [t.strip() for t in args.tables.split(",")] if args.tables else None

    print(f"\n{'='*55}")
    print(f"  SQL Catalog Extractor — MySQL")
    print(f"  Banco : {args.db} @ {args.host}:{args.port}")
    print(f"  Saída : {args.output}")
    print(f"{'='*55}\n")

    extractor = MySQLCatalogExtractor(url, sample_rows=args.sample)
    extracted  = extractor.extract(tables)

    # Merge com template existente
    try:
        catalog = merge_into_catalog(args.template, extracted)
        print(f"\n[✓] Template '{args.template}' carregado e mesclado.")
    except FileNotFoundError:
        print(f"\n[!] Template não encontrado. Gerando catálogo isolado.")
        catalog = {
            "metadata": {"updated_at": datetime.utcnow().isoformat() + "Z"},
            "sources": {"sql": {"tables": extracted}}
        }

    # Salva resultado
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=json_safe)

    print(f"[✓] Catálogo salvo em: {args.output}")
    print(f"\nPróximo passo: abra o arquivo e preencha os campos [MANUAL]\n")


if __name__ == "__main__":
    main()
