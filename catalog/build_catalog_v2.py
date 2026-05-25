#!/usr/bin/env python3
"""
catalog/build_catalog.py  — v2
─────────────────────────────────────────────────────────────────────────────
Correções em relação à v1:

  [FIX 1] Carrega catalog.json existente como base (não o template)
      Se output/catalog.json já existe, ele é a fonte de verdade.
      O template só é usado quando o catálogo ainda não existe.
      Isso garante que campos manuais já preenchidos nunca são perdidos.

  [FIX 2] Processa apenas fontes novas ou ausentes
      Para cada fonte em build_config.json, verifica se já existe no
      catálogo atual. Se existir, pula a extração e preserva tudo.
      Extrai apenas fontes que ainda não estão no catálogo.

  [FIX 3] Atualiza apenas campos auto-extraídos em fontes existentes
      row_count, column_count, null_rates, sample_rows, last_analyzed
      são sempre atualizados (dados frescos do banco).
      Campos manuais (description, business_role, when_to_use,
      embedding_hint, etc.) são sempre preservados.

Uso:
  python build_catalog.py --config build_config.json
  python build_catalog.py --config build_config.json --force   # re-extrai tudo
"""

import argparse
import json
import os
import sys
import getpass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extractors.sql_extractor import MySQLCatalogExtractor, merge_into_catalog as sql_merge
from extractors.spreadsheet_extractor import (SpreadsheetCatalogExtractor,
                                               merge_into_catalog as sheet_merge)
from extractors.api_extractor import APIEndpointExtractor, merge_into_catalog as api_merge


# ─── Campos manuais que NUNCA devem ser sobrescritos ─────────────────────────
MANUAL_FIELDS = (
    "description", "business_role", "when_to_use", "when_NOT_to_use",
    "embedding_hint", "reasoning",
)

HINT_PLACEHOLDER = (
    "[MANUAL] Palavras-chave e frases curtas que descrevem esta fonte "
    "(20–40 palavras). "
    "Ex: 'distribuição entregas unidades depósito status pendente "
    "quantidade enviada por unidade de destino'."
)


def _load_base_catalog(output_path: str, template_path: str) -> dict:
    """
    Carrega o catálogo existente como base.
    Se não existir, usa o template. Se o template também não existir, cria estrutura mínima.
    Nunca sobrescreve o catálogo existente com o template.
    """
    # Prioridade 1: catálogo já existente
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            catalog = json.load(f)
        print(f"[✓] Catálogo existente carregado: {output_path}")
        print(f"    Fontes já catalogadas:")
        sources = catalog.get("sources", {})
        sql_tables = list(sources.get("sql", {}).get("tables", {}).keys())
        api_eps    = list(sources.get("api", {}).get("endpoints", {}).keys())
        sheets     = list(sources.get("spreadsheets", {}).get("files", {}).keys())
        if sql_tables: print(f"      SQL      : {', '.join(sql_tables)}")
        if api_eps:    print(f"      API      : {', '.join(api_eps)}")
        if sheets:     print(f"      Planilhas: {', '.join(sheets)}")
        return catalog

    # Prioridade 2: template (primeira vez)
    if os.path.exists(template_path):
        with open(template_path, encoding="utf-8") as f:
            catalog = json.load(f)
        catalog["metadata"]["created_at"] = datetime.utcnow().isoformat() + "Z"
        print(f"[!] Catálogo não encontrado. Iniciando do template: {template_path}")
        return catalog

    # Prioridade 3: estrutura mínima
    print(f"[!] Nem catálogo nem template encontrados. Criando estrutura mínima.")
    return {
        "metadata": {
            "catalog_version": "1.0.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": "",
            "project": "Framework Agentic AI — Apoio à Decisão Logística",
        },
        "sources": {
            "sql":          {"tables": {}},
            "api":          {"base_url": "", "endpoints": {}},
            "spreadsheets": {"files": {}},
        },
        "context_hints": {
            "rules": [],
            "fallback_rule": {"action": "include_all_compressed"},
        },
    }


def _source_needs_extraction(catalog: dict, source_type: str,
                              key: str, force: bool) -> bool:
    """
    Verifica se uma fonte precisa ser (re)extraída.
    Retorna True se:
      - force=True (re-extração forçada via --force)
      - a fonte não existe no catálogo atual
    Retorna False se a fonte já existe e force=False.
    """
    if force:
        return True
    sources = catalog.get("sources", {})
    if source_type == "sql":
        return key not in sources.get("sql", {}).get("tables", {})
    if source_type == "api":
        return key not in sources.get("api", {}).get("endpoints", {})
    if source_type == "spreadsheet":
        return key not in sources.get("spreadsheets", {}).get("files", {})
    return True


def _inject_embedding_hints(catalog: dict) -> dict:
    """
    Injeta embedding_hint placeholder apenas em fontes que ainda não têm o campo.
    Nunca sobrescreve hints já preenchidos.
    """
    sources = catalog.get("sources", {})

    for meta in sources.get("sql", {}).get("tables", {}).values():
        if "embedding_hint" not in meta:
            meta["embedding_hint"] = HINT_PLACEHOLDER

    for meta in sources.get("api", {}).get("endpoints", {}).values():
        if "embedding_hint" not in meta:
            meta["embedding_hint"] = HINT_PLACEHOLDER

    for meta in sources.get("spreadsheets", {}).get("files", {}).values():
        if "embedding_hint" not in meta:
            meta["embedding_hint"] = HINT_PLACEHOLDER

    return catalog


def _save(catalog: dict, output_path: str):
    """Salva o catálogo no disco."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    catalog["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# ORQUESTRADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def build_catalog(config: dict, force: bool = False) -> str:
    output_path   = config.get("output", "./output/catalog.json")
    template_path = config.get("template", "./catalog_schema_template.json")

    # Carrega base: catálogo existente > template > estrutura mínima
    catalog = _load_base_catalog(output_path, template_path)
    _save(catalog, output_path)

    # ── 1. SQL ────────────────────────────────────────────────────────────────
    sql_cfg = config.get("sql")
    if sql_cfg:
        tables_config = sql_cfg.get("tables") or []

        # Separa: novas (precisam extração) vs existentes (pula)
        new_tables      = [t for t in tables_config
                           if _source_needs_extraction(catalog, "sql", t, force)]
        existing_tables = [t for t in tables_config if t not in new_tables]

        if existing_tables and not force:
            print(f"\n[SQL] Já catalogadas (preservadas): {', '.join(existing_tables)}")

        if new_tables or force:
            print(f"\n{'─'*50}")
            label = "Re-extraindo" if force else "Extraindo novas"
            targets = tables_config if force else new_tables
            print(f"[SQL] {label}: {', '.join(targets)}")
            print(f"{'─'*50}")

            password = sql_cfg.get("password") or getpass.getpass(
                f"Senha MySQL ({sql_cfg['user']}@{sql_cfg['host']}): "
            )
            url = (f"mysql+pymysql://{sql_cfg['user']}:{password}"
                   f"@{sql_cfg['host']}:{sql_cfg.get('port', 3306)}"
                   f"/{sql_cfg['database']}?charset=utf8mb4")

            extractor = MySQLCatalogExtractor(url, sample_rows=sql_cfg.get("sample_rows", 3))
            extracted = extractor.extract(targets)
            catalog   = sql_merge(output_path, extracted)
            _save(catalog, output_path)
            print(f"[✓] SQL: {len(extracted)} fonte(s) processada(s)")
        else:
            print(f"\n[SQL] Nenhuma fonte nova. Pulando extração.")

    # ── 2. API ────────────────────────────────────────────────────────────────
    api_cfg = config.get("api")
    if api_cfg:
        endpoints_config = api_cfg.get("endpoints", [])

        new_eps      = [ep for ep in endpoints_config
                        if _source_needs_extraction(catalog, "api", ep["key"], force)]
        existing_eps = [ep for ep in endpoints_config if ep not in new_eps]

        if existing_eps and not force:
            keys = [ep["key"] for ep in existing_eps]
            print(f"\n[API] Já catalogados (preservados): {', '.join(keys)}")

        if new_eps or force:
            print(f"\n{'─'*50}")
            targets = endpoints_config if force else new_eps
            keys = [ep["key"] for ep in targets]
            label = "Re-extraindo" if force else "Extraindo novos"
            print(f"[API] {label}: {', '.join(keys)}")
            print(f"{'─'*50}")

            catalog["sources"]["api"]["base_url"] = api_cfg["base_url"]
            api_extractor = APIEndpointExtractor(
                api_cfg["base_url"], auth_token=api_cfg.get("token")
            )
            for ep in targets:
                entry   = api_extractor.build_endpoint_entry(ep["path"], ep["key"])
                catalog = api_merge(output_path, ep["key"], entry)
                _save(catalog, output_path)
            print(f"[✓] API: {len(targets)} endpoint(s) processado(s)")
        else:
            print(f"\n[API] Nenhum endpoint novo. Pulando extração.")

    # ── 3. Planilhas ──────────────────────────────────────────────────────────
    sheets_cfg = config.get("spreadsheets", [])
    if sheets_cfg:
        new_sheets      = [s for s in sheets_cfg
                           if _source_needs_extraction(catalog, "spreadsheet", s["key"], force)]
        existing_sheets = [s for s in sheets_cfg if s not in new_sheets]

        if existing_sheets and not force:
            keys = [s["key"] for s in existing_sheets]
            print(f"\n[Planilhas] Já catalogadas (preservadas): {', '.join(keys)}")

        if new_sheets or force:
            print(f"\n{'─'*50}")
            targets = sheets_cfg if force else new_sheets
            keys = [s["key"] for s in targets]
            label = "Re-extraindo" if force else "Extraindo novas"
            print(f"[Planilhas] {label}: {', '.join(keys)}")
            print(f"{'─'*50}")

            for sheet_cfg in targets:
                sheet_extractor = SpreadsheetCatalogExtractor(
                    sheet_cfg["file"], sample_rows=sheet_cfg.get("sample_rows", 3)
                )
                extracted = sheet_extractor.extract()
                catalog   = sheet_merge(output_path, sheet_cfg["key"], extracted)
                _save(catalog, output_path)
            print(f"[✓] Planilhas: {len(targets)} arquivo(s) processado(s)")
        else:
            print(f"\n[Planilhas] Nenhuma planilha nova. Pulando extração.")

    # Injeta embedding_hint placeholder apenas onde não existe
    catalog = _inject_embedding_hints(catalog)
    _save(catalog, output_path)

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data Catalog Builder")
    parser.add_argument("--config", required=True, help="Caminho do build_config.json")
    parser.add_argument("--force",  action="store_true",
                        help="Re-extrai todas as fontes, mesmo as já catalogadas "
                             "(preserva campos manuais)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    print(f"\n{'='*55}")
    print(f"  DATA CATALOG BUILDER v2 — Framework Agentic AI")
    if args.force:
        print(f"  MODO: re-extração forçada (--force)")
    else:
        print(f"  MODO: incremental (preserva fontes existentes)")
    print(f"{'='*55}")

    output = build_catalog(config, force=args.force)

    print(f"\n{'='*55}")
    print(f"  ✅ Catálogo atualizado com sucesso!")
    print(f"  📁 Arquivo: {output}")
    print(f"\n  PRÓXIMOS PASSOS para novas fontes:")
    print(f"  1. Abra {output} e preencha os campos [MANUAL]")
    print(f"     nas fontes recém-adicionadas")
    print(f"  2. Preencha o embedding_hint de cada fonte nova")
    print(f"  3. Atualize context_hints se necessário")
    print(f"\n  DICA: use --force para re-extrair metadados técnicos")
    print(f"  de fontes existentes sem perder campos manuais.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
