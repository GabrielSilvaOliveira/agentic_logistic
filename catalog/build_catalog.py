#!/usr/bin/env python3
"""
catalog/build_catalog.py
─────────────────────────────────────────────────────────────────────────────
Script orquestrador: roda todos os extratores em sequência e gera
o catálogo final catalog.json pronto para uso pelo agente.

Uso rápido:
  python build_catalog.py --config build_config.json

build_config.json (crie este arquivo com suas configurações):
{
  "sql": {
    "host": "localhost",
    "port": 3306,
    "database": "seu_banco",
    "user": "root",
    "password": "sua_senha",
    "tables": ["materiais", "unidades", "catalogo"],
    "sample_rows": 3
  },
  "api": {
    "base_url": "http://localhost:8000",
    "endpoints": [
      {
        "path": "/api/distribuicao/",
        "key": "distribuicao_list"
      }
    ]
  },
  "spreadsheets": [
    {
      "file": "../data/contratos.xlsx",
      "key": "contratos",
      "sample_rows": 3
    }
  ],
  "output": "./output/catalog.json",
  "template": "./catalog_schema_template.json"
}
"""

import argparse
import json
import os
import sys
import getpass
from datetime import datetime
from pathlib import Path

# Adiciona o diretório pai ao path para imports relativos
sys.path.insert(0, str(Path(__file__).parent))

from extractors.sql_extractor import MySQLCatalogExtractor, merge_into_catalog as sql_merge
from extractors.spreadsheet_extractor import (SpreadsheetCatalogExtractor,
                                               merge_into_catalog as sheet_merge)
from extractors.api_extractor import APIEndpointExtractor, merge_into_catalog as api_merge


def _inject_embedding_hints(catalog: dict) -> dict:
    """
    Injeta o campo 'embedding_hint' como placeholder em todas as fontes
    que ainda não possuem esse campo. Não sobrescreve valores já preenchidos.

    O embedding_hint é um texto curto e focado (20–40 palavras) escrito
    manualmente para guiar a seleção semântica do SemanticContextBuilder.
    Quando preenchido, substitui o documento automático gerado dos metadados.
    """
    _HINT_PLACEHOLDER = (
        "[MANUAL] Palavras-chave e frases curtas que descrevem esta fonte "
        "(20\u201340 palavras). "
        "Ex: 'distribui\u00e7\u00e3o entregas unidades dep\u00f3sito status pendente "
        "quantidade enviada por unidade de destino'."
    )
    sources = catalog.get("sources", {})

    for meta in sources.get("sql", {}).get("tables", {}).values():
        if "embedding_hint" not in meta:
            meta["embedding_hint"] = _HINT_PLACEHOLDER

    for meta in sources.get("api", {}).get("endpoints", {}).values():
        if "embedding_hint" not in meta:
            meta["embedding_hint"] = _HINT_PLACEHOLDER

    for meta in sources.get("spreadsheets", {}).get("files", {}).values():
        if "embedding_hint" not in meta:
            meta["embedding_hint"] = _HINT_PLACEHOLDER

    return catalog


def build_catalog(config: dict) -> str:
    """
    Orquestra todos os extratores e salva o catálogo final.
    Retorna o caminho do arquivo gerado.
    """
    output_path = config.get("output", "./output/catalog.json")
    template    = config.get("template", "./catalog_schema_template.json")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Inicializa catálogo a partir do template (se existir)
    if os.path.exists(template):
        with open(template, encoding="utf-8") as f:
            catalog = json.load(f)
        print(f"[✓] Template carregado: {template}")
    else:
        catalog = {
            "metadata": {
                "catalog_version": "1.0.0",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "project": "Framework Agentic AI — Apoio à Decisão Logística"
            },
            "sources": {"sql": {"tables": {}}, "api": {"endpoints": {}},
                        "spreadsheets": {"files": {}}},
            "context_hints": {"rules": [], "fallback_rule": {"action": "include_all_compressed"}}
        }

    # Salva estado inicial
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    # ── 1. Extração SQL ───────────────────────────────────────────────────────
    sql_cfg = config.get("sql")
    if sql_cfg:
        print(f"\n{'─'*50}")
        print(f"[1/3] Extraindo metadados SQL...")
        print(f"{'─'*50}")
        password = sql_cfg.get("password") or getpass.getpass(
            f"Senha MySQL ({sql_cfg['user']}@{sql_cfg['host']}): "
        )
        url = (f"mysql+pymysql://{sql_cfg['user']}:{password}"
               f"@{sql_cfg['host']}:{sql_cfg.get('port', 3306)}"
               f"/{sql_cfg['database']}?charset=utf8mb4")
        extractor = MySQLCatalogExtractor(url, sample_rows=sql_cfg.get("sample_rows", 3))
        extracted = extractor.extract(sql_cfg.get("tables"))
        catalog   = sql_merge(output_path, extracted)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"[✓] SQL extraído: {len(extracted)} tabela(s)")

    # ── 2. Extração API ───────────────────────────────────────────────────────
    api_cfg = config.get("api")
    if api_cfg:
        print(f"\n{'─'*50}")
        print(f"[2/3] Extraindo metadados de API...")
        print(f"{'─'*50}")
        api_extractor = APIEndpointExtractor(
            api_cfg["base_url"],
            auth_token=api_cfg.get("token")
        )
        # Atualiza base_url no catálogo
        catalog["sources"]["api"]["base_url"] = api_cfg["base_url"]

        for ep in api_cfg.get("endpoints", []):
            entry   = api_extractor.build_endpoint_entry(ep["path"], ep["key"])
            catalog = api_merge(output_path, ep["key"], entry)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"[✓] API extraída: {len(api_cfg.get('endpoints', []))} endpoint(s)")

    # ── 3. Extração Planilhas ─────────────────────────────────────────────────
    sheets_cfg = config.get("spreadsheets", [])
    if sheets_cfg:
        print(f"\n{'─'*50}")
        print(f"[3/3] Extraindo metadados de planilhas...")
        print(f"{'─'*50}")
        for sheet_cfg in sheets_cfg:
            sheet_extractor = SpreadsheetCatalogExtractor(
                sheet_cfg["file"],
                sample_rows=sheet_cfg.get("sample_rows", 3)
            )
            extracted = sheet_extractor.extract()
            catalog   = sheet_merge(output_path, sheet_cfg["key"], extracted)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"[✓] Planilhas extraídas: {len(sheets_cfg)} arquivo(s)")

    # Injeta placeholders para embedding_hint em fontes que não possuem o campo
    catalog = _inject_embedding_hints(catalog)

    # Atualiza timestamp final
    catalog["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Data Catalog Builder")
    parser.add_argument("--config", required=True, help="Caminho do build_config.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    print(f"\n{'='*55}")
    print(f"  DATA CATALOG BUILDER — Framework Agentic AI")
    print(f"{'='*55}")

    output = build_catalog(config)

    print(f"\n{'='*55}")
    print(f"  ✅ Catálogo gerado com sucesso!")
    print(f"  📁 Arquivo: {output}")
    print(f"\n  PRÓXIMOS PASSOS:")
    print(f"  1. Abra {output} e preencha os campos [MANUAL]")
    print(f"  2. Preencha: description, business_role, when_to_use")
    print(f"     para cada tabela, endpoint e planilha")
    print(f"  3. Configure as context_hints para roteamento inteligente")
    print(f"  4. Teste o Context Builder:")
    print(f"     python context_builder.py {output}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
