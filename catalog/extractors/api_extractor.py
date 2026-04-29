"""
catalog/extractors/api_extractor.py
─────────────────────────────────────────────────────────────────────────────
Extrator semi-automático para endpoints REST → catálogo JSON.

Como funciona:
  1. Faz uma chamada real ao endpoint (GET sem parâmetros)
  2. Inspeciona a resposta: detecta estrutura JSON, campos, tipos
  3. Para o formato "objeto único + metadados" (seu caso):
       { "count": N, "next": ..., "previous": ..., "results": [...] }
  4. Analisa o primeiro item de `results` para mapear os campos
  5. Preenche automaticamente o que pode; deixa [MANUAL] no restante

Uso:
  python api_extractor.py \
         --url http://localhost:8000/api/distribuicao/ \
         --key distribuicao_list \
         --catalog ../output/catalog.json \
         --output ../output/catalog.json
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any

try:
    import requests
except ImportError:
    raise SystemExit("Execute: pip install requests")


# ─────────────────────────────────────────────────────────────────────────────
# INFERÊNCIA DE TIPO A PARTIR DE VALOR JSON
# ─────────────────────────────────────────────────────────────────────────────

def infer_json_type(value: Any) -> str:
    """Infere tipo semântico a partir de um valor JSON de exemplo."""
    if value is None:
        return "null|unknown"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return f"array[{infer_json_type(value[0]) if value else 'unknown'}]"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, str):
        # Detecta datas
        import re
        if re.match(r"\d{4}-\d{2}-\d{2}", value):
            return "date_string (ISO 8601)"
        if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", value):
            return "datetime_string (ISO 8601)"
        if re.match(r"https?://", value):
            return "url_string"
        if len(value) <= 20 and re.match(r"^[A-Z0-9\-_/]+$", value):
            return "code_identifier"
        return "string"
    return type(value).__name__


def infer_fields_from_object(obj: dict) -> dict:
    """Mapeia campos de um objeto JSON para o schema do catálogo."""
    return {
        field: {
            "type":        infer_json_type(value),
            "example":     value if not isinstance(value, (dict, list)) else str(value)[:80],
            "description": f"[MANUAL] Descreva o significado de '{field}' no contexto logístico",
        }
        for field, value in obj.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXTRATOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class APIEndpointExtractor:

    def __init__(self, base_url: str, auth_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        if auth_token:
            self.session.headers["Authorization"] = f"Bearer {auth_token}"
        self.session.headers["Accept"] = "application/json"

    def probe_endpoint(self, path: str, params: dict | None = None) -> dict:
        """
        Faz uma chamada real ao endpoint e analisa a resposta.
        Detecta automaticamente o formato e mapeia os campos.
        """
        url = f"{self.base_url}{path}"
        print(f"  → Chamando: GET {url} ...", end=" ", flush=True)

        try:
            response = self.session.get(url, params=params or {"limit": 1}, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Conexão recusada")
            return self._offline_template(path)
        except requests.exceptions.Timeout:
            print(f"❌ Timeout")
            return self._offline_template(path)
        except Exception as e:
            print(f"❌ Erro: {e}")
            return self._offline_template(path)

        return self._analyze_response(data, url, response.status_code)

    def _analyze_response(self, data: Any, url: str,
                          status_code: int) -> dict:
        """Analisa a estrutura da resposta e infere o schema."""

        # ── Detecta formato: objeto com metadados (seu caso) ─────────────────
        if isinstance(data, dict) and "results" in data:
            return self._analyze_paginated_response(data, url)

        # ── Lista direta ──────────────────────────────────────────────────────
        if isinstance(data, list) and data:
            first_item = data[0] if isinstance(data[0], dict) else {}
            return {
                "response_format":       "array",
                "total_count":           len(data),
                "result_object_fields":  infer_fields_from_object(first_item),
                "structure": {
                    "root": f"array — lista de {len(data)} objeto(s)"
                },
                "example_response":      data[:1],
                "_probed_at":            datetime.utcnow().isoformat() + "Z",
                "_probe_url":            url,
                "_http_status":          status_code,
            }

        # ── Objeto simples ────────────────────────────────────────────────────
        if isinstance(data, dict):
            return {
                "response_format":       "single_object",
                "result_object_fields":  infer_fields_from_object(data),
                "structure":             {k: infer_json_type(v) for k, v in data.items()},
                "example_response":      data,
                "_probed_at":            datetime.utcnow().isoformat() + "Z",
                "_probe_url":            url,
                "_http_status":          status_code,
            }

        return {"response_format": "unknown", "raw_sample": str(data)[:200]}

    def _analyze_paginated_response(self, data: dict, url: str) -> dict:
        """
        Analisa resposta paginada no formato DRF/FastAPI:
        { count, next, previous, results: [...] }
        """
        results = data.get("results", [])
        first   = results[0] if results and isinstance(results[0], dict) else {}

        structure = {}
        for key, value in data.items():
            if key != "results":
                structure[key] = infer_json_type(value)
        structure["results"] = f"array[object] — {len(results)} item(s) nesta página"

        return {
            "response_format":       "paginated_object_with_metadata",
            "total_count_available": data.get("count"),
            "structure":             structure,
            "result_object_fields":  infer_fields_from_object(first),
            "example_response":      {**{k: v for k, v in data.items()
                                         if k != "results"},
                                      "results": results[:1]},
            "_probed_at":            datetime.utcnow().isoformat() + "Z",
            "_probe_url":            url,
        }

    @staticmethod
    def _offline_template(path: str) -> dict:
        """Template usado quando a API não está acessível."""
        return {
            "response_format":       "[MANUAL] — API não acessível durante extração",
            "result_object_fields":  {
                "[MANUAL] campo_1": {
                    "type":        "[MANUAL] Ex: string",
                    "example":     "[MANUAL]",
                    "description": "[MANUAL]",
                }
            },
            "structure":             {"[MANUAL]": "Descreva a estrutura da resposta"},
            "example_response":      {},
            "_probe_attempted_at":   datetime.utcnow().isoformat() + "Z",
            "_probe_path":           path,
            "_probe_status":         "offline",
        }

    def build_endpoint_entry(self, path: str, key: str,
                             params: dict | None = None) -> dict:
        """
        Monta a entrada completa de um endpoint para o catálogo,
        combinando dados auto-extraídos com placeholders manuais.
        """
        probed = self.probe_endpoint(path, params)

        return {
            "path":             path,
            "method":           "GET",
            "description":      "[MANUAL] Descreva em linguagem de negócio o que este endpoint retorna",
            "business_role":    "[MANUAL] Ex: Fonte primária para rastreamento de distribuições",
            "when_to_use":      "[MANUAL] Ex: Quando a pergunta envolver distribuição ou movimentação",
            "when_NOT_to_use":  "[MANUAL] Ex: Não usar para dados de estoque (usar SQL)",
            "parameters": {
                "query_params": {
                    "limit": {
                        "type":        "integer",
                        "required":    False,
                        "default":     100,
                        "description": "Número máximo de registros por página"
                    },
                    "offset": {
                        "type":        "integer",
                        "required":    False,
                        "default":     0,
                        "description": "Índice inicial para paginação"
                    },
                    "[MANUAL] filtro_1": {
                        "type":        "[MANUAL] Ex: string",
                        "required":    False,
                        "description": "[MANUAL] Ex: Filtra por status da distribuição"
                    }
                }
            },
            "response_schema":   probed,
            "data_quality": {
                "update_frequency": "[MANUAL] Ex: Tempo real",
                "reliability":      "[MANUAL] Ex: Alta — fonte transacional",
                "confidence_base":  0.9,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# MERGE COM CATÁLOGO EXISTENTE
# ─────────────────────────────────────────────────────────────────────────────

def merge_into_catalog(catalog_path: str, endpoint_key: str,
                       endpoint_data: dict) -> dict:
    """Mescla o endpoint extraído no catálogo, preservando anotações manuais."""
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    endpoints = (catalog
                 .setdefault("sources", {})
                 .setdefault("api", {})
                 .setdefault("endpoints", {}))

    existing = endpoints.get(endpoint_key, {})

    # Preserva campos manuais já preenchidos
    for field in ("description", "business_role", "when_to_use", "when_NOT_to_use"):
        if field in existing and "[MANUAL]" not in str(existing[field]):
            endpoint_data[field] = existing[field]

    # Preserva descrições de campos já preenchidas
    existing_fields = (existing.get("response_schema", {})
                               .get("result_object_fields", {}))
    for field_name, field_data in endpoint_data.get(
            "response_schema", {}).get("result_object_fields", {}).items():
        if field_name in existing_fields:
            ex = existing_fields[field_name]
            if "description" in ex and "[MANUAL]" not in str(ex["description"]):
                field_data["description"] = ex["description"]

    endpoints[endpoint_key] = endpoint_data
    catalog["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    return catalog


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrator API REST → Data Catalog")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL da API")
    parser.add_argument("--path",     default="/api/distribuicao/",
                        help="Path do endpoint (relativo à base URL)")
    parser.add_argument("--key",      default="distribuicao_list",
                        help="Chave do endpoint no catálogo")
    parser.add_argument("--token",    default=None,
                        help="Bearer token de autenticação (opcional)")
    parser.add_argument("--catalog",  default="../output/catalog.json",
                        help="Catálogo existente para mesclar")
    parser.add_argument("--output",   default="../output/catalog.json",
                        help="Caminho de saída")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  API Catalog Extractor")
    print(f"  Endpoint : {args.base_url}{args.path}")
    print(f"  Chave    : {args.key}")
    print(f"  Saída    : {args.output}")
    print(f"{'='*55}\n")

    extractor = APIEndpointExtractor(args.base_url, auth_token=args.token)
    endpoint_entry = extractor.build_endpoint_entry(args.path, args.key)

    if os.path.exists(args.catalog):
        catalog = merge_into_catalog(args.catalog, args.key, endpoint_entry)
        print(f"\n[✓] Mesclado com catálogo existente: {args.catalog}")
    else:
        catalog = {
            "metadata": {"updated_at": datetime.utcnow().isoformat() + "Z"},
            "sources":  {"api": {"base_url": args.base_url,
                                 "endpoints": {args.key: endpoint_entry}}}
        }
        print(f"\n[!] Catálogo existente não encontrado. Criando novo.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"[✓] Catálogo salvo em: {args.output}")
    print(f"\nPróximo passo: preencha os campos [MANUAL] no arquivo gerado.\n")


if __name__ == "__main__":
    main()
