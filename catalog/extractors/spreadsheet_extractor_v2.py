"""
catalog/extractors/spreadsheet_extractor.py
─────────────────────────────────────────────────────────────────────────────
Extrator automático de metadados de planilhas Excel/CSV → catálogo JSON.

O que extrai automaticamente:
  ✅ Nome e quantidade de abas
  ✅ Colunas com tipos inferidos (int, float, date, string, boolean)
  ✅ Contagem de linhas por aba
  ✅ Taxa de nulos por coluna
  ✅ Amostra de N linhas
  ✅ Estatísticas básicas de colunas numéricas (min, max, média)
  ✅ Detecção de colunas de data

Uso:
  python spreadsheet_extractor.py \
         --file ../../data/contratos.xlsx \
         --catalog ../output/catalog.json \
         --output ../output/catalog.json
"""

import argparse
import json
import os
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise SystemExit("Execute: pip install pandas openpyxl numpy")


# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZADOR
# ─────────────────────────────────────────────────────────────────────────────

def json_safe(obj: Any) -> Any:
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat() if not pd.isna(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    if pd.isna(obj) if not isinstance(obj, (list, dict, str)) else False:
        return None
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# INFERÊNCIA DE TIPO SEMÂNTICO
# ─────────────────────────────────────────────────────────────────────────────

def infer_semantic_type(series: pd.Series) -> str:
    """
    Infere o tipo semântico de uma coluna além do dtype do pandas.
    Retorna string legível para o catálogo.
    """
    dtype = str(series.dtype)

    if "datetime" in dtype or "timestamp" in dtype.lower():
        return "datetime"

    if dtype in ("int64", "int32", "int16", "int8", "uint64"):
        if series.nunique() == 2:
            return "boolean_integer"
        return "integer"

    if dtype in ("float64", "float32"):
        return "float"

    if dtype == "bool":
        return "boolean"

    if dtype == "object":
        sample = series.dropna().head(50)
        # Tenta detectar datas em string
        try:
            pd.to_datetime(sample, errors="raise")
            return "date_string"
        except Exception:
            pass
        # Detecta se parece código/ID (curto, alfanumérico)
        if sample.str.len().max() <= 20 and sample.str.match(r"^[A-Z0-9\-_/]+$").mean() > 0.8:
            return "code_identifier"
        # Detecta CPF/CNPJ
        if sample.str.match(r"^\d{2,3}\.\d{3}\.\d{3}").mean() > 0.5:
            return "document_number"
        return "string"

    return dtype


# ─────────────────────────────────────────────────────────────────────────────
# EXTRATOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class SpreadsheetCatalogExtractor:

    def __init__(self, filepath: str, sample_rows: int = 3):
        self.filepath = Path(filepath)
        self.sample_rows = sample_rows

        if not self.filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    def extract(self) -> dict:
        """
        Extrai metadados de todas as abas do arquivo.
        Suporta .xlsx, .xls, .csv.
        """
        suffix = self.filepath.suffix.lower()
        print(f"[Spreadsheet Extractor] Arquivo: {self.filepath.name}")

        if suffix == ".csv":
            return self._extract_csv()
        elif suffix in (".xlsx", ".xls"):
            return self._extract_excel()
        else:
            raise ValueError(f"Formato não suportado: {suffix}. Use .xlsx, .xls ou .csv")

    def _extract_excel(self) -> dict:
        """Extrai todas as abas de um arquivo Excel."""
        xl = pd.ExcelFile(self.filepath)
        sheet_names = xl.sheet_names
        print(f"  Abas encontradas: {sheet_names}")

        sheets = {}
        for sheet in sheet_names:
            print(f"  → Extraindo aba '{sheet}' ...", end=" ", flush=True)
            try:
                df = pd.read_excel(xl, sheet_name=sheet)
                sheets[sheet] = self._extract_sheet(df, sheet)
                print(f"✅ ({sheets[sheet]['row_count']:,} linhas × "
                      f"{sheets[sheet]['column_count']} colunas)")
            except Exception as e:
                print(f"❌ Erro: {e}")
                sheets[sheet] = {"error": str(e)}

        return {
            "filename":    self.filepath.name,
            "description": f"[MANUAL] Descreva o propósito da planilha '{self.filepath.name}'",
            "business_role": "[MANUAL] Ex: Qual o papel desta planilha no processo logístico?",
            "when_to_use": "[MANUAL] Ex: Quando o agente deve consultar esta planilha?",
            "when_NOT_to_use": "[MANUAL] Ex: Quando NÃO usar esta planilha?",
            "sheets":      sheets,
            "data_quality": {
                "update_frequency": "[MANUAL] Ex: Mensal / Semanal / Sob demanda",
                "responsible":      "[MANUAL] Ex: Seção de Contratos",
                "reliability":      "[MANUAL] Ex: Média — atualização manual",
                "confidence_base":  0.75,
                "last_analyzed":    datetime.utcnow().isoformat() + "Z",
            }
        }

    def _extract_csv(self) -> dict:
        """Extrai metadados de um CSV."""
        df = pd.read_csv(self.filepath, encoding="utf-8-sig")
        sheet_name = self.filepath.stem
        print(f"  → CSV com {len(df):,} linhas × {len(df.columns)} colunas")
        return {
            "filename":    self.filepath.name,
            "description": f"[MANUAL]",
            "business_role": "[MANUAL]",
            "when_to_use": "[MANUAL]",
            "when_NOT_to_use": "[MANUAL]",
            "sheets":      {sheet_name: self._extract_sheet(df, sheet_name)},
            "data_quality": {
                "update_frequency": "[MANUAL]",
                "responsible":      "[MANUAL]",
                "reliability":      "[MANUAL]",
                "confidence_base":  0.75,
                "last_analyzed":    datetime.utcnow().isoformat() + "Z",
            }
        }

    def _extract_sheet(self, df: pd.DataFrame, sheet_name: str) -> dict:
        """Extrai metadados de um DataFrame (aba ou CSV)."""

        columns = {}
        for col in df.columns:
            series = df[col]
            sem_type = infer_semantic_type(series)
            null_rate = round(float(series.isna().mean()), 4)
            cardinality = int(series.nunique())

            col_info = {
                "type":          sem_type,
                "nullable":      null_rate > 0,
                "null_rate":     null_rate,
                "cardinality":   cardinality,
                "is_unique_key": cardinality == len(df) and null_rate == 0,
                "description":   f"[MANUAL] Descreva o significado de '{col}'",
            }

            # Estatísticas extras para numéricos
            if sem_type in ("integer", "float"):
                col_info["stats"] = {
                    "min":  json_safe(series.min()),
                    "max":  json_safe(series.max()),
                    "mean": json_safe(round(series.mean(), 2)) if sem_type == "float" else None,
                }

            # Range de datas
            if "date" in sem_type or "datetime" in sem_type:
                try:
                    parsed = pd.to_datetime(series, errors="coerce")
                    col_info["date_range"] = {
                        "min": json_safe(parsed.min()),
                        "max": json_safe(parsed.max()),
                    }
                except Exception:
                    pass

            columns[col] = col_info

        # Amostra de dados
        sample_df = df.head(self.sample_rows)
        sample = json.loads(
            sample_df.to_json(orient="records", date_format="iso",
                              force_ascii=False, default_handler=str)
        )

        return {
            "description":  f"[MANUAL] Descreva o conteúdo da aba '{sheet_name}'",
            "row_count":    int(len(df)),
            "column_count": int(len(df.columns)),
            "columns":      columns,
            "sample_rows":  sample,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MERGE COM CATÁLOGO EXISTENTE
# ─────────────────────────────────────────────────────────────────────────────

def merge_into_catalog(catalog_path: str, filename_key: str,
                       extracted: dict) -> dict:
    """
    Carrega o catálogo existente e preenche/atualiza a entrada da planilha,
    preservando anotações manuais já existentes.
    """
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    spreadsheets = catalog.setdefault("sources", {}).setdefault("spreadsheets", {})
    spreadsheets.setdefault("files", {})

    existing = spreadsheets["files"].get(filename_key, {})

    # Preserva campos manuais já preenchidos
    for field in ("description", "business_role", "when_to_use", "when_NOT_to_use", "embedding_hint"):
        if field in existing and "[MANUAL]" not in str(existing[field]):
            extracted[field] = existing[field]

    # Preserva descrições de colunas já preenchidas
    for sheet_name, sheet_data in extracted.get("sheets", {}).items():
        existing_sheet = existing.get("sheets", {}).get(sheet_name, {})
        for col_name, col_data in sheet_data.get("columns", {}).items():
            existing_col = existing_sheet.get("columns", {}).get(col_name, {})
            if "description" in existing_col and "[MANUAL]" not in str(existing_col["description"]):
                col_data["description"] = existing_col["description"]

    spreadsheets["files"][filename_key] = extracted
    catalog["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    return catalog


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrator Planilha → Data Catalog")
    parser.add_argument("--file",    required=True, help="Caminho da planilha (.xlsx ou .csv)")
    parser.add_argument("--key",     default=None,  help="Chave no catálogo (padrão: nome do arquivo sem extensão)")
    parser.add_argument("--sample",  default=3, type=int, help="Linhas de amostra por aba")
    parser.add_argument("--catalog", default="../output/catalog.json",
                        help="Catálogo existente para mesclar (opcional)")
    parser.add_argument("--output",  default="../output/catalog.json",
                        help="Caminho de saída")
    args = parser.parse_args()

    key = args.key or Path(args.file).stem

    print(f"\n{'='*55}")
    print(f"  Spreadsheet Catalog Extractor")
    print(f"  Arquivo : {args.file}")
    print(f"  Chave   : {key}")
    print(f"  Saída   : {args.output}")
    print(f"{'='*55}\n")

    extractor = SpreadsheetCatalogExtractor(args.file, sample_rows=args.sample)
    extracted = extractor.extract()

    if os.path.exists(args.catalog):
        catalog = merge_into_catalog(args.catalog, key, extracted)
        print(f"\n[✓] Mesclado com catálogo existente: {args.catalog}")
    else:
        catalog = {
            "metadata": {"updated_at": datetime.utcnow().isoformat() + "Z"},
            "sources":  {"spreadsheets": {"files": {key: extracted}}}
        }
        print(f"\n[!] Catálogo existente não encontrado. Criando novo.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=json_safe)

    print(f"[✓] Catálogo salvo em: {args.output}")
    print(f"\nPróximo passo: preencha os campos [MANUAL] no arquivo gerado.\n")


if __name__ == "__main__":
    main()
