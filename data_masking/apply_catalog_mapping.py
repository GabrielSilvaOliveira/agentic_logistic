"""
Substitui os valores da coluna CODIGO_CATALOGO em materiais.csv
pelo inteiro sequencial correspondente definido em _counter_IDEQP
do mapping_dictionary.json.

Uso:
    python apply_catalog_mapping.py
"""

import json
from pathlib import Path

import pandas as pd

# ── Caminhos ─────────────────────────────────────────────────────────────────
PASTA = Path("dataset_mascarado")
MATERIAIS_CSV     = PASTA / "materiais.csv"
MAPPING_JSON      = PASTA / "mapping_dictionary.json"
MATERIAIS_OUT_CSV = PASTA / "materiais.csv"   # sobrescreve; troque o nome se preferir

# ── Leitura ───────────────────────────────────────────────────────────────────
print(f"Lendo {MATERIAIS_CSV} ...")
df = pd.read_csv(MATERIAIS_CSV, dtype=str, keep_default_na=False)

print(f"Lendo {MAPPING_JSON} ...")
with MAPPING_JSON.open(encoding="utf-8") as f:
    mapping = json.load(f)

counter_map: dict = mapping.get("_counter_IDEQP", {})
if not counter_map:
    raise ValueError("Chave '_counter_IDEQP' não encontrada no mapping_dictionary.json.")

# ── Substituição ──────────────────────────────────────────────────────────────
col = "CODIGO_CATALOGO"

before_unique = df[col].nunique()

def substituir(valor: str):
    if valor == "":
        return valor
    resultado = counter_map.get(valor)
    if resultado is None:
        return valor          # mantém original se não encontrado no mapeamento
    return str(resultado)     # inteiro como string para preservar dtype homogêneo

df[col] = df[col].apply(substituir)

after_unique = df[col].nunique()

# ── Salvamento ────────────────────────────────────────────────────────────────
df.to_csv(MATERIAIS_OUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n[✓] Coluna '{col}' substituída.")
print(f"    Valores únicos antes : {before_unique}")
print(f"    Valores únicos depois: {after_unique}")
print(f"    Total de linhas      : {len(df):,}")
print(f"    Arquivo salvo em     : {MATERIAIS_OUT_CSV.resolve()}")
