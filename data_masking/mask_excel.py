"""
=============================================================================
DATA MASKING PARA PLANILHAS EXCEL — Dissertação de Mestrado
=============================================================================
Objetivo: Mascarar dados sensíveis em planilhas Excel preservando estrutura e
          integridade referencial entre colunas.

Tipos de dado mascarados:
  - Equipamentos / materiais        → EQP001, EQP002 ...
  - Fabricantes / fornecedores      → FAB001, FAB002 ...
  - Locais / unidades / endereços   → LOC001, LOC002 ...
  - Pessoas / usuários              → USR001, USR002 ...
  - Valores monetários              → escala proporcional com ruído
  - Datas                           → deslocamento fixo por entidade

Como usar:
  1. Configure PLANILHAS_E_COLUNAS com suas planilhas e colunas reais
  2. Execute: python mask_excel.py
  3. O arquivo mascarado será gerado na mesma pasta da planilha original
=============================================================================
"""

import pandas as pd
import random
import hashlib
from datetime import timedelta, datetime
from pathlib import Path
import json

# =============================================================================
# CONFIGURAÇÃO DAS PLANILHAS E COLUNAS SENSÍVEIS
#
# Para cada planilha, defina:
#   "tipo_coluna" → qual máscara aplicar:
#     "equipamento"  → EQP001, EQP002 ...
#     "fabricante"   → FAB001, FAB002 ...
#     "local"        → LOC001, LOC002 ...
#     "usuario"      → USR001, USR002 ...
#     "valor"        → escala proporcional (mantém distribuição)
#     "data"         → deslocamento temporal fixo
#     "texto_livre"  → substitui por hash curto
#     "manter"       → não mascara (IDs, flags, quantidades, etc.)
#
# EXEMPLO REAL → preencha com seus nomes reais de planilha/coluna:
# =============================================================================
# Diretório onde as planilhas estão localizadas
DIRETORIO_PLANILHAS = "C:\\Users\\Usuario\\Documents\\Mestrado\\csv_banco_de_dados\\Planilhas\\"

# Atualiza o dicionário para buscar as planilhas no diretório especificado
PLANILHAS_E_COLUNAS = {
    (DIRETORIO_PLANILHAS + "contratos.xlsx"): {
        "ID_CONTRATO_ANO": "manter",
        "ANO": "manter",
        "CONTRATO": "contrato",
        "EMPRESA": "manter",
        "ASS DO CTR": "manter",
        "VIGÊNCIA": "manter",
        "ADT": "manter",
        "Aditivação": "manter",
        "FUNDS-1": "funds",
        "FUNDS-2": "funds",
        "FUNDS-3": "funds",
        "FUNDS-4": "funds",
        "FUNDS-5": "funds",
        "VALOR USD (contrato)": "valor",
        "ENTREGA 1": "manter",
        "ENTREGA 2": "manter",
        "ENTREGA 3": "manter",
        "ENTREGA 4": "manter",
        "ENTREGA 5": "manter",
        "ENTREGA 6": "manter",
        "FORMA_DE_ENVIO": "manter",
        "UNIDADE_ESTOQUE": "manter",
        "QI": "manter",
        "ITEM QI": "manter",
        "ITEM PC": "manter",
        "MATERIAL": "equipamento",
        "Quantidade": "manter",
        "Invoice": "invoice",
        "Andamento": "manter",
        "Guarantia": "manter",
        "Chegada no BR": "manter",
        "Inicio da Garantia": "manter",
        "Fim da Garantia": "manter",
        "DOC_RECEBIMENTO": "manter",
    },
}

# =============================================================================
# PARÂMETROS DE MASCARAMENTO
# =============================================================================

# Deslocamento de datas: todas as datas são deslocadas pelo mesmo número de dias
# (mantém intervalos e sazonalidade, mas datas absolutas ficam fictícias)
DATE_OFFSET_DAYS = random.randint(180, 730)  # entre 6 meses e 2 anos

# Ruído nos valores monetários: ±N% aleatório aplicado sobre a escala
VALOR_NOISE_PERCENT = 0.05   # 5% de ruído — mantém distribuição próxima

# Fator de escala global para valores (multiplica todos os valores por este fator)
# Útil para não revelar a magnitude real. Ex.: 0.1 divide tudo por 10.
VALOR_SCALE_FACTOR = 1.0     # 1.0 = sem escala, ajuste se necessário

# =============================================================================
# IMPLEMENTAÇÃO DO PIPELINE
# =============================================================================

class MaskingDictionary:
    """Gerencia dicionários de mapeamento por categoria, com persistência em JSON."""

    PREFIXOS = {
        "equipamento": ("MAT", 3),
        "fabricante":  ("FAB", 3),
        "local":       ("LOC", 3),
        "usuario":     ("USR", 3),
        "contrato":    ("CTR", 3),
        "funds":       ("FND", 3),
        "invoice":     ("INV", 3),
    }

    def __init__(self, seed: int = 42):
        self._maps: dict[str, dict] = {k: {} for k in self.PREFIXOS}
        self._counters: dict[str, int] = {k: 1 for k in self.PREFIXOS}
        self._rng = random.Random(seed)

    def mask_entity(self, category: str, original_value) -> str:
        """Retorna o código mascarado para um valor, criando se não existir."""
        if original_value is None or (isinstance(original_value, float) and
                                       original_value != original_value):
            return None

        key = str(original_value).strip()
        if not key:
            return ""

        if key not in self._maps[category]:
            prefix, digits = self.PREFIXOS[category]
            code = f"{prefix}{str(self._counters[category]).zfill(digits)}"
            self._maps[category][key] = code
            self._counters[category] += 1

        return self._maps[category][key]

    def mask_composite_entity(self, category: str, original_value: str, masked_value: str):
        """Armazena o mapeamento de um valor composto."""
        if original_value not in self._maps[category]:
            self._maps[category][original_value] = masked_value
        return self._maps[category][original_value]


def mask_value(value, rng: random.Random, noise: float, scale: float):
    """Mascara valor monetário com escala + ruído proporcional."""
    if value is None:
        return None
    try:
        v = float(value)
        noise_factor = 1 + rng.uniform(-noise, noise)
        return round(v * scale * noise_factor, 2)
    except (TypeError, ValueError):
        return value


def mask_date(value, offset_days: int):
    """Desloca uma data por um número fixo de dias."""
    if value is None:
        return None
    try:
        if isinstance(value, datetime):
            return value + timedelta(days=offset_days)
        # string
        dt = datetime.fromisoformat(str(value))
        return dt + timedelta(days=offset_days)
    except (TypeError, ValueError):
        return value


def mask_text(value) -> str:
    """Substitui texto livre por hash curto (8 chars), preservando None/vazio."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return s
    h = hashlib.md5(s.encode()).hexdigest()[:8].upper()
    return f"TXT_{h}"


def mask_dataframe(df: pd.DataFrame, column_config: dict, md: MaskingDictionary,
                   rng: random.Random, date_offset: int, noise: float, scale: float) -> pd.DataFrame:
    """Aplica mascaramento a um DataFrame conforme configuração de colunas."""
    TIPOS_PREFIXO = ("equipamento", "fabricante", "local", "usuario", "contrato", "funds", "invoice")

    df = df.copy()

    for col, tipo in column_config.items():
        if col not in df.columns:
            print(f"    [!] Coluna '{col}' não encontrada na planilha — ignorada.")
            continue

        if tipo == "manter":
            continue  # sem alteração

        elif tipo in TIPOS_PREFIXO:
            df[col] = df[col].apply(lambda v: md.mask_entity(tipo, v))

        elif tipo == "valor":
            df[col] = df[col].apply(lambda v: mask_value(v, rng, noise, scale))

        elif tipo == "data":
            df[col] = df[col].apply(lambda v: mask_date(v, date_offset))

        elif tipo == "texto_livre":
            df[col] = df[col].apply(mask_text)

        elif col == "ID_CONTRATO_ANO":
            if "CONTRATO" in df.columns and "ANO" in df.columns:
                df[col] = df.apply(
                    lambda row: md.mask_composite_entity(
                        "contrato_ano",
                        f"{row['CONTRATO']}-{str(row['ANO'])[-2:]}",
                        f"{row['CONTRATO']}_MASK-{str(row['ANO'])[-2:]}"
                    ) if pd.notnull(row['CONTRATO']) and pd.notnull(row['ANO']) else None,
                    axis=1
                )
            else:
                print("    [!] Colunas 'CONTRATO' e 'ANO' necessárias para 'ID_CONTRATO_ANO' não encontradas.")

        else:
            print(f"    [!] Tipo desconhecido '{tipo}' para coluna '{col}' — ignorado.")

    return df


def run_pipeline():
    print("=" * 65)
    print("  DATA MASKING PARA PLANILHAS EXCEL")
    print("=" * 65)

    random.seed(42)
    rng = random.Random(42)
    md = MaskingDictionary(seed=42)
    

    output_dir = Path(DIRETORIO_PLANILHAS) / "mascarado"
    output_dir.mkdir(exist_ok=True)

    for planilha, config in PLANILHAS_E_COLUNAS.items():
        print(f"\n[→] Processando planilha: {planilha}")
        try:
            path = Path(planilha)
            if not path.exists():
                print(f"  [✗] Planilha '{planilha}' não encontrada.")
                continue

            df = pd.read_excel(path)
            print(f"  Lidas {len(df):,} linhas × {len(df.columns)} colunas")

            df_masked = mask_dataframe(
                df, config, md, rng,
                date_offset=DATE_OFFSET_DAYS,
                noise=VALOR_NOISE_PERCENT,
                scale=VALOR_SCALE_FACTOR,
            )

            output_path = output_dir / f"{path.stem}_mascarado{path.suffix}"
            df_masked.to_excel(output_path, index=False)
            print(f"  [✓] Planilha mascarada salva em: {output_path}")

        except Exception as e:
            print(f"  [✗] Erro ao processar '{planilha}': {e}")

    # Salva o dicionário de mapeamento após processar todas as planilhas
    save_mapping_dictionary(md, output_dir)
    print("\nPipeline concluído com sucesso!")
    print("=" * 65)


def save_mapping_dictionary(md: MaskingDictionary, output_dir: Path):
    """Salva o dicionário de mapeamento em um arquivo JSON."""
    mapping_path = output_dir / "mapping_dictionary_excel.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(md._maps, f, indent=2, ensure_ascii=False)
    print(f"  [✓] Dicionário de mapeamento salvo em: {mapping_path}")


if __name__ == "__main__":
    run_pipeline()