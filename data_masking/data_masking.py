"""
=============================================================================
DATA MASKING PIPELINE — Dissertação de Mestrado
=============================================================================
Objetivo: Mascarar dados sensíveis preservando estrutura relacional e
          integridade referencial entre tabelas.

Tipos de dado mascarados:
  - Equipamentos / materiais        → EQP001, EQP002 ...
  - Fabricantes / fornecedores      → FAB001, FAB002 ...
  - Locais / unidades / endereços   → LOC001, LOC002 ...
  - Pessoas / usuários              → USR001, USR002 ...
  - Valores monetários              → escala proporcional com ruído
  - Datas                           → deslocamento fixo por entidade

Como usar:
  1. Configure CONFIGURAÇÃO DO BANCO abaixo
  2. Configure TABELAS_E_COLUNAS com suas tabelas reais
  3. Execute: python data_masking_pipeline.py
  4. O arquivo mapping_dictionary.json é gerado — guarde-o com segurança
     (permite rastrear o mapeamento original ↔ mascarado se necessário)
=============================================================================
"""

import json
import random
import hashlib
import re
from datetime import timedelta, date, datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDÊNCIAS — instale com:  pip install sqlalchemy pandas faker
# ─────────────────────────────────────────────────────────────────────────────
try:
    import sqlalchemy as sa
    import pandas as pd
except ImportError:
    raise SystemExit("Execute: pip install sqlalchemy pandas faker")

# =============================================================================
# ① CONFIGURAÇÃO DO BANCO DE DADOS
#    Ajuste a connection string para o seu banco.
#    Exemplos:
#      PostgreSQL : "postgresql+psycopg2://user:senha@host:5432/dbname"
#      MySQL      : "mysql+pymysql://user:senha@host:3306/dbname"
#      SQL Server : "mssql+pyodbc://user:senha@host/dbname?driver=ODBC+Driver+17+for+SQL+Server"
#      SQLite     : "sqlite:///meu_banco.db"
# =============================================================================
DATABASE_URL = "mysql+pymysql://root:EngComp2020!@localhost:3306/article_logistics"

# Banco de destino — pode ser o mesmo (esquema diferente) ou um novo banco
# Se None, gera CSVs em vez de gravar no banco
DATABASE_DESTINO_URL = None  # ou: "mysql+pymysql://user:senha@host:3306/banco_mascarado"
ESQUEMA_DESTINO = "masked"   # usado se gravar no mesmo banco, em esquema separado

# Pasta para salvar CSVs mascarados (sempre gerados, independente do banco destino)
PASTA_SAIDA = Path("dataset_mascarado")

# Semente aleatória — garante reprodutibilidade
RANDOM_SEED = 42

# =============================================================================
# ② MAPEAMENTO DE TABELAS E COLUNAS SENSÍVEIS
#
#    Para cada tabela, defina:
#      "tipo_coluna" → qual máscara aplicar:
#        "equipamento"  → EQP001, EQP002 ...
#        "fabricante"   → FAB001, FAB002 ...
#        "local"        → LOC001, LOC002 ...
#        "usuario"      → USR001, USR002 ...
#        "valor"        → escala proporcional (mantém distribuição)
#        "data"         → deslocamento temporal fixo
#        "texto_livre"  → substitui por hash curto
#        "manter"       → não mascara (IDs, flags, quantidades, etc.)
#
#    EXEMPLO REAL → preencha com seus nomes reais de tabela/coluna:
# =============================================================================
TABELAS_E_COLUNAS = {

    # ── Tabela de materiais/equipamentos ─────────────────────────────────────
    "materiais": {
        "CODIGO": "manter",
        "CODIGO_UNIDADE": "manter",
        "CODIGO_CATALOGO": "manter",	
        "SN": "numero_serie",  # número de série → SN001, SN002 ...
        "SITUACAOPARTIM": "manter",
        "MOTIVOPATRIMONIAL": "manter",
        "SITUACAOFISIC": "manter",
        "MOTIVOINDISP": "manter",
        "HOMOLOGAR": "manter",
        "ACAUTELADO": "manter",
        "MOTIVOCAUTELADO": "manter",
        "OMCAUTELADO": "manter",
        "OBS": "texto_livre",  # texto livre → hash curto
        "STATUSDESCARGA": "manter",
        "DOCREFERENCIA": "manter",
        "DATADESCARGA": "manter",
        "DATAINCLUSAOCARGA": "manter",
        "CONFERIDO": "manter",
        "CONFERIDOEM": "manter",
        "DEPENDENCIA": "manter",
        "PT": "manter",
        "TEAM": "manter",
        "ESTADOMATERIAL": "manter",
        "PARECER": "manter",
        "DESTINOMATERIAL": "manter",
        "BOLETIMREGIONAL": "manter",
        "EXCLUIRMATERIAL": "manter",
        "JUSTIFICMATERIAL": "texto_livre",
        "AUTORIZADO": "manter",
        "DATAAUTORIZADO": "manter",
        "HOMOLOGADO": "manter",
        "ULTIMAATUALIZACAO": "manter",
        "MOTIVOEXCLUIR": "manter",
        "motivoreprovacao": "manter",
        "nec_mnt": "manter",
        "prioridade": "manter",
        "usuario_decisao_id": "manter",
    },

    # ── Tabela de catalogo de materiais/fornecedores ────────────────────────────────────
    "catalogo": {
        "IDEQP": "contador",
        "ID_SISTEMA_IDENTIFICACAO": "manter",
        "NUMERO_DE_ESTOQUE": "numero_estoque",
        "NOME": "equipamento",  # "Rádio HF AN/PRC-150" → EQP001
        "TIPO": "manter",
        "MODELO": "modelo",
        "FABRICANTE": "fabricante",
        "RADIO": "manter",
        "PN": "part_number",
        "SUBSISTEMA": "subsistema",
        "GRUPO": "grupo",
        "TIPOEQP": "tipo_equipamento",
    },

    # ── Tabela de unidades/localidades ────────────────────────────────────────
    "unidades": {
        "CODIGO": "manter",
        "CODLOC": "manter",
        "SIGLA": "sigla",
        "REGIAO": "regiao",
        "AREA": "area",
        "DIVISAO": "divisao",
        "CIDADEESTADO": "manter",
        "Tipo": "manter",
        "UNIDADE_SUPERIOR": "unidade_superior",
        "ENDERECO": "endereco",
        "CEP": "cep",
        "NIVEL": "manter",
        "ID": "manter",
        "CODREGRA": "manter",
        "CIDADE": "manter",
        "ESTADO": "manter",
        "LATITUDE": "valor",
        "LONGITUDE": "valor",
    },

    # ── Adicione suas outras tabelas seguindo o mesmo padrão ─────────────────
    # "tb_outra_tabela": {
    #     "coluna_a": "manter",
    #     "coluna_b": "equipamento",
    # },
}

# =============================================================================
# ③ PARÂMETROS DE MASCARAMENTO
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
        "equipamento": ("EQP", 3),
        "fabricante":  ("FAB", 3),
        "local":       ("LOC", 3),
        "usuario":     ("USR", 3),
        "numero_serie": ("SN", 5),
        "numero_estoque": ("EST", 5),
        "modelo": ("MDL", 4),
        "part_number": ("PN", 5),
        "subsistema": ("SUB", 2),
        "grupo": ("GRP", 2),
        "tipo_equipamento": ("TEQ", 3),
        "unidade_superior": ("UNI", 3),
        "endereco": ("END", 4),
        "cep": ("CEP", 5),
        "sigla": ("SIG", 3),
        "area": ("AREA", 3),
        "regiao": ("REG", 3),
        "divisao": ("DIV", 3),
    }

    def __init__(self, seed: int = 42):
        self._maps: dict[str, dict] = {k: {} for k in self.PREFIXOS}
        self._counters: dict[str, int] = {k: 1 for k in self.PREFIXOS}
        self._counter_maps: dict[str, dict] = {}   # para tipo "contador"
        self._counter_vals: dict[str, int] = {}    # contador numérico por categoria
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

    def mask_counter(self, category: str, original_value) -> int | None:
        """Retorna um inteiro sequencial para cada valor único dentro de uma categoria."""
        if original_value is None or (isinstance(original_value, float) and
                                       original_value != original_value):
            return None

        key = str(original_value).strip()
        if not key:
            return None

        if category not in self._counter_maps:
            self._counter_maps[category] = {}
            self._counter_vals[category] = 1

        if key not in self._counter_maps[category]:
            self._counter_maps[category][key] = self._counter_vals[category]
            self._counter_vals[category] += 1

        return self._counter_maps[category][key]

    def to_dict(self) -> dict:
        result = dict(self._maps)
        result.update({f"_counter_{k}": v for k, v in self._counter_maps.items()})
        return result

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"  [✓] Dicionário de mapeamento salvo em: {path}")

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
        if isinstance(value, date):
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


def mask_dataframe(df: pd.DataFrame,
                   column_config: dict,
                   md: MaskingDictionary,
                   rng: random.Random,
                   date_offset: int,
                   noise: float,
                   scale: float) -> pd.DataFrame:
    """Aplica mascaramento a um DataFrame conforme configuração de colunas."""
    TIPOS_PREFIXO = ("equipamento", "fabricante", "local", "usuario",
                     "numero_serie", "numero_estoque", "modelo", "part_number",
                     "subsistema", "grupo", "tipo_equipamento", "unidade_superior",
                     "endereco", "cep", "sigla", "area", "regiao", "divisao")

    df = df.copy()

    for col, tipo in column_config.items():
        if col not in df.columns:
            print(f"    [!] Coluna '{col}' não encontrada na tabela — ignorada.")
            continue

        if tipo == "manter":
            continue  # sem alteração

        elif tipo == "contador":
            df[col] = df[col].apply(lambda v, c=col: md.mask_counter(c, v))

        elif tipo in TIPOS_PREFIXO:
            df[col] = df[col].apply(lambda v: md.mask_entity(tipo, v))

        elif tipo == "valor":
            df[col] = df[col].apply(
                lambda v: mask_value(v, rng, noise, scale))

        elif tipo == "data":
            df[col] = df[col].apply(lambda v: mask_date(v, date_offset))

        elif tipo == "texto_livre":
            df[col] = df[col].apply(mask_text)

        else:
            print(f"    [!] Tipo desconhecido '{tipo}' para coluna '{col}' — ignorado.")

    return df


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def run_pipeline():
    print("=" * 65)
    print("  DATA MASKING PIPELINE — Dissertação de Mestrado")
    print("=" * 65)

    random.seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)
    PASTA_SAIDA.mkdir(exist_ok=True)

    md = MaskingDictionary(seed=RANDOM_SEED)

    print(f"\n[1/4] Conectando ao banco de dados...")
    try:
        engine = sa.create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        print("  [✓] Conexão estabelecida.")
    except Exception as e:
        print(f"  [✗] Falha na conexão: {e}")
        print("\n  MODO DEMO: gerando dados sintéticos para demonstração...")
        run_demo(md, rng)
        return

    print(f"\n[2/4] Processando {len(TABELAS_E_COLUNAS)} tabelas...")

    masked_dfs = {}
    for tabela, config in TABELAS_E_COLUNAS.items():
        print(f"\n  → Tabela: {tabela}")
        try:
            df = pd.read_sql_table(tabela, engine)
            print(f"    Lidas {len(df):,} linhas × {len(df.columns)} colunas")

            df_masked = mask_dataframe(
                df, config, md, rng,
                date_offset=DATE_OFFSET_DAYS,
                noise=VALOR_NOISE_PERCENT,
                scale=VALOR_SCALE_FACTOR,
            )
            masked_dfs[tabela] = df_masked
            print(f"    [✓] Mascaramento concluído.")
        except Exception as e:
            print(f"    [✗] Erro: {e}")

    print(f"\n[3/4] Salvando CSVs mascarados em '{PASTA_SAIDA}/'...")
    for tabela, df in masked_dfs.items():
        path = PASTA_SAIDA / f"{tabela}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  [✓] {path}  ({len(df):,} linhas)")

    if DATABASE_DESTINO_URL:
        print(f"\n[3b] Gravando no banco de destino...")
        engine_dest = sa.create_engine(DATABASE_DESTINO_URL)
        for tabela, df in masked_dfs.items():
            df.to_sql(tabela, engine_dest, schema=ESQUEMA_DESTINO,
                      if_exists="replace", index=False)
            print(f"  [✓] {ESQUEMA_DESTINO}.{tabela}")

    print(f"\n[4/4] Salvando dicionário de mapeamento...")
    md.save(PASTA_SAIDA / "mapping_dictionary.json")

    print(f"\n{'=' * 65}")
    print(f"  Pipeline concluído com sucesso!")
    print(f"  Tabelas processadas : {len(masked_dfs)}")
    print(f"  Saída               : {PASTA_SAIDA.resolve()}")
    print(f"  Deslocamento datas  : +{DATE_OFFSET_DAYS} dias")
    print(f"  Ruído em valores    : ±{VALOR_NOISE_PERCENT * 100:.0f}%")
    print(f"{'=' * 65}\n")


# =============================================================================
# MODO DEMO — roda sem banco real, gera dados sintéticos para testar o pipeline
# =============================================================================

def run_demo(md: MaskingDictionary, rng: random.Random):
    """Demonstra o pipeline com dados fictícios sem necessidade de banco real."""
    print("\n  Gerando dados de demonstração...\n")
    PASTA_SAIDA.mkdir(exist_ok=True)

    # Fabricantes fictícios
    fabricantes_orig = ["Motorola", "Harris Corporation", "Rohde & Schwarz",
                        "Thales Group", "Leonardo S.p.A"]
    df_fab = pd.DataFrame({
        "id_fabricante":   range(1, 6),
        "nome_fabricante": fabricantes_orig,
        "cnpj":            ["12.345.678/0001-99"] * 5,
        "endereco":        ["Av. Paulista, 1000", "Rua XV, 200", "Av. Rio Branco, 50",
                            "Rua Augusta, 800", "Av. Atlântica, 3000"],
        "cidade":          ["São Paulo", "Curitiba", "Rio de Janeiro",
                            "Belo Horizonte", "Rio de Janeiro"],
    })

    # Equipamentos fictícios
    equipamentos_orig = ["Rádio HF AN/PRC-150", "Estação VHF MBITR", "Cripto KG-175",
                         "Terminal SATCOM AN/PSC-5", "Rádio UHF PRC-117G",
                         "Antena SINCGARS", "Repetidor VHF RT-1547"]
    df_mat = pd.DataFrame({
        "id_material":    range(1, 8),
        "nome_material":  equipamentos_orig,
        "cod_fabricante": [1, 2, 3, 4, 1, 2, 3],
        "descricao":      ["Comunicações táticas HF"] * 7,
        "preco_unitario": [45000.00, 38000.00, 92000.00, 125000.00,
                           51000.00, 8500.00, 22000.00],
        "data_cadastro":  pd.to_datetime(["2021-03-15", "2021-07-20", "2022-01-10",
                                          "2022-04-05", "2022-09-30", "2023-02-14",
                                          "2023-06-01"]),
    })

    # Unidades fictícias
    df_uni = pd.DataFrame({
        "id_unidade":  range(1, 6),
        "nome_unidade": ["Base Aérea de Brasília", "Base Aérea de Manaus",
                         "Ala 5 Campo Grande", "CINDACTA I", "DECEA"],
        "sigla":       ["BABS", "BAMN", "ALA5", "CINDACTA1", "DECEA"],
        "cidade":      ["Brasília", "Manaus", "Campo Grande", "Brasília", "Rio de Janeiro"],
        "estado":      ["DF", "AM", "MS", "DF", "RJ"],
        "latitude":    [-15.869, -3.038, -20.469, -15.728, -22.910],
        "longitude":   [-47.918, -60.049, -54.665, -47.919, -43.172],
    })

    # Movimentações fictícias
    df_mov = pd.DataFrame({
        "id_mov":            range(1, 11),
        "id_material":       [1, 3, 2, 5, 1, 4, 6, 2, 7, 3],
        "id_unidade_orig":   [1, 1, 2, 3, 4, 1, 2, 5, 3, 4],
        "id_unidade_dest":   [2, 3, 4, 5, 2, 3, 1, 4, 1, 5],
        "id_usuario":        [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
        "quantidade":        [2, 1, 3, 1, 5, 1, 2, 4, 1, 2],
        "valor_total":       [90000, 92000, 114000, 51000, 225000,
                              125000, 17000, 152000, 22000, 184000],
        "data_movimentacao": pd.to_datetime(["2023-01-10", "2023-02-15", "2023-03-20",
                                             "2023-04-05", "2023-05-12", "2023-06-18",
                                             "2023-07-22", "2023-08-30", "2023-09-14",
                                             "2023-10-01"]),
        "prazo_entrega":     pd.to_datetime(["2023-01-25", "2023-03-01", "2023-04-10",
                                             "2023-04-20", "2023-05-30", "2023-07-05",
                                             "2023-08-10", "2023-09-15", "2023-09-30",
                                             "2023-10-20"]),
        "status":            ["CONCLUIDO", "CONCLUIDO", "EM_TRANSITO", "CONCLUIDO",
                              "CONCLUIDO", "PENDENTE", "CONCLUIDO", "EM_TRANSITO",
                              "CONCLUIDO", "PENDENTE"],
        "observacoes":       ["Entrega urgente para operação"] * 10,
    })

    tabelas_demo = {
        "tb_fabricantes": (df_fab, TABELAS_E_COLUNAS.get("tb_fabricantes", {})),
        "tb_materiais":   (df_mat, TABELAS_E_COLUNAS.get("tb_materiais", {})),
        "tb_unidades":    (df_uni, TABELAS_E_COLUNAS.get("tb_unidades", {})),
        "tb_movimentacoes": (df_mov, TABELAS_E_COLUNAS.get("tb_movimentacoes", {})),
    }

    print("  ORIGINAL → MASCARADO")
    print("  " + "-" * 45)

    for tabela, (df, config) in tabelas_demo.items():
        df_masked = mask_dataframe(
            df, config, md, rng,
            date_offset=DATE_OFFSET_DAYS,
            noise=VALOR_NOISE_PERCENT,
            scale=VALOR_SCALE_FACTOR,
        )
        path = PASTA_SAIDA / f"{tabela}.csv"
        df_masked.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n  [{tabela}]")

        # Mostra preview de colunas mascaradas
        for col, tipo in config.items():
            if tipo == "manter" or col not in df.columns:
                continue
            orig_sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
            mask_sample = df_masked[col].dropna().iloc[0] if not df_masked[col].dropna().empty else "N/A"
            print(f"    {col} ({tipo}): '{orig_sample}' → '{mask_sample}'")

    md.save(PASTA_SAIDA / "mapping_dictionary.json")

    print(f"\n  [✓] Demo concluído! Arquivos em: {PASTA_SAIDA.resolve()}/")
    print(f"  [✓] Dicionário de mapeamento: mapping_dictionary.json")
    print(f"\n  ► Para usar com banco real: ajuste DATABASE_URL no topo do arquivo.\n")


if __name__ == "__main__":
    run_pipeline()