import pandas as pd
import os
from sqlalchemy import create_engine, text
from sdv.metadata import MultiTableMetadata
from sdv.metadata.errors import InvalidMetadataError
from sdv.multi_table import HMASynthesizer
from dotenv import load_dotenv

# --------------------------------------------------
# CONFIGURAÇÃO
# --------------------------------------------------
load_dotenv()

MYSQL_URI = os.getenv("MYSQL_URI")
OUTPUT_DB = os.getenv("OUTPUT_DB")

EXCLUDE_TABLES = [
    "sc_log",
    "sc_log_contratos",
    "sec_application",
    "sec_users",
    "sec_users_groups",
    "sec_groups",
    "sec_users_applications",
    "sec_groups_applications",
    "sec_apps",
    "sec_groups_apps",
    "sec_users_apps",
    "django_migrations",
    "django_session",
    "django_admin_log",
    "django_content_type",
    "auth_group",
    "auth_group_permissions",
    "auth_permission",
    "users_perfil",
    "users_perfil_user_permissions",
    "users_perfil_groups",
]

# --------------------------------------------------
# CONECTAR BANCO
# --------------------------------------------------

engine = create_engine(MYSQL_URI)


# --------------------------------------------------
# DESCOBRIR TABELAS
# --------------------------------------------------

def get_tables():

    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = DATABASE()
    AND table_type = 'BASE TABLE'
    AND table_name NOT IN ({})
    """.format(", ".join(f"'{table}'" for table in EXCLUDE_TABLES))

    tables = pd.read_sql(query, engine)
    print(tables)
    return tables["TABLE_NAME"].tolist()


# --------------------------------------------------
# DESCOBRIR PRIMARY KEYS
# --------------------------------------------------

def get_primary_keys():

    query = """
    SELECT
        table_name,
        column_name
    FROM information_schema.key_column_usage
    WHERE constraint_name = 'PRIMARY'
    AND table_schema = DATABASE()
    AND table_name NOT IN ({})
    """.format(", ".join(f"'{table}'" for table in EXCLUDE_TABLES))

    return pd.read_sql(query, engine)


# --------------------------------------------------
# DESCOBRIR FOREIGN KEYS
# --------------------------------------------------

def get_foreign_keys():

    query = """
    SELECT
        table_name,
        column_name,
        referenced_table_name,
        referenced_column_name
    FROM information_schema.key_column_usage
    WHERE referenced_table_name IS NOT NULL
    AND table_schema = DATABASE()
    AND table_name NOT IN ({})
    AND referenced_table_name NOT IN ({})
    """.format(
        ", ".join(f"'{table}'" for table in EXCLUDE_TABLES),
        ", ".join(f"'{table}'" for table in EXCLUDE_TABLES)
    )

    return pd.read_sql(query, engine)


# --------------------------------------------------
# CARREGAR DADOS
# --------------------------------------------------

def load_tables(table_names):

    data = {}

    for table in table_names:

        print(f"Loading table {table}")

        df = pd.read_sql(f"SELECT * FROM {table}", engine)

        data[table] = df

    return data


def remove_empty_tables(data, pk_df, fk_df):

    empty_tables = [table for table, df in data.items() if df.empty]

    if empty_tables:
        print("\n[INFO] Tabelas vazias removidas antes da sintese:")
        for table in empty_tables:
            print(f"  - {table}")

    filtered_data = {
        table: df
        for table, df in data.items()
        if not df.empty
    }

    included_tables = set(filtered_data.keys())

    filtered_pk_df = pk_df[
        pk_df["TABLE_NAME"].isin(included_tables)
    ].copy()

    filtered_fk_df = fk_df[
        fk_df["TABLE_NAME"].isin(included_tables)
        & fk_df["REFERENCED_TABLE_NAME"].isin(included_tables)
    ].copy()

    return filtered_data, filtered_pk_df, filtered_fk_df, empty_tables


# --------------------------------------------------
# GERAR METADATA AUTOMÁTICO
# --------------------------------------------------

def build_metadata(data, pk_df, fk_df):

    metadata = MultiTableMetadata()
    included_tables = set(data.keys())
    relationships_applied = 0
    relationships_invalid = 0

    # detectar estrutura de cada tabela
    for table, df in data.items():

        metadata.detect_table_from_dataframe(
            table_name=table,
            data=df
        )

    # debug: identificar exatamente relacoes envolvendo auth_group
    debug_table = "auth_group"
    fk_debug = fk_df[
        (fk_df["TABLE_NAME"] == debug_table)
        | (fk_df["REFERENCED_TABLE_NAME"] == debug_table)
    ]

    if not fk_debug.empty:
        print("\n[DEBUG] Relacionamentos envolvendo 'auth_group':")
        for _, row in fk_debug.iterrows():
            print(
                "  "
                f"{row['TABLE_NAME']}.{row['COLUMN_NAME']} -> "
                f"{row['REFERENCED_TABLE_NAME']}.{row['REFERENCED_COLUMN_NAME']}"
            )
    else:
        print("\n[DEBUG] Nenhum relacionamento envolvendo 'auth_group' foi encontrado.")

    # adicionar primary keys
    for _, row in pk_df.iterrows():
        table_name = row["TABLE_NAME"]
        column_name = row["COLUMN_NAME"]

        if table_name not in included_tables:
            print(
                "[SKIP PK] "
                f"{table_name}.{column_name} ignorada "
                "(tabela nao carregada no metadata)"
            )
            continue
        
        metadata.update_column(
            table_name=table_name,
            column_name=column_name,
            sdtype="id"
        )

        metadata.set_primary_key(
            table_name=table_name,
            column_name=column_name
        )

    # adicionar relacionamentos
    for _, row in fk_df.iterrows():
        child_table = row["TABLE_NAME"]
        parent_table = row["REFERENCED_TABLE_NAME"]
        child_foreign_key = row["COLUMN_NAME"]
        parent_primary_key = row["REFERENCED_COLUMN_NAME"]

        if child_table not in included_tables or parent_table not in included_tables:
            print(
                "[SKIP FK] "
                f"{child_table}.{child_foreign_key} -> "
                f"{parent_table}.{parent_primary_key} ignorada "
                "(child/parent fora das tabelas carregadas)"
            )
            continue

        try:
            metadata.add_relationship(
                parent_table_name=parent_table,
                child_table_name=child_table,
                parent_primary_key=parent_primary_key,
                child_foreign_key=child_foreign_key
            )
            relationships_applied += 1
        except InvalidMetadataError as err:
            relationships_invalid += 1
            print(
                "[SKIP FK INVALID] "
                f"{child_table}.{child_foreign_key} -> "
                f"{parent_table}.{parent_primary_key} ignorada: {err}"
            )

    relationship_stats = {
        "applied": relationships_applied,
        "invalid_or_cycle": relationships_invalid,
    }

    return metadata, relationship_stats


def normalize_datetime_columns(data, metadata):

    metadata_dict = metadata.to_dict()
    coerced_columns = []

    for table_name, df in data.items():

        table_meta = metadata_dict.get("tables", {}).get(table_name, {})
        columns_meta = table_meta.get("columns", {})

        for column_name, column_meta in columns_meta.items():

            if column_meta.get("sdtype") != "datetime":
                continue

            if column_name not in df.columns:
                continue

            datetime_format = column_meta.get("datetime_format")

            # Converte strings vazias para nulo e faz parse tolerante.
            cleaned = df[column_name].replace("", pd.NA)

            if datetime_format:
                parsed = pd.to_datetime(cleaned, format=datetime_format, errors="coerce")
            else:
                parsed = pd.to_datetime(cleaned, errors="coerce")

            invalid_count = cleaned.notna().sum() - parsed.notna().sum()

            if invalid_count > 0:
                print(
                    "[DATETIME COERCE] "
                    f"{table_name}.{column_name} possui {invalid_count} valores invalidos "
                    "convertidos para nulo"
                )
                coerced_columns.append(
                    {
                        "table": table_name,
                        "column": column_name,
                        "invalid_count": int(invalid_count),
                    }
                )

            data[table_name][column_name] = parsed

    return data, coerced_columns


def print_final_report(empty_tables, relationship_stats, coerced_columns):

    print("\n=== RELATORIO FINAL ===")
    print(f"Tabelas removidas por estarem vazias: {len(empty_tables)}")
    print(f"Relacionamentos aplicados: {relationship_stats['applied']}")
    print(
        "Relacionamentos ignorados por incompatibilidade/ciclo: "
        f"{relationship_stats['invalid_or_cycle']}"
    )

    print("Colunas datetime com coerção:")
    if not coerced_columns:
        print("  - Nenhuma")
    else:
        for item in coerced_columns:
            print(
                "  - "
                f"{item['table']}.{item['column']} "
                f"(valores invalidos convertidos: {item['invalid_count']})"
            )


# --------------------------------------------------
# SINTETIZAR
# --------------------------------------------------

def synthesize_database(data, metadata):

    if not data:
        raise ValueError("Nenhuma tabela com dados para sintetizar apos filtros.")

    synthesizer = HMASynthesizer(metadata)

    print("Training synthesizer...")
    synthesizer.fit(data)

    print("Generating synthetic dataset...")
    synthetic_data = synthesizer.sample()

    return synthetic_data


# --------------------------------------------------
# EXPORTAR
# --------------------------------------------------

def export_database(synthetic_data):

    engine_out = create_engine(OUTPUT_DB)

    for table, df in synthetic_data.items():

        print(f"Exporting {table}")

        df.to_sql(
            table,
            engine_out,
            if_exists="replace",
            index=False
        )


# --------------------------------------------------
# PIPELINE PRINCIPAL
# --------------------------------------------------

def main():

    print("Discovering tables...")

    tables = get_tables()

    print(f"{len(tables)} tables found")

    pk_df = get_primary_keys()
    fk_df = get_foreign_keys()

    print("Loading data...")

    data = load_tables(tables)
    data, pk_df, fk_df, empty_tables = remove_empty_tables(data, pk_df, fk_df)

    print(f"{len(data)} tabelas com dados apos filtro de tabelas vazias")

    if not data:
        raise ValueError("Todas as tabelas carregadas estao vazias. Nada para sintetizar.")

    print("Building metadata...")

    metadata, relationship_stats = build_metadata(data, pk_df, fk_df)

    os.makedirs("output", exist_ok=True)
    metadata_path = "output/metadata_snapshot.json"
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    metadata.save_to_json(metadata_path)
    print("Metadata salva em output/metadata_snapshot.json")

    print("Normalizing datetime columns...")
    data, coerced_columns = normalize_datetime_columns(data, metadata)

    print("Synthesizing dataset...")

    synthetic_data = synthesize_database(data, metadata)

    print("Exporting synthetic database...")

    export_database(synthetic_data)

    print_final_report(empty_tables, relationship_stats, coerced_columns)

    print("DONE")


if __name__ == "__main__":
    main()