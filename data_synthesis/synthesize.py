import os
import pandas as pd
from sqlalchemy import create_engine
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


INPUT_SPREADSHEET_DIR = "input/spreadsheets"
OUTPUT_SPREADSHEET_DIR = "output/spreadsheets"

INPUT_DB = "input/database/sg7.db"
OUTPUT_DB = "output/database/sg7_synthetic.db"

os.makedirs(OUTPUT_SPREADSHEET_DIR, exist_ok=True)
os.makedirs("output/database", exist_ok=True)
os.makedirs("output/api", exist_ok=True)


def synthesize_dataframe(df, rows=None):

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    synthesizer = GaussianCopulaSynthesizer(metadata)

    synthesizer.fit(df)

    if rows is None:
        rows = len(df)

    synthetic = synthesizer.sample(rows)

    return synthetic


def synthesize_spreadsheets():

    for file in os.listdir(INPUT_SPREADSHEET_DIR):

        if not file.endswith(".xlsx"):
            continue

        path = os.path.join(INPUT_SPREADSHEET_DIR, file)

        print(f"Synthesizing spreadsheet {file}")

        df = pd.read_excel(path)

        synthetic_df = synthesize_dataframe(df)

        output_path = os.path.join(
            OUTPUT_SPREADSHEET_DIR,
            file.replace(".xlsx", "_synthetic.xlsx")
        )

        synthetic_df.to_excel(output_path, index=False)


def synthesize_database():

    engine_in = create_engine(f"sqlite:///{INPUT_DB}")
    engine_out = create_engine(f"sqlite:///{OUTPUT_DB}")

    tables = engine_in.table_names()

    for table in tables:

        print(f"Synthesizing table {table}")

        df = pd.read_sql_table(table, engine_in)

        synthetic_df = synthesize_dataframe(df)

        synthetic_df.to_sql(
            table,
            engine_out,
            if_exists="replace",
            index=False
        )


def export_api_data():

    engine = create_engine(f"sqlite:///{OUTPUT_DB}")

    try:
        contratos = pd.read_sql("SELECT * FROM contratos", engine)

        contratos.to_json(
            "output/api/contratos.json",
            orient="records",
            indent=2
        )

    except:
        print("Tabela contratos não encontrada")


def main():

    print("Step 1: synthesizing spreadsheets")
    synthesize_spreadsheets()

    print("Step 2: synthesizing database")
    synthesize_database()

    print("Step 3: exporting API datasets")
    export_api_data()

    print("Synthetic dataset generated")


if __name__ == "__main__":
    main()