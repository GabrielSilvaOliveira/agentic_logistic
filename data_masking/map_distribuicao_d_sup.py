"""
Mascara colunas de distribuicao_d_sup.csv reaproveitando dicionarios existentes.

Mapeamentos aplicados:
  - mapping_dictionary_excel.json:
      invoice -> invoice
      contrato -> contrato
      item -> equipamento
  - mapping_dictionary.json:
      pn -> part_number
      nomenclatura_sigelog -> equipamento
      om_destino -> sigla
      rm_de_destino -> regiao
      subsistema -> subsistema
      tipoeqp -> tipo_equipamento

Uso:
  python data_masking/map_distribuicao_d_sup.py
  python data_masking/map_distribuicao_d_sup.py --input "caminho/do/distribuicao_d_sup.csv"
  python data_masking/map_distribuicao_d_sup.py --output "caminho/do/saida.csv"
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "dataset_mascarado" / "distribuicao_d_sup.csv"
DEFAULT_MAPPING_JSON = BASE_DIR / "dataset_mascarado" / "mapping_dictionary.json"
DEFAULT_MAPPING_EXCEL_JSON = BASE_DIR / "dataset_mascarado" / "mapping_dictionary_excel.json"
DEFAULT_CONSOLIDATED_MASK_OUTPUT = BASE_DIR / "dataset_mascarado" / "distribuicao_d_sup_mask.csv"


MAPPING_PLAN = [
    ("invoice", "excel", "invoice"),
    ("contrato", "excel", "ID_CONTRATO_ANO"),
    ("acao", "excel", "funds"),
    ("item", "excel", "equipamento"),
    ("pn", "json", "part_number"),
    ("nomenclatura_sigelog", "json", "equipamento"),
    ("om_destino", "json", "sigla"),
    ("rm_de_destino", "json", "regiao"),
    ("subsistema", "json", "subsistema"),
    ("tipoeqp", "json", "tipo_equipamento"),
]


SIMILARITY_RULES = {
    "invoice": {"threshold": 0.85, "inclusive": False},
    "contrato": {"threshold": 0.80, "inclusive": False},
    "acao": {"threshold": 0.80, "inclusive": False},
    "item": {"threshold": 0.80, "inclusive": False},
    "pn": {"threshold": 0.87, "inclusive": False},
    "nomenclatura_sigelog": {"threshold": 0.80, "inclusive": False},
    "om_destino": {"threshold": 0.92, "inclusive": True},
    "subsistema": {"threshold": 0.80, "inclusive": False},
    "tipoeqp": {"threshold": 0.80, "inclusive": False},
}


AUTO_CODE_CONFIG = {
    "invoice": {
        "source_name": "excel",
        "category": "invoice",
        "prefix": "INV",
        "digits": 3,
        "suffix_from_value": False,
    },
    "contrato": {
        "source_name": "excel",
        "category": "ID_CONTRATO_ANO",
        "prefix": "CTR",
        "digits": 3,
        "suffix_from_value": True,
    },
    "acao": {
        "source_name": "excel",
        "category": "funds",
        "prefix": "FND",
        "digits": 3,
        "suffix_from_value": False,
    },
    "pn": {
        "source_name": "json",
        "category": "part_number",
        "prefix": "PN",
        "digits": 5,
        "suffix_from_value": False,
    },
}


def normalize_text(value: str) -> str:
    """Normaliza texto para melhorar taxa de mapeamento sem mudar semantica."""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def build_lookup(mapping: dict) -> dict[str, str]:
    """Cria indice de busca com chave original e normalizada."""
    lookup: dict[str, str] = {}
    for key, masked in mapping.items():
        if key is None:
            continue
        key_str = str(key).strip()
        if not key_str:
            continue

        normalized = normalize_text(key_str)
        if key_str not in lookup:
            lookup[key_str] = str(masked)
        if normalized not in lookup:
            lookup[normalized] = str(masked)
    return lookup


def is_empty(value) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def map_cell(value, lookup: dict[str, str]):
    if is_empty(value):
        return value, False, False

    raw = str(value).strip()
    direct = lookup.get(raw)
    if direct is not None:
        return direct, True, False

    norm = normalize_text(raw)
    by_norm = lookup.get(norm)
    if by_norm is not None:
        return by_norm, True, False

    return value, False, True


def passes_threshold(score: float, threshold: float, inclusive: bool) -> bool:
    if inclusive:
        return score >= threshold
    return score > threshold


def find_best_similarity_match(value: str, reference_entries: list[tuple[str, str, str]]):
    query = normalize_text(value)
    if not query or not reference_entries:
        return "", "", 0.0

    best_original = ""
    best_masked = ""
    best_score = 0.0

    for ref_original, ref_norm, ref_masked in reference_entries:
        score = SequenceMatcher(None, query, ref_norm).ratio()
        if score > best_score:
            best_score = score
            best_original = ref_original
            best_masked = ref_masked

    return best_original, best_masked, best_score


def extract_contract_year_suffix(value: str) -> str:
    match = re.search(r"-(\d{2})$", str(value).strip())
    if match:
        return f"-{match.group(1)}"
    return ""


def next_counter_from_mapping_values(
    mapping_values: list[str],
    prefix: str,
    digits: int,
    has_optional_suffix: bool,
) -> int:
    suffix_pattern = r"(?:-.+)?" if has_optional_suffix else ""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{{digits}}}){suffix_pattern}$")

    last_counter = 0
    for code in mapping_values:
        match = pattern.match(str(code).strip())
        if match:
            last_counter = int(match.group(1))

    if last_counter > 0:
        return last_counter + 1

    max_counter = 0
    for code in mapping_values:
        match = pattern.match(str(code).strip())
        if match:
            max_counter = max(max_counter, int(match.group(1)))
    return max_counter + 1


def build_auto_code_generator(
    category_map: dict,
    prefix: str,
    digits: int,
    suffix_from_value: bool,
):
    counter = next_counter_from_mapping_values(
        list(category_map.values()),
        prefix=prefix,
        digits=digits,
        has_optional_suffix=suffix_from_value,
    )
    generated_by_raw: dict[str, str] = {}

    def generate(raw_value: str) -> str:
        nonlocal counter
        raw_key = str(raw_value).strip()
        if raw_key in generated_by_raw:
            return generated_by_raw[raw_key]

        suffix = extract_contract_year_suffix(raw_key) if suffix_from_value else ""
        code = f"{prefix}{str(counter).zfill(digits)}{suffix}"
        counter += 1
        generated_by_raw[raw_key] = code
        return code

    return generate


def mask_column(
    df: pd.DataFrame,
    column: str,
    lookup: dict[str, str],
    reference_entries: list[tuple[str, str, str]],
    similarity_threshold: float,
    similarity_inclusive: bool,
    context_series: pd.Series | None = None,
    auto_code_generator=None,
):
    mapped_count = 0
    similarity_mapped_count = 0
    auto_code_mapped_count = 0
    unmapped_count = 0
    non_empty_count = 0

    unmapped_values: dict[str, int] = {}
    unmapped_contexts: dict[str, dict[str, int]] = {}
    similarity_hits: dict[str, dict] = {}
    new_values = []
    similarity_cache: dict[str, tuple[str, str, float, bool]] = {}

    for idx, value in enumerate(df[column]):
        new_value, mapped, unmapped = map_cell(value, lookup)
        new_values.append(new_value)

        if not is_empty(value):
            non_empty_count += 1
        if mapped:
            mapped_count += 1
            continue

        if unmapped:
            raw = str(value).strip()

            if raw not in similarity_cache:
                best_original, best_masked, best_score = find_best_similarity_match(raw, reference_entries)
                accept = bool(best_masked) and passes_threshold(
                    best_score, similarity_threshold, similarity_inclusive
                )
                similarity_cache[raw] = (best_original, best_masked, best_score, accept)

            best_original, best_masked, best_score, accept = similarity_cache[raw]
            if accept:
                new_values[-1] = best_masked
                mapped_count += 1
                similarity_mapped_count += 1

                if raw not in similarity_hits:
                    similarity_hits[raw] = {
                        "valor_original": raw,
                        "valor_referencia": best_original,
                        "valor_mascarado_aplicado": best_masked,
                        "similaridade": round(best_score, 4),
                        "ocorrencias": 0,
                    }
                similarity_hits[raw]["ocorrencias"] += 1
                continue

            if auto_code_generator is not None:
                generated_code = auto_code_generator(raw)
                new_values[-1] = generated_code

                lookup[raw] = generated_code
                lookup[normalize_text(raw)] = generated_code

                mapped_count += 1
                auto_code_mapped_count += 1
                continue

            unmapped_count += 1
            key = raw
            unmapped_values[key] = unmapped_values.get(key, 0) + 1

            if context_series is not None:
                context_value = ""
                if idx < len(context_series):
                    context_raw = context_series.iloc[idx]
                    if not is_empty(context_raw):
                        context_value = str(context_raw).strip()

                context_bucket = unmapped_contexts.setdefault(key, {})
                context_bucket[context_value] = context_bucket.get(context_value, 0) + 1

    df[column] = new_values
    return {
        "column": column,
        "non_empty": non_empty_count,
        "mapped": mapped_count,
        "similarity_mapped": similarity_mapped_count,
        "auto_code_mapped": auto_code_mapped_count,
        "unmapped": unmapped_count,
        "mapped_pct": round((mapped_count / non_empty_count) * 100, 2) if non_empty_count else 0.0,
        "unmapped_values": unmapped_values,
        "unmapped_contexts": unmapped_contexts,
        "similarity_hits": list(similarity_hits.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mascara distribuicao_d_sup.csv usando mapping_dictionary.json e mapping_dictionary_excel.json"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV de entrada")
    parser.add_argument("--output", type=Path, default=None, help="CSV de saida mascarado")
    parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON, help="mapping_dictionary.json")
    parser.add_argument(
        "--mapping-excel-json",
        type=Path,
        default=DEFAULT_MAPPING_EXCEL_JSON,
        help="mapping_dictionary_excel.json",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.92,
        help="Limiar de similaridade para sugestoes nos nao mapeados (padrao: 0.92)",
    )
    parser.add_argument(
        "--max-suggestions",
        type=int,
        default=3,
        help="Numero maximo de sugestoes por valor nao mapeado (padrao: 3)",
    )
    parser.add_argument(
        "--consolidated-output",
        type=Path,
        default=DEFAULT_CONSOLIDATED_MASK_OUTPUT,
        help="CSV consolidado final com valores mascarados e nao mapeados preservados",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_reference_entries(category_map: dict) -> list[tuple[str, str, str]]:
    """Retorna lista (valor_original, valor_normalizado, valor_mascarado)."""
    entries: list[tuple[str, str, str]] = []
    for raw_key, masked_value in category_map.items():
        if raw_key is None:
            continue

        original = str(raw_key).strip()
        if not original:
            continue

        normalized = normalize_text(original)
        entries.append((original, normalized, str(masked_value)))
    return entries


def get_similarity_suggestions(
    value: str,
    reference_entries: list[tuple[str, str, str]],
    threshold: float,
    inclusive: bool,
    max_suggestions: int,
) -> tuple[str, float, list[tuple[str, float, str]]]:
    """
    Retorna (melhor_valor, melhor_score, sugestoes_acima_limiar).
    Nao altera mapeamento, apenas calcula proximidade para revisao humana.
    """
    query = normalize_text(value)
    if not query or not reference_entries:
        return "", 0.0, []

    best_value = ""
    best_score = 0.0
    candidates: list[tuple[str, float, str]] = []

    for ref_original, ref_norm, ref_masked in reference_entries:
        score = SequenceMatcher(None, query, ref_norm).ratio()
        if score > best_score:
            best_score = score
            best_value = ref_original

        if passes_threshold(score, threshold, inclusive):
            candidates.append((ref_original, score, ref_masked))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return best_value, round(best_score, 4), candidates[:max_suggestions]


def main():
    args = parse_args()

    if args.similarity_threshold < 0 or args.similarity_threshold > 1:
        raise ValueError("--similarity-threshold deve estar entre 0 e 1")
    if args.max_suggestions < 1:
        raise ValueError("--max-suggestions deve ser maior ou igual a 1")

    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        output_path = input_path.with_name(f"{input_path.stem}_mascarado_referenciado{input_path.suffix}")
    consolidated_output_path = args.consolidated_output

    report_path = output_path.with_name(f"{output_path.stem}_relatorio_mapeamento.csv")
    unmapped_path = output_path.with_name(f"{output_path.stem}_nao_mapeados.csv")
    similarity_map_path = output_path.with_name(f"{output_path.stem}_mapeados_por_similaridade.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"CSV de entrada nao encontrado: {input_path}")
    if not args.mapping_json.exists():
        raise FileNotFoundError(f"mapping_dictionary.json nao encontrado: {args.mapping_json}")
    if not args.mapping_excel_json.exists():
        raise FileNotFoundError(f"mapping_dictionary_excel.json nao encontrado: {args.mapping_excel_json}")

    print(f"Lendo CSV: {input_path}")
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    original_df = df.copy(deep=True)

    mapping_json = load_json(args.mapping_json)
    mapping_excel = load_json(args.mapping_excel_json)
    mapping_json_changed = False
    mapping_excel_changed = False

    reports = []
    unmapped_rows = []
    similarity_rows = []
    missing_columns = []
    missing_categories = []

    for column, source_name, category in MAPPING_PLAN:
        source = mapping_excel if source_name == "excel" else mapping_json
        category_map = source.get(category)
        if not isinstance(category_map, dict):
            missing_categories.append((source_name, category))
            continue

        if column not in df.columns:
            missing_columns.append(column)
            continue

        rule = SIMILARITY_RULES.get(
            column,
            {"threshold": args.similarity_threshold, "inclusive": True},
        )
        threshold = float(rule["threshold"])
        inclusive = bool(rule["inclusive"])

        lookup = build_lookup(category_map)
        reference_entries = build_reference_entries(category_map)
        auto_code_generator = None
        auto_cfg = AUTO_CODE_CONFIG.get(column)
        if auto_cfg and auto_cfg["source_name"] == source_name and auto_cfg["category"] == category:
            auto_code_generator = build_auto_code_generator(
                category_map,
                prefix=auto_cfg["prefix"],
                digits=int(auto_cfg["digits"]),
                suffix_from_value=bool(auto_cfg["suffix_from_value"]),
            )
        context_series = None
        if column == "item" and "contrato" in original_df.columns:
            # Mantem o contrato original para facilitar validacao externa em contratos.xlsx.
            context_series = original_df["contrato"]

        result = mask_column(
            df,
            column,
            lookup,
            reference_entries,
            similarity_threshold=threshold,
            similarity_inclusive=inclusive,
            context_series=context_series,
            auto_code_generator=auto_code_generator,
        )

        # Persiste no dicionario da categoria os novos codigos gerados para reuso em futuras execucoes.
        if auto_code_generator is not None:
            for original_value, masked_value in zip(original_df[column], df[column]):
                if is_empty(original_value):
                    continue

                raw_original = str(original_value).strip()
                if raw_original in category_map:
                    continue

                if not is_empty(masked_value):
                    category_map[raw_original] = str(masked_value)

        if result["auto_code_mapped"] > 0:
            if source_name == "excel":
                mapping_excel_changed = True
            else:
                mapping_json_changed = True

        reports.append(
            {
                "coluna": result["column"],
                "dicionario": source_name,
                "categoria": category,
                "criterio_similaridade": f"{'>=' if inclusive else '>'}{threshold:.2f}",
                "registros_nao_vazios": result["non_empty"],
                "registros_mapeados": result["mapped"],
                "mapeados_por_similaridade": result["similarity_mapped"],
                "mapeados_por_codigo": result["auto_code_mapped"],
                "registros_nao_mapeados": result["unmapped"],
                "percentual_mapeado": result["mapped_pct"],
            }
        )

        for hit in sorted(result["similarity_hits"], key=lambda item: (-item["ocorrencias"], item["valor_original"])):
            similarity_rows.append(
                {
                    "coluna": column,
                    "dicionario": source_name,
                    "categoria": category,
                    "criterio_similaridade": f"{'>=' if inclusive else '>'}{threshold:.2f}",
                    "valor_original": hit["valor_original"],
                    "valor_referencia": hit["valor_referencia"],
                    "valor_mascarado_aplicado": hit["valor_mascarado_aplicado"],
                    "similaridade": hit["similaridade"],
                    "ocorrencias": hit["ocorrencias"],
                }
            )

        for original_value, count in sorted(
            result["unmapped_values"].items(), key=lambda item: (-item[1], item[0])
        ):
            best_value, best_score, near_matches = get_similarity_suggestions(
                original_value,
                reference_entries,
                threshold=threshold,
                inclusive=inclusive,
                max_suggestions=args.max_suggestions,
            )

            contratos_relacionados = ""
            if column == "item":
                context_counts = result.get("unmapped_contexts", {}).get(original_value, {})
                if context_counts:
                    contratos_relacionados = " | ".join(
                        f"{contrato or '[vazio]'} ({ocorrencias})"
                        for contrato, ocorrencias in sorted(
                            context_counts.items(), key=lambda item: (-item[1], item[0])
                        )
                    )

            unmapped_rows.append(
                {
                    "coluna": column,
                    "valor_original": original_value,
                    "ocorrencias": count,
                    "contratos_relacionados": contratos_relacionados,
                    "melhor_correspondencia": best_value,
                    "similaridade_melhor": best_score,
                    "sugestoes_acima_limiar": " | ".join(
                        f"{match} ({score:.4f}) => {masked}"
                        for match, score, masked in near_matches
                    ),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_output_path.parent.mkdir(parents=True, exist_ok=True)

    if mapping_excel_changed:
        save_json(args.mapping_excel_json, mapping_excel)

    if mapping_json_changed:
        save_json(args.mapping_json, mapping_json)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    # Consolidado final para consumo externo:
    # valores mapeados (direto/similaridade/codigo) e nao mapeados sem mascara.
    df.to_csv(consolidated_output_path, index=False, encoding="utf-8-sig")

    report_df = pd.DataFrame(reports)
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    similarity_df = pd.DataFrame(similarity_rows)
    if not similarity_df.empty:
        similarity_df.to_csv(similarity_map_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            columns=[
                "coluna",
                "dicionario",
                "categoria",
                "criterio_similaridade",
                "valor_original",
                "valor_referencia",
                "valor_mascarado_aplicado",
                "similaridade",
                "ocorrencias",
            ]
        ).to_csv(similarity_map_path, index=False, encoding="utf-8-sig")

    unmapped_df = pd.DataFrame(unmapped_rows)
    if not unmapped_df.empty:
        unmapped_df.to_csv(unmapped_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            columns=[
                "coluna",
                "valor_original",
                "ocorrencias",
                "contratos_relacionados",
                "melhor_correspondencia",
                "similaridade_melhor",
                "sugestoes_acima_limiar",
            ]
        ).to_csv(
            unmapped_path, index=False, encoding="utf-8-sig"
        )

    print("\nResumo por coluna:")
    if reports:
        for row in reports:
            print(
                "  - {coluna}: {registros_mapeados}/{registros_nao_vazios} mapeados "
                "({percentual_mapeado:.2f}%), nao mapeados={registros_nao_mapeados}".format(**row)
            )
    else:
        print("  Nenhuma coluna foi processada.")

    if missing_columns:
        print("\n[!] Colunas nao encontradas no CSV:")
        for col in missing_columns:
            print(f"  - {col}")

    if missing_categories:
        print("\n[!] Categorias nao encontradas nos dicionarios:")
        for source_name, category in missing_categories:
            print(f"  - {source_name}:{category}")

    if mapping_excel_changed:
        print(f"[✓] Novos pares salvos em: {args.mapping_excel_json}")
    if mapping_json_changed:
        print(f"[✓] Novos pares salvos em: {args.mapping_json}")

    print(f"\nCSV mascarado salvo em: {output_path}")
    print(f"CSV consolidado (mask) salvo em: {consolidated_output_path}")
    print(f"Relatorio de mapeamento: {report_path}")
    print(f"Mapeados por similaridade: {similarity_map_path}")
    print(f"Valores nao mapeados: {unmapped_path}")
    print(
        "Sugestoes de similaridade: "
        f"limiar_base_nao_mapeados={args.similarity_threshold:.2f}, max_por_valor={args.max_suggestions}"
    )


if __name__ == "__main__":
    main()