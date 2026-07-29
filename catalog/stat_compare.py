#!/usr/bin/env python3
"""
stat_compare.py
────────────────────────────────────────────────────────────────────────
Testes de significância estatística (bootstrap pareado) com correção de
Holm-Bonferroni para comparação entre estratégias de seleção de fontes
do Semantic Context Builder (SCB).

Consome diretamente os arquivos results_<modelo>.json gerados por
run_benchmark.py / run_llm_benchmark.py (schema: chave "all_results",
lista de dicts com id, difficulty, cluster, f1, precision, recall,
top1_acc, error_rate, ndcg por consulta).

Três modos de uso
─────────────────
  1) ranking   — compara todas as estratégias dentro de UM diretório
                 (ex.: desempate final entre embeddings + SLM + LLM na v3)

  2) ablation  — compara a MESMA estratégia entre múltiplos diretórios
                 (ex.: v1 vs v2 vs v3 do catálogo, por modelo)

  3) pairs     — comparações específicas definidas em um arquivo de
                 configuração JSON (ex.: reranker on/off, GPT-4o
                 comprimido vs completo, BM25 vs melhor embedding)

Cada modo aplica a correção de Holm-Bonferroni DENTRO da família de
testes que ele mesmo gera — famílias diferentes nunca são misturadas
na mesma correção.

Exemplos de uso
───────────────
  # Desempate final entre todas as estratégias avaliadas na v3
  python stat_compare.py ranking --dir ./results_v3 --metric f1 \
      --output tabela_ranking_v3.csv

  # Cada estratégia comparada apenas contra a de melhor F1 médio
  python stat_compare.py ranking --dir ./results_v3 --vs-best \
      --output tabela_vs_best.csv

  # Ablação de catálogo por modelo (v1 → v2 → v3)
  python stat_compare.py ablation \
      --dir v1=./results_v1 --dir v2=./results_v2 --dir v3=./results_v3 \
      --metric f1 --output tabela_ablacao.csv

  # Comparações específicas (reranker, GPT-4o comp/full, SLM vs embedding)
  python stat_compare.py pairs --config comparisons.json \
      --output tabela_pares.csv

  # Testar só dentro de um subgrupo (ex.: consultas de discriminação fina)
  python stat_compare.py ranking --dir ./results_v3 --difficulty discriminacao_fina

Requisitos
──────────
  pip install numpy statsmodels
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    sys.exit(
        "Pacote 'statsmodels' não encontrado. Instale com:\n"
        "  pip install statsmodels --break-system-packages"
    )


METRICS = ("f1", "precision", "recall", "top1_acc", "error_rate", "ndcg")


# ─────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────

def load_per_query_metric(
    results_path: Path,
    metric: str = "f1",
    difficulty: str | None = None,
    cluster: str | None = None,
) -> dict[str, float]:
    """
    Carrega o valor de `metric` por consulta (id -> valor) de um arquivo
    results_<modelo>.json. Filtros opcionais de difficulty/cluster permitem
    testar significância dentro de um subgrupo específico (ex.: apenas
    consultas de 'discriminacao_fina').
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {results_path}")

    data = json.loads(results_path.read_text(encoding="utf-8"))
    if "all_results" not in data:
        raise ValueError(
            f"{results_path} não contém a chave 'all_results'. Este script "
            f"espera o schema gerado por run_benchmark.py / run_llm_benchmark.py."
        )

    out: dict[str, float] = {}
    for r in data["all_results"]:
        if difficulty and r.get("difficulty") != difficulty:
            continue
        if cluster and r.get("cluster") != cluster:
            continue
        if metric not in r:
            raise KeyError(
                f"Métrica '{metric}' ausente em {results_path} (query {r.get('id')})"
            )
        out[r["id"]] = float(r[metric])

    if not out:
        raise ValueError(
            f"Nenhuma consulta restante em {results_path} após aplicar os "
            f"filtros difficulty={difficulty!r} cluster={cluster!r}."
        )
    return out


def model_label_from_filename(path: Path) -> str:
    """results_bge-m3.json -> bge-m3 ; results_azure_gpt-4o_compressed.json -> azure_gpt-4o_compressed"""
    return re.sub(r"^results_", "", path.stem)


def discover_results(directory: Path) -> dict[str, Path]:
    """Retorna {label_do_modelo: caminho} para todo results_*.json em um diretório."""
    if not directory.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {directory}")
    files = sorted(directory.glob("results_*.json"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo results_*.json encontrado em {directory}")
    return {model_label_from_filename(f): f for f in files}


# ─────────────────────────────────────────────────────────────────────────
# BOOTSTRAP PAREADO
# ─────────────────────────────────────────────────────────────────────────

def paired_bootstrap(
    values_a: dict[str, float],
    values_b: dict[str, float],
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap pareado sobre a diferença (a - b), casando por chave (id da
    consulta). Retorna diferença observada, p-valor bicaudal e IC 95%.
    """
    common = sorted(set(values_a) & set(values_b))
    if len(common) != len(values_a) or len(common) != len(values_b):
        missing_a = sorted(set(values_b) - set(values_a))
        missing_b = sorted(set(values_a) - set(values_b))
        raise ValueError(
            f"IDs de consulta não coincidem entre os dois conjuntos "
            f"(comum={len(common)}, a={len(values_a)}, b={len(values_b)}). "
            f"Faltando em A: {missing_a[:5]}{'...' if len(missing_a) > 5 else ''} "
            f"Faltando em B: {missing_b[:5]}{'...' if len(missing_b) > 5 else ''}"
        )
    if not common:
        raise ValueError("Nenhuma consulta em comum entre os dois conjuntos.")

    a = np.array([values_a[k] for k in common])
    b = np.array([values_b[k] for k in common])
    diffs = a - b
    observed = float(diffs.mean())

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.array(
        [rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)]
    )
    p_value = float(2 * min((boot_means >= 0).mean(), (boot_means <= 0).mean()))
    p_value = min(p_value, 1.0)
    ci_low, ci_high = (float(x) for x in np.percentile(boot_means, [2.5, 97.5]))

    return {
        "n_queries": n,
        "diff": observed,
        "p_raw": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ─────────────────────────────────────────────────────────────────────────
# CORREÇÃO DE HOLM-BONFERRONI POR FAMÍLIA
# ─────────────────────────────────────────────────────────────────────────

def apply_holm(rows: list[dict], alpha: float = 0.05) -> list[dict]:
    if not rows:
        return rows
    pvals = [r["p_raw"] for r in rows]
    reject, p_corr, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    for r, p_adj, rej in zip(rows, p_corr, reject):
        r["p_holm"] = float(p_adj)
        r["significativo_holm"] = bool(rej)
    return rows


# ─────────────────────────────────────────────────────────────────────────
# SAÍDA
# ─────────────────────────────────────────────────────────────────────────

def print_and_save(rows: list[dict], output: Path | None, metric: str):
    if not rows:
        print("[!] Nenhuma comparação para exibir.")
        return

    label = f"Δ{metric}"
    header = (
        f"{'Comparação':<48} {'n':>4} {label:>10} "
        f"{'IC 95%':>22} {'p_raw':>9} {'p_holm':>9}  sig."
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        sig = "✓" if r["significativo_holm"] else " "
        print(
            f"{r['comparacao']:<48} {r['n_queries']:>4} {r['diff']:>+10.4f} "
            f"{ci:>22} {r['p_raw']:>9.4f} {r['p_holm']:>9.4f}   {sig}"
        )
    print()

    if output:
        fieldnames = ["comparacao", "familia", "n_queries", "diff", "p_raw",
                      "p_holm", "ci_low", "ci_high", "significativo_holm"]
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"[✓] Tabela salva em: {output}")


# ─────────────────────────────────────────────────────────────────────────
# MODO 1 — RANKING (comparações dentro de um único diretório)
# ─────────────────────────────────────────────────────────────────────────

def cmd_ranking(args):
    directory = Path(args.dir)
    models = discover_results(directory)
    print(f"[i] {len(models)} estratégias encontradas em {directory}: {', '.join(sorted(models))}")

    metric_data = {
        name: load_per_query_metric(path, args.metric, args.difficulty, args.cluster)
        for name, path in models.items()
    }

    if args.vs_best:
        means = {name: float(np.mean(list(v.values()))) for name, v in metric_data.items()}
        best_name = max(means, key=means.get)
        pairs_to_test = [(name, best_name) for name in models if name != best_name]
        print(f"[i] Modo --vs-best: referência = '{best_name}' "
              f"(média {args.metric}={means[best_name]:.4f})")
    else:
        pairs_to_test = list(combinations(sorted(models), 2))

    rows = []
    for a, b in pairs_to_test:
        result = paired_bootstrap(metric_data[a], metric_data[b])
        rows.append({"comparacao": f"{a} vs {b}", "familia": f"ranking:{directory.name}", **result})

    rows = apply_holm(rows, alpha=args.alpha)
    print_and_save(rows, Path(args.output) if args.output else None, args.metric)


# ─────────────────────────────────────────────────────────────────────────
# MODO 2 — ABLATION (mesma estratégia entre diretórios/versões de catálogo)
# ─────────────────────────────────────────────────────────────────────────

def _parse_labeled_dirs(dir_args: list[str]) -> dict[str, Path]:
    """Converte ['v1=./results_v1', 'v2=./results_v2'] em {'v1': Path(...), ...}"""
    out = {}
    for item in dir_args:
        if "=" not in item:
            sys.exit(f"Formato inválido para --dir: '{item}'. Use rotulo=caminho, ex: v1=./results_v1")
        label, path = item.split("=", 1)
        out[label] = Path(path)
    return out


def cmd_ablation(args):
    labeled_dirs = _parse_labeled_dirs(args.dir)
    if len(labeled_dirs) < 2:
        sys.exit("Modo 'ablation' requer ao menos 2 diretórios rotulados "
                  "(ex.: --dir v1=./r1 --dir v2=./r2)")

    version_labels = list(labeled_dirs.keys())  # ordem de entrada = ordem de comparação
    models_by_version = {label: discover_results(path) for label, path in labeled_dirs.items()}

    # Só modelos presentes em TODAS as versões são comparáveis
    common_models = set.intersection(*[set(m) for m in models_by_version.values()])
    if not common_models:
        sys.exit("Nenhum modelo em comum entre todos os diretórios informados.")
    skipped = set.union(*[set(m) for m in models_by_version.values()]) - common_models
    if skipped:
        print(f"[!] Modelos ignorados por não existirem em todas as versões: {sorted(skipped)}")

    # Pares consecutivos (v1→v2, v2→v3, ...) + extremos (v1→v_ultima), sem duplicar
    consecutive_pairs = list(zip(version_labels[:-1], version_labels[1:]))
    extreme_pair = (version_labels[0], version_labels[-1])
    version_pairs = consecutive_pairs if extreme_pair in consecutive_pairs else consecutive_pairs + [extreme_pair]

    rows = []
    for model in sorted(common_models):
        for v_a, v_b in version_pairs:
            path_a = models_by_version[v_a][model]
            path_b = models_by_version[v_b][model]
            fa = load_per_query_metric(path_a, args.metric, args.difficulty, args.cluster)
            fb = load_per_query_metric(path_b, args.metric, args.difficulty, args.cluster)
            result = paired_bootstrap(fa, fb)
            rows.append({
                "comparacao": f"{model}: {v_a} vs {v_b}",
                "familia": "ablacao_catalogo",
                **result,
            })

    rows = apply_holm(rows, alpha=args.alpha)
    print_and_save(rows, Path(args.output) if args.output else None, args.metric)


# ─────────────────────────────────────────────────────────────────────────
# MODO 3 — PAIRS (comparações explícitas via config JSON)
# ─────────────────────────────────────────────────────────────────────────

def cmd_pairs(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    entries = config if isinstance(config, list) else config.get("comparisons", [])
    if not entries:
        sys.exit(f"Nenhuma comparação encontrada em {args.config}")

    families: dict[str, list[dict]] = {}
    for entry in entries:
        name = entry["nome"]
        familia = entry.get("familia", "geral")
        path_a = Path(entry["a"])
        path_b = Path(entry["b"])
        fa = load_per_query_metric(path_a, args.metric, args.difficulty, args.cluster)
        fb = load_per_query_metric(path_b, args.metric, args.difficulty, args.cluster)
        result = paired_bootstrap(fa, fb)
        families.setdefault(familia, []).append({"comparacao": name, "familia": familia, **result})

    # Correção de Holm aplicada DENTRO de cada família separadamente
    all_rows = []
    for familia, rows in families.items():
        print(f"\n### Família: {familia}")
        rows = apply_holm(rows, alpha=args.alpha)
        all_rows.extend(rows)

    print_and_save(all_rows, Path(args.output) if args.output else None, args.metric)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def _add_common_args(p: argparse.ArgumentParser):
    """Adiciona as opções compartilhadas a um subparser, para que funcionem
    quando digitadas DEPOIS do subcomando (uso natural na linha de comando)."""
    p.add_argument("--metric", default="f1", choices=METRICS,
                   help="Métrica a testar (padrão: f1)")
    p.add_argument("--difficulty", default=None,
                   help="Filtra apenas um nível de dificuldade (ex.: discriminacao_fina)")
    p.add_argument("--cluster", default=None,
                   help="Filtra apenas um cluster de domínio")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Nível de significância antes da correção (padrão: 0.05)")
    p.add_argument("--output", default=None, help="Caminho do CSV de saída (opcional)")


def main():
    parser = argparse.ArgumentParser(
        description="Testes de significância (bootstrap pareado + Holm-Bonferroni) "
                    "para os resultados do SCB."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_rank = sub.add_parser("ranking", help="Compara estratégias dentro de um diretório")
    p_rank.add_argument("--dir", required=True, help="Diretório com results_*.json")
    p_rank.add_argument("--vs-best", action="store_true",
                        help="Compara cada estratégia só contra a de melhor média "
                             "(em vez de todos-contra-todos)")
    _add_common_args(p_rank)
    p_rank.set_defaults(func=cmd_ranking)

    p_abl = sub.add_parser("ablation", help="Compara a mesma estratégia entre versões de catálogo")
    p_abl.add_argument("--dir", action="append", required=True,
                       help="rotulo=caminho, repetível (ex.: --dir v1=./r1 --dir v2=./r2 --dir v3=./r3)")
    _add_common_args(p_abl)
    p_abl.set_defaults(func=cmd_ablation)

    p_pairs = sub.add_parser("pairs", help="Comparações explícitas via arquivo de configuração JSON")
    p_pairs.add_argument("--config", required=True, help="Caminho do JSON de comparações")
    _add_common_args(p_pairs)
    p_pairs.set_defaults(func=cmd_pairs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()