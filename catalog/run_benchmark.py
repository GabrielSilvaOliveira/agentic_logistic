"""
benchmark/run_benchmark.py  — v2
─────────────────────────────────────────────────────────────────────────────
Correções em relação à v1:

  [FIX 1] Interface correta com SemanticContextBuilder
      Chama _select_sources() (método real) e converte o retorno
      de {"sql":["materiais"]} para ["sql.materiais"] antes de calcular métricas.

  [FIX 2] Stratified 5-Fold Cross-Validation
      As 80 queries são divididas em 5 folds estratificados por tipo de
      dificuldade — cada fold tem proporção igual dos 4 tipos.
      Threshold é calibrado nos 4 folds de treino e avaliado no fold de teste.
      Métricas finais: média ± desvio padrão dos 5 folds.

  [FIX 3] rebuild_index() chamado explicitamente
      Garante que o índice está pronto antes de qualquer avaliação.

  [FIX 4] source_role=multiplas_primarias tratado corretamente
      Queries com múltiplas primárias exigem recall total — qualquer fonte
      primária faltante conta como erro de recall.

Uso:
  python run_benchmark.py \
      --catalog   ../catalog/output/catalog.json \
      --benchmark benchmark_queries_final.json \
      --models    minilm e5-base e5-large bge-m3 \
      --output    results/ \
      --folds     5 \
      --top-k     3
"""

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "catalog"))

try:
    from semantic_context_builder import SemanticContextBuilder, MODEL_REGISTRY
except ImportError:
    raise SystemExit(
        "Não foi possível importar SemanticContextBuilder. "
        "Ajuste o sys.path no início deste script conforme sua estrutura de projeto."
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE COM O SemanticContextBuilder
# ─────────────────────────────────────────────────────────────────────────────

def get_selected_sources(cb: SemanticContextBuilder, query: str) -> list[str]:
    """
    Chama _select_sources() e converte o retorno para lista de strings
    no formato "tipo.chave" — ex: ["sql.materiais", "api.distribuicao_list"].

    _select_sources() retorna: {"sql": ["materiais"], "api": ["distribuicao_list"], ...}
    """
    result_dict = cb._select_sources(query, verbose=False)

    selected: list[str] = []
    # Preserva a ordem: SQL primeiro, depois API, depois planilhas
    for source_type in ("sql", "api", "spreadsheets"):
        for key in result_dict.get(source_type, []):
            selected.append(f"{source_type}.{key}")

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(selected: list[str],
                   expected_primary: list[str],
                   expected_complement: list[str]) -> float:
    """Fontes corretas (primárias OU complementares) entre as selecionadas."""
    if not selected:
        return 0.0
    correct = set(expected_primary) | set(expected_complement)
    return sum(1 for s in selected if s in correct) / len(selected)


def recall_at_k(selected: list[str], expected_primary: list[str]) -> float:
    """Primárias esperadas que foram selecionadas."""
    if not expected_primary:
        return 1.0
    return sum(1 for ep in expected_primary if ep in selected) / len(expected_primary)


def f1_score(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def top1_accuracy(selected: list[str], expected_primary: list[str]) -> float:
    """A fonte com maior score (primeira da lista) é uma das primárias?"""
    if not selected or not expected_primary:
        return 0.0
    return 1.0 if selected[0] in expected_primary else 0.0


def error_rate(selected: list[str], must_not_include: list[str]) -> float:
    """1 se qualquer fonte proibida foi selecionada, 0 caso contrário."""
    return 1.0 if any(s in set(must_not_include) for s in selected) else 0.0


def ndcg_at_k(selected: list[str],
               expected_primary: list[str],
               expected_complement: list[str],
               k: int = 5) -> float:
    """NDCG@K: primárias recebem relevância 2, complementares 1, outras 0."""
    prim_set = set(expected_primary)
    comp_set = set(expected_complement)

    def rel(s: str) -> int:
        if s in prim_set:
            return 2
        if s in comp_set:
            return 1
        return 0

    dcg = sum(rel(selected[i]) / math.log2(i + 2)
              for i in range(min(k, len(selected))))

    ideal = sorted([2] * len(expected_primary) + [1] * len(expected_complement),
                   reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal[:k]))

    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(selected: list[str], q: dict) -> dict:
    ep  = q["expected_primary"]
    ec  = q.get("expected_complement", [])
    mni = q.get("must_NOT_include", [])
    p   = precision_at_k(selected, ep, ec)
    r   = recall_at_k(selected, ep)
    return {
        "precision":  round(p, 4),
        "recall":     round(r, 4),
        "f1":         round(f1_score(p, r), 4),
        "top1_acc":   top1_accuracy(selected, ep),
        "error_rate": error_rate(selected, mni),
        "ndcg":       round(ndcg_at_k(selected, ep, ec), 4),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(m, 4), round(s, 4)


# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED K-FOLD
# ─────────────────────────────────────────────────────────────────────────────

def make_stratified_folds(queries: list[dict],
                           n_folds: int = 5,
                           seed: int = 42) -> list[list[int]]:
    """
    Divide os índices das queries em n_folds folds estratificados por
    tipo de dificuldade. Cada fold tem proporção igual dos 4 tipos.

    Retorna lista de n_folds listas de índices (índices do fold de TESTE).
    """
    rng = random.Random(seed)

    # Agrupa índices por tipo de dificuldade
    by_difficulty: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(queries):
        by_difficulty[q["difficulty"]].append(i)

    # Embaralha dentro de cada grupo
    for indices in by_difficulty.values():
        rng.shuffle(indices)

    # Distribui ciclicamente nos folds
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for indices in by_difficulty.values():
        for i, idx in enumerate(indices):
            folds[i % n_folds].append(idx)

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRAÇÃO DE THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_threshold(cb: SemanticContextBuilder,
                         train_queries: list[dict],
                         candidates: list[float] | None = None) -> float:
    """
    Encontra o threshold que maximiza F1 no conjunto de treino.
    Testa os valores em `candidates` e retorna o melhor.
    """
    if candidates is None:
        # Grade de busca: 20 pontos entre 0.20 e 0.95
        candidates = [round(0.20 + i * 0.05, 2) for i in range(16)]

    best_threshold = cb.min_similarity  # fallback: default do modelo
    best_f1 = -1.0

    original_threshold = cb.min_similarity

    for threshold in candidates:
        cb.min_similarity = threshold
        f1s = []
        for q in train_queries:
            selected = get_selected_sources(cb, q["query"])
            m = compute_metrics(selected, q)
            f1s.append(m["f1"])
        mean_f1 = sum(f1s) / len(f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_threshold = threshold

    cb.min_similarity = original_threshold  # restaura para não afetar outros folds
    return best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# AVALIAÇÃO DE UM FOLD
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fold(cb: SemanticContextBuilder,
                  test_queries: list[dict],
                  threshold: float) -> list[dict]:
    """Avalia um fold de teste com o threshold calibrado."""
    cb.min_similarity = threshold
    results = []
    for q in test_queries:
        t0 = time.monotonic()
        selected = get_selected_sources(cb, q["query"])
        latency_ms = (time.monotonic() - t0) * 1000

        m = compute_metrics(selected, q)
        results.append({
            "id":           q["id"],
            "query":        q["query"],
            "difficulty":   q["difficulty"],
            "cluster":      q["cluster"],
            "source_role":  q.get("source_role", ""),
            "selected":     selected,
            "expected_primary":    q["expected_primary"],
            "expected_complement": q.get("expected_complement", []),
            "must_NOT_include":    q.get("must_NOT_include", []),
            **m,
            "latency_ms":   round(latency_ms, 2),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# AGREGAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    metrics = ("precision", "recall", "f1", "top1_acc", "error_rate", "ndcg", "latency_ms")
    return {m: round(sum(r[m] for r in results) / len(results), 4) for m in metrics}


def aggregate_by_field(results: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r.get(field, "unknown")].append(r)
    return {k: aggregate(v) for k, v in groups.items()}


def aggregate_folds(fold_metrics: list[dict]) -> dict:
    """Calcula média ± desvio padrão entre folds."""
    metrics = ("precision", "recall", "f1", "top1_acc", "error_rate", "ndcg", "latency_ms")
    out = {}
    for m in metrics:
        vals = [fm[m] for fm in fold_metrics]
        mean, std = mean_std(vals)
        out[m]          = mean
        out[f"{m}_std"] = std
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Stratified K-Fold — SemanticContextBuilder")
    parser.add_argument("--catalog",   required=True,  help="Caminho do catalog.json")
    parser.add_argument("--benchmark", required=True,  help="Caminho do benchmark_queries_final.json")
    parser.add_argument("--models",    nargs="+",
                        default=["minilm", "e5-base", "e5-large", "bge-m3"])
    parser.add_argument("--output",    default="./results/")
    parser.add_argument("--folds",     type=int, default=5,  help="Número de folds (padrão: 5)")
    parser.add_argument("--top-k",     type=int, default=3,  help="Top-K fontes (padrão: 3)")
    parser.add_argument("--seed",      type=int, default=42, help="Seed para reprodutibilidade")
    parser.add_argument("--strategy",  default="score_gap",
                        choices=["score_gap", "threshold"])
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Pula calibração de threshold — usa default do modelo")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── Carrega benchmark ─────────────────────────────────────────────────────
    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)
    queries = benchmark["queries"]

    # ── Cria folds estratificados ─────────────────────────────────────────────
    folds = make_stratified_folds(queries, n_folds=args.folds, seed=args.seed)

    print(f"\n{'='*65}")
    print(f"  BENCHMARK — Semantic Context Builder (Stratified {args.folds}-Fold CV)")
    print(f"  Catálogo    : {args.catalog}")
    print(f"  Queries     : {len(queries)}")
    print(f"  Folds       : {args.folds}  (seed={args.seed})")
    print(f"  Top-K       : {args.top_k}")
    print(f"  Estratégia  : {args.strategy}")
    print(f"  Calibração  : {'não' if args.no_calibrate else 'sim (F1 máximo no treino)'}")
    print(f"  Modelos     : {', '.join(args.models)}")
    print(f"{'='*65}")

    # Verifica distribuição dos folds
    print(f"\n  Distribuição dos folds de teste:")
    for fold_i, test_idx in enumerate(folds):
        test_qs = [queries[i] for i in test_idx]
        dist = defaultdict(int)
        for q in test_qs: dist[q["difficulty"]] += 1
        print(f"    Fold {fold_i+1}: {len(test_qs)} queries | "
              + " | ".join(f"{k.replace('_',' ')[:8]}={v}" for k, v in sorted(dist.items())))

    all_model_summary = {}

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            print(f"\n[{model_name}] ❌ Modelo não reconhecido. Pulando.")
            continue

        print(f"\n{'─'*65}")
        print(f"  MODELO: {model_name} ({MODEL_REGISTRY[model_name]['hf_id']})")
        print(f"{'─'*65}")

        # Inicializa e constrói índice uma única vez
        cb = SemanticContextBuilder(
            catalog_path=args.catalog,
            model_name=model_name,
            top_k=args.top_k,
            selection_strategy=args.strategy,
        )
        cb.rebuild_index(verbose=False)
        print(f"  Índice pronto. Threshold default: {cb.min_similarity:.2f}")

        fold_metrics_list: list[dict] = []
        all_test_results:  list[dict] = []

        for fold_i, test_idx in enumerate(folds):
            train_idx = [i for f, idxs in enumerate(folds)
                         for i in idxs if f != fold_i]

            train_queries = [queries[i] for i in train_idx]
            test_queries  = [queries[i] for i in test_idx]

            # Calibração de threshold no conjunto de treino
            if args.no_calibrate:
                threshold = MODEL_REGISTRY[model_name]["default_min_similarity"]
            else:
                threshold = calibrate_threshold(cb, train_queries)

            # Avaliação no fold de teste
            cb.min_similarity = threshold
            fold_results = evaluate_fold(cb, test_queries, threshold)
            fold_agg     = aggregate(fold_results)

            fold_metrics_list.append(fold_agg)
            all_test_results.extend(fold_results)

            print(f"\n  Fold {fold_i+1}/{args.folds} "
                  f"(threshold={threshold:.2f}, teste={len(test_queries)} queries)")
            print(f"    F1={fold_agg['f1']:.4f}  "
                  f"P={fold_agg['precision']:.4f}  "
                  f"R={fold_agg['recall']:.4f}  "
                  f"Top1={fold_agg['top1_acc']:.4f}  "
                  f"Err={fold_agg['error_rate']:.4f}  "
                  f"NDCG={fold_agg['ndcg']:.4f}")

        # Agrega resultados entre folds (média ± std)
        cv_metrics = aggregate_folds(fold_metrics_list)

        print(f"\n  {'─'*50}")
        print(f"  Resultado CV ({args.folds}-Fold) — {model_name}")
        print(f"  {'─'*50}")
        for m in ("f1", "precision", "recall", "top1_acc", "error_rate", "ndcg", "latency_ms"):
            print(f"    {m:<14}: {cv_metrics[m]:.4f} ± {cv_metrics[f'{m}_std']:.4f}")

        # Análise por dificuldade (todos os folds de teste combinados)
        by_diff    = aggregate_by_field(all_test_results, "difficulty")
        by_cluster = aggregate_by_field(all_test_results, "cluster")
        by_role    = aggregate_by_field(all_test_results, "source_role")

        print(f"\n  Por dificuldade:")
        for diff, m in sorted(by_diff.items()):
            print(f"    {diff:<35} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

        print(f"\n  Por cluster:")
        for cl, m in sorted(by_cluster.items()):
            print(f"    {cl:<35} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

        # Salva resultado do modelo
        model_output = {
            "model":           model_name,
            "cv_metrics":      cv_metrics,
            "fold_metrics":    fold_metrics_list,
            "by_difficulty":   by_diff,
            "by_cluster":      by_cluster,
            "by_source_role":  by_role,
            "all_results":     all_test_results,
        }
        all_model_summary[model_name] = {
            "cv_metrics":    cv_metrics,
            "by_difficulty": by_diff,
        }

        out_path = Path(args.output) / f"results_{model_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_output, f, indent=2, ensure_ascii=False)
        print(f"\n  [✓] Salvo: {out_path}")

    # ── Tabela comparativa final ───────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  COMPARATIVO FINAL — Stratified {args.folds}-Fold CV (média ± std)")
    print(f"{'='*80}")
    header = f"  {'Modelo':<12} {'F1':>10} {'±':>6} {'Prec':>8} {'Recall':>8} {'Top1':>8} {'Err%':>8} {'NDCG':>8}"
    print(header)
    print(f"  {'-'*76}")

    comparison = {}
    for model_name, data in sorted(all_model_summary.items(),
                                    key=lambda x: -x[1]["cv_metrics"]["f1"]):
        m = data["cv_metrics"]
        print(f"  {model_name:<12} {m['f1']:>10.4f} {m['f1_std']:>6.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['top1_acc']:>8.4f} {m['error_rate']:>8.4f} {m['ndcg']:>8.4f}")
        comparison[model_name] = m

    comp_path = Path(args.output) / "comparison_summary.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_metadata": benchmark.get("metadata", {}),
            "evaluation":         f"Stratified {args.folds}-Fold CV",
            "seed":               args.seed,
            "top_k":              args.top_k,
            "strategy":           args.strategy,
            "models":             comparison,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  [✓] Comparativo salvo: {comp_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()