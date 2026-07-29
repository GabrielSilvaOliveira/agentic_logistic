#!/usr/bin/env python3
"""
lexical_overlap.py
────────────────────────────────────────────────────────────────────────
Segmentação de consultas por SOBREPOSIÇÃO LÉXICA query×catálogo, e
comparação de estratégias de seleção de fontes DENTRO de cada faixa de
sobreposição.

Motivação científica
─────────────────────
O benchmark do SCB mostra (n=80, bootstrap + Holm) que BGE-M3 (melhor
encoder denso) NÃO é estatisticamente distinguível do BM25 (scorer
puramente lexical). Esse "empate global" pode esconder um efeito real:
espera-se que o ganho de um encoder semântico sobre o BM25 se concentre
justamente nas consultas de BAIXA sobreposição léxica com a fonte
correta (paráfrase, sinonímia, vocabulário indireto) — onde o casamento
de termos do BM25 falha por construção. Nas consultas de ALTA
sobreposição, o BM25 tem sinal direto e o encoder não tem o que
acrescentar.

Este script testa exatamente essa hipótese: mede, por consulta, quanto
do vocabulário de conteúdo da consulta aparece no texto da(s) fonte(s)
correta(s) no catálogo (a "sobreposição léxica"), particiona as
consultas em faixas dessa sobreposição, e reaplica o bootstrap pareado
+ Holm-Bonferroni DENTRO de cada faixa.

Dois desfechos, ambos publicáveis:
  • Encoder vence na faixa BAIXA  → resgata a narrativa semântica de
    forma escopada: "embeddings valem exatamente quando o vocabulário
    não bate".
  • BM25 aguenta até a faixa BAIXA → achado contraintuitivo mais forte:
    "em domínio fechado, o léxico resiste mesmo sob paráfrase".

Operacionalização da "sobreposição léxica"
──────────────────────────────────────────
Para cada consulta q com fonte(s)-ouro G (expected_primary +
expected_complement), monta-se o texto de cada fonte a partir dos campos
descritivos do catálogo (embedding_hint, description, business_role,
when_to_use). A sobreposição é medida entre os TOKENS DE CONTEÚDO da
consulta e os tokens do texto-ouro. Métrica primária:

    containment(q, G) = |tokens(q) ∩ tokens(texto_ouro(G))| / |tokens(q)|

isto é, a fração dos termos de conteúdo da consulta que o BM25 pode, em
princípio, casar diretamente com a fonte correta. É a grandeza da qual o
BM25 fundamentalmente depende. Jaccard também é computado como
alternativa (--metric-overlap jaccard).

Agregação sobre múltiplas fontes-ouro: --gold-agg union (padrão, une os
textos das fontes-ouro antes de medir) | max (maior sobreposição entre
as fontes) | mean.

Modos de uso
────────────
  # 1) Perfil de um único modelo por faixa (distribuição + F1 por banda)
  #    + validação de construto contra o rótulo humano 'difficulty'
  python lexical_overlap.py profile \
      --results results_bm25_v3.json \
      --catalog catalog_v3.json \
      --output-queries overlap_por_consulta.csv

  # 2) Comparação entre modelos DENTRO de cada faixa (o teste central)
  python lexical_overlap.py compare \
      --results bm25=results_bm25_v3.json \
      --results bge-m3=results_bge-m3_v3.json \
      --catalog catalog_v3.json \
      --baseline bm25 \
      --output tabela_overlap_por_banda.csv

Requisitos
──────────
  pip install numpy statsmodels
  (reusa paired_bootstrap / apply_holm de stat_compare.py, no mesmo dir)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from itertools import combinations
from pathlib import Path

import numpy as np

# ── Reuso das rotinas estatísticas já validadas do stat_compare.py ──────
# Mantém uma única fonte de verdade para o bootstrap e a correção de Holm.
try:
    from stat_compare import paired_bootstrap, apply_holm
except Exception:
    # Fallback embutido (caso o script seja rodado fora do diretório do
    # stat_compare.py). Implementações idênticas em comportamento.
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        sys.exit("Instale statsmodels: pip install statsmodels --break-system-packages")

    def paired_bootstrap(values_a, values_b, n_boot=10000, seed=42):
        common = sorted(set(values_a) & set(values_b))
        if not common:
            raise ValueError("Nenhuma consulta em comum entre os dois conjuntos.")
        a = np.array([values_a[k] for k in common])
        b = np.array([values_b[k] for k in common])
        diffs = a - b
        observed = float(diffs.mean())
        rng = np.random.default_rng(seed)
        n = len(diffs)
        boot = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
        p = float(2 * min((boot >= 0).mean(), (boot <= 0).mean()))
        p = min(p, 1.0)
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        return {"n_queries": n, "diff": observed, "p_raw": p, "ci_low": lo, "ci_high": hi}

    def apply_holm(rows, alpha=0.05):
        if not rows:
            return rows
        pvals = [r["p_raw"] for r in rows]
        reject, p_corr, _, _ = multipletests(pvals, alpha=alpha, method="holm")
        for r, p_adj, rej in zip(rows, p_corr, reject):
            r["p_holm"] = float(p_adj)
            r["significativo_holm"] = bool(rej)
        return rows


# ─────────────────────────────────────────────────────────────────────────
# TOKENIZAÇÃO PT-BR
# ─────────────────────────────────────────────────────────────────────────

# Stopwords PT-BR de conteúdo-neutro. Remover é essencial: sem isso, termos
# funcionais ("qual", "a", "dos") inflam a sobreposição com ruído comum a
# todas as fontes, mascarando o sinal léxico real.
STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos",
    "das", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "sob",
    "sobre", "entre", "ate", "e", "ou", "que", "qual", "quais", "quando",
    "onde", "como", "quanto", "quanta", "quantos", "quantas", "quem", "cujo",
    "cuja", "se", "ao", "aos", "à", "às", "pelo", "pela", "pelos", "pelas",
    "num", "numa", "nesse", "nessa", "neste", "nesta", "esse", "essa", "este",
    "esta", "isso", "isto", "aquele", "aquela", "seu", "sua", "seus", "suas",
    "meu", "minha", "é", "sao", "foi", "ser", "estar", "esta", "estao", "ha",
    "há", "tem", "têm", "tem", "mais", "menos", "muito", "pouco", "todo",
    "toda", "todos", "todas", "cada", "algum", "alguma", "nenhum", "nao",
    "não", "sim", "ja", "já", "the", "of",
}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _light_stem_pt(tok: str) -> str:
    """
    Stemmer de sufixos MUITO leve para PT-BR — apenas normaliza plurais e
    algumas terminações verbais/nominais comuns, sem dependência externa.
    Objetivo: "terminais"≈"terminal", "equipamentos"≈"equipamento",
    "monitoramento"≈"monitorar" NÃO é coberto (fica conservador de
    propósito para não colar termos distintos). Toggle via --stem.
    """
    for suf in ("acoes", "coes", "oes", "ais", "eis", "ives", "res", "ns"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            # plurais irregulares comuns: terminais->terminal, papeis->papel
            if suf == "ais":
                return tok[:-3] + "al"
            if suf == "eis":
                return tok[:-3] + "el"
            if suf == "oes" or suf == "coes" or suf == "acoes":
                return tok[: -len(suf)] + "ao"
            if suf == "ns":
                return tok[:-2] + "m"
            return tok[: -len(suf)]
    for suf in ("s", "es"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str, remove_stop: bool = True, stem: bool = False) -> set[str]:
    text = _strip_accents(str(text).lower())
    raw = []
    cur = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                raw.append("".join(cur))
                cur = []
    if cur:
        raw.append("".join(cur))
    toks = []
    for t in raw:
        if len(t) <= 1:
            continue
        if remove_stop and t in STOPWORDS_PT:
            continue
        if stem:
            t = _light_stem_pt(t)
        toks.append(t)
    return set(toks)


# ─────────────────────────────────────────────────────────────────────────
# ÍNDICE DE TEXTO DO CATÁLOGO
# ─────────────────────────────────────────────────────────────────────────

CATALOG_TEXT_FIELDS = ("embedding_hint", "description", "business_role", "when_to_use")


def build_catalog_index(catalog_path: Path) -> dict[str, str]:
    """
    Retorna {source_id: texto_descritivo}. source_id no formato usado pelos
    results: 'sql.<tabela>', 'api.<endpoint>', 'spreadsheets.<arquivo>'
    (com alias 'xlsx.<arquivo>' para robustez).
    """
    cat = json.loads(catalog_path.read_text(encoding="utf-8"))
    src = cat["sources"]
    idx: dict[str, str] = {}

    def add(prefix: str, name: str, meta: dict, extra_alias: str | None = None):
        parts = [str(meta.get(f, "") or "") for f in CATALOG_TEXT_FIELDS]
        # inclui também os nomes das colunas (quando SQL) — vocabulário real
        cols = meta.get("columns")
        if isinstance(cols, dict):
            parts.append(" ".join(cols.keys()))
            for cmeta in cols.values():
                if isinstance(cmeta, dict) and cmeta.get("description"):
                    parts.append(str(cmeta["description"]))
        txt = " ".join(p for p in parts if p)
        idx[f"{prefix}.{name}"] = txt
        if extra_alias:
            idx[f"{extra_alias}.{name}"] = txt

    for name, meta in src.get("sql", {}).get("tables", {}).items():
        add("sql", name, meta)
    for name, meta in src.get("api", {}).get("endpoints", {}).items():
        add("api", name, meta)
    for name, meta in src.get("spreadsheets", {}).get("files", {}).items():
        add("spreadsheets", name, meta, extra_alias="xlsx")

    if not idx:
        raise ValueError(f"Nenhuma fonte extraída de {catalog_path}.")
    return idx


def normalize_source_id(sid: str) -> str:
    """Limpa rótulos compostos do benchmark, ex.:
    'spreadsheets.contratos, sheet.fornecimento' -> 'spreadsheets.contratos'."""
    return sid.split(",")[0].strip()


# ─────────────────────────────────────────────────────────────────────────
# CÁLCULO DA SOBREPOSIÇÃO POR CONSULTA
# ─────────────────────────────────────────────────────────────────────────

def _overlap(q_toks: set[str], d_toks: set[str], metric: str) -> float:
    if not q_toks:
        return 0.0
    inter = len(q_toks & d_toks)
    if metric == "containment":
        return inter / len(q_toks)
    if metric == "jaccard":
        union = len(q_toks | d_toks)
        return inter / union if union else 0.0
    raise ValueError(f"metric_overlap desconhecido: {metric}")


def compute_overlaps(
    results_path: Path,
    catalog_idx: dict[str, str],
    metric_overlap: str = "containment",
    gold_agg: str = "union",
    remove_stop: bool = True,
    stem: bool = False,
) -> dict[str, dict]:
    """
    Retorna {query_id: {overlap, f1, difficulty, cluster, n_gold, query,
    unresolved_gold}}.
    """
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if "all_results" not in data:
        raise ValueError(f"{results_path} não contém 'all_results'.")

    # pré-tokeniza textos do catálogo
    cat_toks = {
        sid: tokenize(txt, remove_stop=remove_stop, stem=stem)
        for sid, txt in catalog_idx.items()
    }

    out: dict[str, dict] = {}
    for r in data["all_results"]:
        qid = r["id"]
        q_toks = tokenize(r.get("query", ""), remove_stop=remove_stop, stem=stem)
        gold = list(r.get("expected_primary") or []) + list(r.get("expected_complement") or [])
        gold_norm = [normalize_source_id(g) for g in gold]

        resolved, unresolved = [], []
        for g in gold_norm:
            if g in cat_toks:
                resolved.append(cat_toks[g])
            elif f"xlsx.{g.split('.',1)[-1]}" in cat_toks:
                resolved.append(cat_toks[f"xlsx.{g.split('.',1)[-1]}"])
            else:
                unresolved.append(g)

        if not resolved:
            overlap = float("nan")
        elif gold_agg == "union":
            u = set().union(*resolved)
            overlap = _overlap(q_toks, u, metric_overlap)
        elif gold_agg == "max":
            overlap = max(_overlap(q_toks, d, metric_overlap) for d in resolved)
        elif gold_agg == "mean":
            overlap = float(np.mean([_overlap(q_toks, d, metric_overlap) for d in resolved]))
        else:
            raise ValueError(f"gold_agg desconhecido: {gold_agg}")

        out[qid] = {
            "overlap": overlap,
            "f1": float(r.get("f1", float("nan"))),
            "difficulty": r.get("difficulty", ""),
            "cluster": r.get("cluster", ""),
            "n_gold": len(gold_norm),
            "query": r.get("query", ""),
            "unresolved_gold": ";".join(unresolved),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────
# BINNING
# ─────────────────────────────────────────────────────────────────────────

def assign_bins(
    overlaps: dict[str, dict],
    mode: str = "quantile",
    n_bins: int = 3,
    fixed_edges: list[float] | None = None,
) -> tuple[dict[str, str], list[str], dict]:
    """
    Retorna (qid->rótulo_banda, rótulos_ordenados, meta). Consultas com
    overlap NaN (fonte-ouro não resolvida) ficam de fora do binning.
    """
    valid = {q: d["overlap"] for q, d in overlaps.items() if not np.isnan(d["overlap"])}
    vals = np.array(list(valid.values()))

    if mode == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(vals, qs)
        edges[0], edges[-1] = -np.inf, np.inf
        # rótulos por faixa (tertis: baixa/média/alta quando n_bins=3)
        if n_bins == 3:
            names = ["baixa", "media", "alta"]
        else:
            names = [f"q{i+1}" for i in range(n_bins)]
    elif mode == "fixed":
        e = fixed_edges or [1 / 3, 2 / 3]
        edges = np.array([-np.inf, *e, np.inf])
        names = [f"[{lo:.2f},{hi:.2f})" for lo, hi in zip([0.0, *e], [*e, 1.0])]
        n_bins = len(names)
    else:
        raise ValueError(f"bins mode desconhecido: {mode}")

    assign = {}
    for q, v in valid.items():
        # índice do bin: maior i tal que v >= edges[i]
        b = int(np.searchsorted(edges, v, side="right") - 1)
        b = max(0, min(b, n_bins - 1))
        assign[q] = names[b]

    meta = {
        "mode": mode,
        "edges": [float(x) for x in edges],
        "names": names,
        "n_excluded_nan": len(overlaps) - len(valid),
    }
    return assign, names, meta


# ─────────────────────────────────────────────────────────────────────────
# MODO 1 — PROFILE (um modelo: distribuição, F1 por banda, construto)
# ─────────────────────────────────────────────────────────────────────────

def cmd_profile(args):
    cat_idx = build_catalog_index(Path(args.catalog))
    label, path = _parse_one_results(args.results)
    ov = compute_overlaps(
        Path(path), cat_idx,
        metric_overlap=args.metric_overlap, gold_agg=args.gold_agg,
        remove_stop=not args.keep_stopwords, stem=args.stem,
    )
    assign, names, meta = assign_bins(ov, args.bins, args.n_bins, _parse_edges(args.fixed_edges))

    print(f"[i] Modelo: {label}  |  sobreposição={args.metric_overlap} "
          f"({args.gold_agg}, stem={args.stem}, stopwords_removidas={not args.keep_stopwords})")
    vals = np.array([d["overlap"] for d in ov.values() if not np.isnan(d["overlap"])])
    print(f"[i] Sobreposição léxica query×fonte-ouro (n={len(vals)}): "
          f"min={vals.min():.3f} q25={np.quantile(vals,.25):.3f} "
          f"mediana={np.median(vals):.3f} q75={np.quantile(vals,.75):.3f} max={vals.max():.3f}")
    if meta["n_excluded_nan"]:
        print(f"[!] {meta['n_excluded_nan']} consulta(s) sem fonte-ouro resolvida — fora do binning.")

    # F1 por banda
    print(f"\n{'Banda':<18}{'faixa overlap':<20}{'n':>4}{'F1 médio':>12}{'F1 dp':>10}")
    print("-" * 64)
    for i, nm in enumerate(names):
        qs = [q for q, b in assign.items() if b == nm]
        f1 = np.array([ov[q]["f1"] for q in qs])
        lo = meta["edges"][i]; hi = meta["edges"][i + 1]
        lo_s = "-inf" if lo == -np.inf else f"{lo:.3f}"
        hi_s = "+inf" if hi == np.inf else f"{hi:.3f}"
        print(f"{nm:<18}{f'[{lo_s},{hi_s})':<20}{len(qs):>4}{f1.mean():>12.4f}{f1.std():>10.4f}")

    # Validação de construto: sobreposição computada × rótulo humano difficulty
    print(f"\n[Construto] Sobreposição média por rótulo humano 'difficulty':")
    by_diff = {}
    for q, d in ov.items():
        if np.isnan(d["overlap"]):
            continue
        by_diff.setdefault(d["difficulty"], []).append(d["overlap"])
    for k in sorted(by_diff, key=lambda x: np.mean(by_diff[x])):
        v = np.array(by_diff[k])
        print(f"   {k:32s} n={len(v):>2}  overlap médio={v.mean():.3f}  (F1 médio={np.mean([ov[q]['f1'] for q in ov if ov[q]['difficulty']==k]):.3f})")
    print("   → se as consultas rotuladas 'semantica'/'discriminacao_fina' têm "
          "overlap sistematicamente menor, isso valida a métrica de sobreposição.")

    if args.output_queries:
        _write_query_csv(args.output_queries, ov, assign)
        print(f"\n[✓] Tabela por consulta salva em: {args.output_queries}")


# ─────────────────────────────────────────────────────────────────────────
# MODO 2 — COMPARE (modelos entre si, DENTRO de cada banda)
# ─────────────────────────────────────────────────────────────────────────

def cmd_compare(args):
    cat_idx = build_catalog_index(Path(args.catalog))
    models = dict(_parse_one_results(x) for x in args.results)
    if len(models) < 2:
        sys.exit("Modo 'compare' requer ao menos 2 --results rotulo=arquivo.")

    # overlaps são definidos por CONSULTA (independem do modelo); usa-se o
    # texto da consulta e da fonte-ouro. Calcula a partir do primeiro
    # results (todos compartilham as mesmas consultas/ouro) e reaproveita.
    ref_label = args.baseline if args.baseline in models else sorted(models)[0]
    ov_ref = compute_overlaps(
        Path(models[ref_label]), cat_idx,
        metric_overlap=args.metric_overlap, gold_agg=args.gold_agg,
        remove_stop=not args.keep_stopwords, stem=args.stem,
    )
    assign, names, meta = assign_bins(ov_ref, args.bins, args.n_bins, _parse_edges(args.fixed_edges))

    # F1 por consulta de cada modelo
    f1_by_model = {}
    for label, path in models.items():
        o = compute_overlaps(Path(path), cat_idx, args.metric_overlap, args.gold_agg,
                             not args.keep_stopwords, args.stem)
        f1_by_model[label] = {q: d["f1"] for q, d in o.items()}

    # pares a testar
    if args.baseline:
        if args.baseline not in models:
            sys.exit(f"--baseline '{args.baseline}' não está entre os modelos: {sorted(models)}")
        pairs = [(m, args.baseline) for m in sorted(models) if m != args.baseline]
    else:
        pairs = list(combinations(sorted(models), 2))

    print(f"[i] Sobreposição={args.metric_overlap} ({args.gold_agg}); "
          f"bandas={args.bins}({','.join(names)}); baseline={args.baseline or '(todos-x-todos)'}")
    print(f"[i] Bordas das bandas: {['%.3f'%e for e in meta['edges']]}")
    if meta["n_excluded_nan"]:
        print(f"[!] {meta['n_excluded_nan']} consulta(s) sem fonte-ouro resolvida — excluída(s).")

    # Uma FAMÍLIA de correção de Holm POR BANDA (as bandas são estratos
    # independentes; não se misturam correções entre elas).
    all_rows = []
    for nm in names:
        qids = [q for q, b in assign.items() if b == nm]
        rows = []
        for a, b in pairs:
            va = {q: f1_by_model[a][q] for q in qids if q in f1_by_model[a]}
            vb = {q: f1_by_model[b][q] for q in qids if q in f1_by_model[b]}
            common = set(va) & set(vb)
            va = {q: va[q] for q in common}; vb = {q: vb[q] for q in common}
            res = paired_bootstrap(va, vb)
            rows.append({"comparacao": f"[{nm}] {a} vs {b}", "familia": f"banda:{nm}", **res})
        rows = apply_holm(rows, alpha=args.alpha)
        all_rows.extend(rows)

    _print_compare(all_rows, args.metric)
    if args.output:
        _write_compare_csv(args.output, all_rows)
        print(f"[✓] Tabela salva em: {args.output}")


# ─────────────────────────────────────────────────────────────────────────
# MODO 3 — INTERACTION (teste FORMAL de que a vantagem depende da sobreposição)
# ─────────────────────────────────────────────────────────────────────────
#
# "Significativo na banda baixa e não na alta" NÃO é um teste de interação —
# é a falácia "diferença em significância ≠ significância da diferença".
# Este modo testa diretamente se a vantagem de um modelo sobre o BM25
# DEPENDE da sobreposição léxica, de duas formas complementares:
#
#   (A) CONTÍNUO (primário): regride o delta por consulta (F1_modelo −
#       F1_bm25) sobre a sobreposição contínua e testa a INCLINAÇÃO.
#       Inclinação < 0 ⇒ a vantagem encolhe conforme a sobreposição cresce
#       (i.e. o ganho está na baixa sobreposição). Bootstrap sobre consultas
#       para IC/p — coerente com o resto da metodologia; sem binning.
#
#   (B) BINADO (secundário/interpretável): difference-of-differences entre
#       a banda mais baixa e a mais alta:
#         DiD = média(delta | baixa) − média(delta | alta)
#       Amostras NÃO pareadas (consultas distintas por banda) ⇒ bootstrap
#       de duas amostras independentes.

def _bootstrap_slope(x: np.ndarray, y: np.ndarray, n_boot=10000, seed=42):
    """Inclinação OLS de y~x com IC/p por bootstrap de casos (reamostra
    pares (x_i,y_i)). p bicaudal pela proporção de inclinações que cruzam 0."""
    n = len(x)
    def slope(xx, yy):
        xm, ym = xx.mean(), yy.mean()
        denom = ((xx - xm) ** 2).sum()
        return float(((xx - xm) * (yy - ym)).sum() / denom) if denom else 0.0
    obs = slope(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = slope(x[idx], y[idx])
    p = float(2 * min((boots >= 0).mean(), (boots <= 0).mean()))
    p = min(p, 1.0)
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return {"slope": obs, "p_raw": p, "ci_low": lo, "ci_high": hi, "n": n}


def _bootstrap_did(delta_low: np.ndarray, delta_high: np.ndarray, n_boot=10000, seed=42):
    """Difference-of-differences com bootstrap de duas amostras independentes.
    DiD = mean(delta_low) − mean(delta_high)."""
    obs = float(delta_low.mean() - delta_high.mean())
    rng = np.random.default_rng(seed)
    nl, nh = len(delta_low), len(delta_high)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bl = delta_low[rng.integers(0, nl, nl)].mean()
        bh = delta_high[rng.integers(0, nh, nh)].mean()
        boots[i] = bl - bh
    p = float(2 * min((boots >= 0).mean(), (boots <= 0).mean()))
    p = min(p, 1.0)
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return {"did": obs, "p_raw": p, "ci_low": lo, "ci_high": hi,
            "n_low": nl, "n_high": nh}


def cmd_interaction(args):
    cat_idx = build_catalog_index(Path(args.catalog))
    models = dict(_parse_one_results(x) for x in args.results)
    if args.baseline not in models:
        sys.exit(f"--baseline '{args.baseline}' precisa estar entre os --results.")
    others = [m for m in sorted(models) if m != args.baseline]
    if not others:
        sys.exit("Informe ao menos um modelo além do --baseline.")

    # sobreposição por consulta (definida pela consulta+ouro; usa baseline)
    ov = compute_overlaps(
        Path(models[args.baseline]), cat_idx,
        metric_overlap=args.metric_overlap, gold_agg=args.gold_agg,
        remove_stop=not args.keep_stopwords, stem=args.stem,
    )
    assign, names, meta = assign_bins(ov, args.bins, args.n_bins, _parse_edges(args.fixed_edges))
    low_name, high_name = names[0], names[-1]

    # F1 por consulta de cada modelo
    f1 = {}
    for label, path in models.items():
        o = compute_overlaps(Path(path), cat_idx, args.metric_overlap, args.gold_agg,
                             not args.keep_stopwords, args.stem)
        f1[label] = {q: d["f1"] for q, d in o.items()}

    base = f1[args.baseline]

    print(f"[i] Teste de interação sobreposição×modelo (baseline={args.baseline}, "
          f"overlap={args.metric_overlap}/{args.gold_agg})")
    print(f"[i] H0: a vantagem sobre o {args.baseline} NÃO depende da sobreposição léxica.")

    # ── (A) inclinação contínua ─────────────────────────────────────────
    print("\n(A) INCLINAÇÃO CONTÍNUA — delta(modelo−baseline) ~ sobreposição")
    print("    (inclinação < 0 ⇒ vantagem concentrada na BAIXA sobreposição)")
    hdr = f"{'Modelo':<12}{'n':>4}{'inclinação':>12}{'IC 95%':>24}{'p_raw':>9}{'p_holm':>9}  sig."
    print(hdr); print("-" * len(hdr))
    rowsA = []
    for m in others:
        qids = [q for q in ov if not np.isnan(ov[q]["overlap"]) and q in f1[m] and q in base]
        x = np.array([ov[q]["overlap"] for q in qids])
        y = np.array([f1[m][q] - base[q] for q in qids])
        rowsA.append((m, _bootstrap_slope(x, y)))
    _holm_inplace([r for _, r in rowsA], args.alpha)
    for m, r in rowsA:
        sig = "✓" if r["significativo_holm"] else " "
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"{m:<12}{r['n']:>4}{r['slope']:>+12.4f}{ci:>24}{r['p_raw']:>9.4f}{r['p_holm']:>9.4f}   {sig}")

    # ── (B) DiD binado baixa×alta ───────────────────────────────────────
    print(f"\n(B) DIFFERENCE-OF-DIFFERENCES — banda '{low_name}' vs '{high_name}'")
    print(f"    DiD = média(delta | {low_name}) − média(delta | {high_name}); "
          f">0 ⇒ vantagem maior na banda baixa")
    hdr2 = f"{'Modelo':<12}{'n_baixa':>8}{'n_alta':>7}{'DiD':>10}{'IC 95%':>24}{'p_raw':>9}{'p_holm':>9}  sig."
    print(hdr2); print("-" * len(hdr2))
    low_q = [q for q, b in assign.items() if b == low_name]
    high_q = [q for q, b in assign.items() if b == high_name]
    rowsB = []
    for m in others:
        dl = np.array([f1[m][q] - base[q] for q in low_q if q in f1[m] and q in base])
        dh = np.array([f1[m][q] - base[q] for q in high_q if q in f1[m] and q in base])
        rowsB.append((m, _bootstrap_did(dl, dh)))
    _holm_inplace([r for _, r in rowsB], args.alpha)
    for m, r in rowsB:
        sig = "✓" if r["significativo_holm"] else " "
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"{m:<12}{r['n_low']:>8}{r['n_high']:>7}{r['did']:>+10.4f}{ci:>24}"
              f"{r['p_raw']:>9.4f}{r['p_holm']:>9.4f}   {sig}")

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["teste", "modelo", "estatistica", "valor", "ci_low", "ci_high",
                        "p_raw", "p_holm", "significativo_holm"])
            for m, r in rowsA:
                w.writerow(["inclinacao_continua", m, "slope", f"{r['slope']:.6f}",
                            f"{r['ci_low']:.6f}", f"{r['ci_high']:.6f}", f"{r['p_raw']:.6f}",
                            f"{r['p_holm']:.6f}", r["significativo_holm"]])
            for m, r in rowsB:
                w.writerow(["did_baixa_vs_alta", m, "DiD", f"{r['did']:.6f}",
                            f"{r['ci_low']:.6f}", f"{r['ci_high']:.6f}", f"{r['p_raw']:.6f}",
                            f"{r['p_holm']:.6f}", r["significativo_holm"]])
        print(f"\n[✓] Tabela salva em: {args.output}")
    print("\n[nota] Correção de Holm aplicada DENTRO de cada teste (A e B) "
          "separadamente — famílias distintas.")


def _holm_inplace(rows: list[dict], alpha: float):
    """Aplica Holm sobre uma lista de dicts que já contêm 'p_raw'."""
    if not rows:
        return
    dummy = [{"p_raw": r["p_raw"]} for r in rows]
    dummy = apply_holm(dummy, alpha=alpha)
    for r, d in zip(rows, dummy):
        r["p_holm"] = d["p_holm"]
        r["significativo_holm"] = d["significativo_holm"]


# ─────────────────────────────────────────────────────────────────────────
# AUX
# ─────────────────────────────────────────────────────────────────────────

def _parse_one_results(item: str) -> tuple[str, str]:
    if "=" in item:
        label, path = item.split("=", 1)
        return label, path
    return Path(item).stem.replace("results_", ""), item


def _parse_edges(s: str | None) -> list[float] | None:
    if not s:
        return None
    return [float(x) for x in s.split(",")]


def _write_query_csv(path, ov, assign):
    fields = ["id", "banda", "overlap", "f1", "difficulty", "cluster", "n_gold", "unresolved_gold", "query"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for q, d in sorted(ov.items()):
            w.writerow({
                "id": q, "banda": assign.get(q, "(nan)"),
                "overlap": f"{d['overlap']:.4f}" if not np.isnan(d["overlap"]) else "",
                "f1": f"{d['f1']:.4f}", "difficulty": d["difficulty"], "cluster": d["cluster"],
                "n_gold": d["n_gold"], "unresolved_gold": d["unresolved_gold"], "query": d["query"],
            })


def _print_compare(rows, metric):
    label = f"Δ{metric}"
    header = f"{'Comparação (por banda)':<42}{'n':>4}{label:>10}{'IC 95%':>22}{'p_raw':>9}{'p_holm':>9}  sig."
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        sig = "✓" if r["significativo_holm"] else " "
        print(f"{r['comparacao']:<42}{r['n_queries']:>4}{r['diff']:>+10.4f}{ci:>22}"
              f"{r['p_raw']:>9.4f}{r['p_holm']:>9.4f}   {sig}")
    print()


def _write_compare_csv(path, rows):
    fields = ["comparacao", "familia", "n_queries", "diff", "p_raw", "p_holm",
              "ci_low", "ci_high", "significativo_holm"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def _add_common(p):
    p.add_argument("--catalog", required=True, help="catalog_v3.json")
    p.add_argument("--metric", default="f1", help="Métrica de qualidade por consulta (padrão: f1)")
    p.add_argument("--metric-overlap", default="containment", choices=["containment", "jaccard"],
                   help="Métrica de sobreposição léxica (padrão: containment)")
    p.add_argument("--gold-agg", default="union", choices=["union", "max", "mean"],
                   help="Agregação sobre múltiplas fontes-ouro (padrão: union)")
    p.add_argument("--bins", default="quantile", choices=["quantile", "fixed"],
                   help="Estratégia de binning (padrão: quantile=tertis)")
    p.add_argument("--n-bins", type=int, default=3, help="Nº de bandas (padrão: 3)")
    p.add_argument("--fixed-edges", default=None,
                   help="Bordas para --bins fixed, ex.: 0.33,0.66")
    p.add_argument("--keep-stopwords", action="store_true",
                   help="NÃO remover stopwords PT (padrão: remove)")
    p.add_argument("--stem", action="store_true",
                   help="Aplica stemmer leve PT (plurais/terminações) — análise de sensibilidade")
    p.add_argument("--alpha", type=float, default=0.05)


def main():
    ap = argparse.ArgumentParser(
        description="Segmentação por sobreposição léxica query×catálogo e "
                    "comparação de modelos por faixa (extensão do stat_compare.py).")
    sub = ap.add_subparsers(dest="mode", required=True)

    pp = sub.add_parser("profile", help="Perfil de um modelo: distribuição, F1 por banda, construto")
    pp.add_argument("--results", required=True, help="rotulo=arquivo OU só o arquivo")
    pp.add_argument("--output-queries", default=None, help="CSV por consulta (id,banda,overlap,f1,...)")
    _add_common(pp)
    pp.set_defaults(func=cmd_profile)

    pc = sub.add_parser("compare", help="Compara modelos DENTRO de cada banda (teste central)")
    pc.add_argument("--results", action="append", required=True,
                    help="rotulo=arquivo, repetível (ex.: --results bm25=... --results bge-m3=...)")
    pc.add_argument("--baseline", default=None,
                    help="Rótulo do modelo de referência (cada modelo é testado contra ele). "
                         "Sem isto: todos-contra-todos por banda.")
    pc.add_argument("--output", default=None, help="CSV de saída")
    _add_common(pc)
    pc.set_defaults(func=cmd_compare)

    pi = sub.add_parser("interaction",
                        help="Teste FORMAL de que a vantagem depende da sobreposição "
                             "(inclinação contínua + DiD baixa×alta)")
    pi.add_argument("--results", action="append", required=True,
                    help="rotulo=arquivo, repetível (inclua o baseline)")
    pi.add_argument("--baseline", required=True, help="Rótulo do modelo de referência (ex.: bm25)")
    pi.add_argument("--output", default=None, help="CSV de saída")
    _add_common(pi)
    pi.set_defaults(func=cmd_interaction)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
