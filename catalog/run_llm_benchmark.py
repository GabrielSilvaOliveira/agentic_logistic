"""
benchmark/run_llm_benchmark.py
─────────────────────────────────────────────────────────────────────────────
Experimento Condição B e C: LLM/SLM como seletor de fontes.

Condições:
  azure/gpt-4o-mini  : LLM via Azure OpenAI
  azure/gpt-4o       : LLM via Azure OpenAI (validação)
  ollama/<modelo>    : SLM local (ex: llama3.2:3b)

Formatos de catálogo:
  full        : todas as 16 fontes com descrição completa (~1200 tokens)
  compressed  : só nome + embedding_hint (~300 tokens)

Uso:
  # Testa conectividade
  python run_llm_benchmark.py --test-connection --provider azure --model gpt-4o-mini

  # Roda benchmark LLM
  python run_llm_benchmark.py \\
      --catalog   ./output/catalog.json \\
      --benchmark benchmark_queries_final_v2.json \\
      --provider  azure --model gpt-4o-mini \\
      --catalog-format both --output results_llm/

  # Roda benchmark SLM
  python run_llm_benchmark.py \\
      --catalog   ./output/catalog.json \\
      --benchmark benchmark_queries_final_v2.json \\
      --provider  ollama --model llama3.2:3b \\
      --catalog-format both --output results_slm/
"""

import argparse, json, math, os, statistics, time, re
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Carrega variáveis de ambiente do arquivo .env

# ── Serialização do catálogo ──────────────────────────────────────────────────

def serialize_full(catalog):
    lines = ["# CATÁLOGO DE FONTES DE DADOS\n"]
    src = catalog.get("sources", {})
    base_url = src.get("api", {}).get("base_url", "")

    for name, meta in src.get("sql", {}).get("tables", {}).items():
        lines.append(f"## sql.{name}  (SQL)")
        for f, l in [("description","Descrição"),("business_role","Papel"),
                     ("embedding_hint","Usar para"),("when_to_use","Quando usar")]:
            v = meta.get(f, "")
            if v and "[MANUAL]" not in v:
                lines.append(f"**{l}:** {v}")
        lines.append("")

    for name, meta in src.get("api", {}).get("endpoints", {}).items():
        lines.append(f"## api.{name}  (API REST)")
        path = meta.get("path","")
        if path: lines.append(f"**Endpoint:** GET {base_url}{path}")
        for f, l in [("description","Descrição"),("business_role","Papel"),
                     ("embedding_hint","Usar para")]:
            v = meta.get(f, "")
            if v and "[MANUAL]" not in v:
                lines.append(f"**{l}:** {v}")
        lines.append("")

    for name, meta in src.get("spreadsheets", {}).get("files", {}).items():
        lines.append(f"## spreadsheets.{name}  (Planilha)")
        fname = meta.get("filename","")
        if fname: lines.append(f"**Arquivo:** {fname}")
        for f, l in [("description","Descrição"),("business_role","Papel"),
                     ("embedding_hint","Usar para")]:
            v = meta.get(f, "")
            if v and "[MANUAL]" not in v:
                lines.append(f"**{l}:** {v}")
        lines.append("")

    return "\n".join(lines)


def serialize_compressed(catalog):
    lines = ["# FONTES DISPONÍVEIS\n"]
    src = catalog.get("sources", {})

    for name, meta in src.get("sql", {}).get("tables", {}).items():
        hint = meta.get("embedding_hint", meta.get("description", ""))
        if "[MANUAL]" not in hint:
            lines.append(f"- `sql.{name}` — {hint[:130]}")

    for name, meta in src.get("api", {}).get("endpoints", {}).items():
        hint = meta.get("embedding_hint", meta.get("description", ""))
        if "[MANUAL]" not in hint:
            lines.append(f"- `api.{name}` — {hint[:130]}")

    for name, meta in src.get("spreadsheets", {}).get("files", {}).items():
        hint = meta.get("embedding_hint", meta.get("description", ""))
        if "[MANUAL]" not in hint:
            lines.append(f"- `spreadsheets.{name}` — {hint[:130]}")

    return "\n".join(lines)


def build_system_prompt(catalog_text):
    return f"""Você é um assistente especializado em dados logísticos.
Fontes de dados disponíveis:

{catalog_text}

Identifique quais fontes são necessárias para responder à pergunta do usuário.
Responda APENAS com JSON válido no formato:
{{"sources": ["source_id_1", "source_id_2"]}}

Use apenas os source_ids listados. Inclua somente fontes relevantes."""


# ── Clientes LLM ──────────────────────────────────────────────────────────────

class AzureOpenAIClient:
    def __init__(self, model):
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise SystemExit("Execute: pip install openai")

        self.model = model
        endpoint    = os.getenv("OPENAI_API_AZURE_ENDPOINT", "")
        api_key     = os.getenv("AZURE_API_KEY", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not endpoint or not api_key:
            raise ValueError(
                "Configure as variáveis de ambiente:\n"
                "  OPENAI_API_AZURE_ENDPOINT=https://seu-recurso.openai.azure.com/\n"
                "  AZURE_API_KEY=sua-chave\n"
                "  AZURE_OPENAI_API_VERSION=2024-02-01"
            )
        self.client = AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version
        )

    def query(self, user_msg, system_prompt):
        t0 = time.monotonic()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_msg}],
            temperature=0, max_tokens=200,
        )
        ms = (time.monotonic()-t0)*1000
        content = resp.choices[0].message.content.strip()
        usage = {
            "prompt_tokens":     resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens":      resp.usage.total_tokens,
            "latency_ms":        round(ms, 2),
        }
        return content, usage


class OllamaClient:
    def __init__(self, model, base_url="http://localhost:11434"):
        self.model    = model
        self.base_url = base_url.rstrip("/")
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
        except Exception as e:
            raise ConnectionError(
                f"Ollama não acessível em {self.base_url}.\n"
                f"Inicie com: ollama serve\nErro: {e}"
            )

    def query(self, user_msg, system_prompt):
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role":"system","content":system_prompt},
                         {"role":"user","content":user_msg}],
            "stream": False,
            "options": {"temperature": 0},
        }).encode("utf-8")
        t0 = time.monotonic()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type":"application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read().decode("utf-8"))
        ms = (time.monotonic()-t0)*1000
        content = result.get("message",{}).get("content","")
        u = result.get("usage", {})
        usage = {
            "prompt_tokens":     u.get("prompt_tokens",0),
            "completion_tokens": u.get("completion_tokens",0),
            "total_tokens":      u.get("prompt_tokens",0)+u.get("completion_tokens",0),
            "latency_ms":        round(ms,2),
        }
        return content, usage


def parse_sources(content):
    content = re.sub(r"```(?:json)?", "", content).strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "sources" in data:
            return [str(s) for s in data["sources"] if isinstance(s, str)]
    except Exception:
        pass
    match = re.search(r'\{[^{}]*"sources"\s*:\s*\[[^\]]*\][^{}]*\}', content)
    if match:
        try:
            data = json.loads(match.group())
            if "sources" in data:
                return [str(s) for s in data["sources"]]
        except Exception:
            pass
    # fallback: extrai padrões source_id do texto
    found = re.findall(
        r'\b(sql\.[a-zA-Z_]+|api\.[a-zA-Z_]+|spreadsheets\.[a-zA-Z_]+)\b', content
    )
    return list(dict.fromkeys(found))


# ── Métricas ──────────────────────────────────────────────────────────────────

def metrics(selected, q):
    ep  = q["expected_primary"]
    ec  = q.get("expected_complement", [])
    mni = q.get("must_NOT_include", [])
    correct = set(ep)|set(ec)

    prec = sum(1 for s in selected if s in correct)/len(selected) if selected else 0.0
    rec  = sum(1 for e in ep if e in selected)/len(ep) if ep else 1.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    top1 = 1.0 if selected and selected[0] in ep else 0.0
    err  = 1.0 if any(s in set(mni) for s in selected) else 0.0

    prim = set(ep)
    def rel(s): return 2 if s in prim else (1 if s in set(ec) else 0)
    k = 5
    dcg   = sum(rel(selected[i])/math.log2(i+2) for i in range(min(k,len(selected))))
    ideal = sorted([2]*len(ep)+[1]*len(ec), reverse=True)
    idcg  = sum(r/math.log2(i+2) for i,r in enumerate(ideal[:k]))
    ndcg  = dcg/idcg if idcg>0 else 0.0

    return {"precision":round(prec,4),"recall":round(rec,4),"f1":round(f1,4),
            "top1_acc":top1,"error_rate":err,"ndcg":round(ndcg,4)}

def agg(results, extra_keys=()):
    keys = ("precision","recall","f1","top1_acc","error_rate","ndcg",
            "latency_ms","total_tokens")+tuple(extra_keys)
    out = {}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        out[k] = round(sum(vals)/len(vals),4) if vals else 0.0
    return out

def mean_std(vals):
    if not vals: return 0.0, 0.0
    return round(sum(vals)/len(vals),4), round(statistics.stdev(vals) if len(vals)>1 else 0.0, 4)

def stratified_folds(queries, n=5, seed=42):
    import random
    rng = random.Random(seed)
    by_diff = defaultdict(list)
    for i,q in enumerate(queries):
        by_diff[q["difficulty"]].append(i)
    for v in by_diff.values(): rng.shuffle(v)
    folds = [[] for _ in range(n)]
    for v in by_diff.values():
        for i,idx in enumerate(v): folds[i%n].append(idx)
    return folds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM/SLM para seleção de fontes")
    parser.add_argument("--catalog",         required=True)
    parser.add_argument("--benchmark",       required=True)
    parser.add_argument("--provider",        required=True, choices=["azure","ollama"])
    parser.add_argument("--model",           required=True,
                        help="gpt-4o-mini | gpt-4o | llama3.2:3b | etc.")
    parser.add_argument("--catalog-format",  default="both",
                        choices=["full","compressed","both"])
    parser.add_argument("--output",          default="./results_llm/")
    parser.add_argument("--folds",           type=int, default=5)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--ollama-url",      default="http://localhost:11434")
    parser.add_argument("--test-connection", action="store_true")
    parser.add_argument("--max-queries",     type=int, default=None,
                        help="Limita queries para testes rápidos")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    with open(args.catalog, encoding="utf-8") as f:
        catalog = json.load(f)
    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)
    queries = benchmark["queries"]
    if args.max_queries:
        queries = queries[:args.max_queries]

    # Instancia cliente
    if args.provider == "azure":
        client = AzureOpenAIClient(args.model)
        print(f"✅ Azure OpenAI — modelo: {args.model}")
    else:
        client = OllamaClient(args.model, args.ollama_url)
        print(f"✅ Ollama local — modelo: {args.model}")

    # Teste de conectividade
    if args.test_connection:
        cat_text = serialize_compressed(catalog)
        sys_p    = build_system_prompt(cat_text)
        content, usage = client.query(
            "Quais equipamentos estão disponíveis para distribuição?", sys_p
        )
        sources = parse_sources(content)
        print(f"  Resposta bruta: {content[:200]}")
        print(f"  Fontes extraídas: {sources}")
        print(f"  Uso: {usage}")
        return

    formats = ["full","compressed"] if args.catalog_format == "both" else [args.catalog_format]

    for fmt in formats:
        cat_text   = serialize_full(catalog) if fmt=="full" else serialize_compressed(catalog)
        sys_prompt = build_system_prompt(cat_text)
        n_words    = len(cat_text.split())
        slug = (f"{args.provider}_{args.model.replace('/','_').replace(':','_')}_{fmt}")

        print(f"\n{'='*65}")
        print(f"  {args.provider}/{args.model} | catálogo: {fmt} (~{n_words} palavras)")
        print(f"{'='*65}")

        folds            = stratified_folds(queries, n=args.folds, seed=args.seed)
        fold_metrics_all = []
        all_results      = []

        for fold_i, test_idx in enumerate(folds):
            fold_res = []
            for idx in test_idx:
                q = queries[idx]
                try:
                    content, usage = client.query(q["query"], sys_prompt)
                    selected = parse_sources(content)
                except Exception as e:
                    print(f"  ⚠️  {q['id']} erro: {e}")
                    selected, usage = [], {"total_tokens":0,"latency_ms":0,
                                           "prompt_tokens":0,"completion_tokens":0}

                m = metrics(selected, q)
                r = {
                    "id":q["id"],"query":q["query"],
                    "difficulty":q["difficulty"],"cluster":q["cluster"],
                    "selected":selected,
                    "expected_primary":q["expected_primary"],
                    "expected_complement":q.get("expected_complement",[]),
                    "must_NOT_include":q.get("must_NOT_include",[]),
                    **m,
                    "latency_ms":  usage.get("latency_ms",0),
                    "total_tokens":usage.get("total_tokens",0),
                    "prompt_tokens":usage.get("prompt_tokens",0),
                    "completion_tokens":usage.get("completion_tokens",0),
                }
                fold_res.append(r)
                all_results.append(r)

            fold_agg = agg(fold_res)
            fold_metrics_all.append(fold_agg)
            avg_tok = int(sum(r["total_tokens"] for r in fold_res)/max(len(fold_res),1))
            print(f"  Fold {fold_i+1}: F1={fold_agg['f1']:.4f}  "
                  f"P={fold_agg['precision']:.4f}  R={fold_agg['recall']:.4f}  "
                  f"Err={fold_agg['error_rate']:.4f}  Tokens/q≈{avg_tok}")

        # CV final
        cv = {}
        for k in ("precision","recall","f1","top1_acc","error_rate","ndcg",
                  "latency_ms","total_tokens"):
            vals = [fm[k] for fm in fold_metrics_all]
            m, s = mean_std(vals)
            cv[k], cv[f"{k}_std"] = m, s

        by_diff = {}; by_cl = {}
        for field, target in [("difficulty",by_diff),("cluster",by_cl)]:
            groups = defaultdict(list)
            for r in all_results: groups[r[field]].append(r)
            for k,v in groups.items(): target[k] = agg(v)

        print(f"\n  CV ({args.folds}-Fold)  F1={cv['f1']:.4f}±{cv['f1_std']:.4f}  "
              f"P={cv['precision']:.4f}  R={cv['recall']:.4f}  "
              f"Top1={cv['top1_acc']:.4f}  Err={cv['error_rate']:.4f}  "
              f"Tokens/q≈{cv['total_tokens']:.0f}  Lat={cv['latency_ms']:.0f}ms")

        out_data = {
            "provider":args.provider,"model":args.model,"catalog_format":fmt,
            "cv_metrics":cv,"fold_metrics":fold_metrics_all,
            "by_difficulty":by_diff,"by_cluster":by_cl,"all_results":all_results,
        }
        out_path = Path(args.output) / f"results_{slug}.json"
        with open(out_path,"w",encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"  [✓] Salvo: {out_path}")

    print(f"\n{'='*65}\n  Benchmark concluído.\n{'='*65}\n")

if __name__ == "__main__":
    main()
