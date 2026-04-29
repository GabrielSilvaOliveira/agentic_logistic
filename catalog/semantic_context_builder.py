"""
catalog/semantic_context_builder.py
─────────────────────────────────────────────────────────────────────────────
Semantic Context Builder (v2) — seleção de fontes por similaridade semântica.

Herda todos os renderizadores Markdown do ContextBuilder (v1) e substitui
apenas o método _select_sources(), usando embeddings ao invés de keywords.

Modelos suportados:
  "minilm"    → sentence-transformers/all-MiniLM-L6-v2   (EN, leve)
  "e5-base"   → intfloat/multilingual-e5-base             (multilingual)
  "e5-large"  → intfloat/multilingual-e5-large            (multilingual, maior)
  "bge-m3"    → BAAI/bge-m3                               (multilingual, melhor qualidade)

Estratégia de seleção:
  1. Embedda o texto de cada fonte do catálogo (documentos)
  2. Embedda a query do usuário
  3. Calcula cosine similarity query ↔ cada documento
  4. Aplica estratégia de corte ("score_gap" padrão ou "threshold")
       score_gap  : corta no maior salto entre scores consecutivos
                    + floor min_similarity + teto top_k
       threshold  : clássico — inclui os top_k acima de min_similarity
  5. Fallback: se nenhuma fonte selecionada → inclui todas (igual v1)

Construção dos documentos de embedding (prioridade):
  1. embedding_hint  : campo manual curto/focado no catálogo — usado se preenchido
  2. Automático      : description + business_role + top-8 colunas discriminativas
                       (sem when_to_use / when_NOT_to_use — diluem o sinal)

Cache de embeddings:
  - Salvo em <cache_dir>/<model_slug>.pkl
  - Invalidado automaticamente se o catalog.json mudar (SHA-256)
  - Rebuild forçado com rebuild_index(force=True)

Uso:
  cb = SemanticContextBuilder(
      catalog_path="./output/catalog.json",
      model_name="e5-base",        # threshold padrão do modelo: 0.79
      top_k=3,
      selection_strategy="score_gap",  # corte pelo maior gap de score
  )
  context = cb.build_context("Quais distribuições pendentes para LOC002?")
  # → string Markdown idêntica ao v1
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from catalog.context_builder import ContextBuilder


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRO DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────

# Cada entrada define:
#   hf_id        : identificador no HuggingFace Hub
#   query_prefix : prefixo exigido pelo modelo na query (E5 precisa de "query: ")
#   doc_prefix   : prefixo exigido pelo modelo nos documentos (E5 precisa de "passage: ")
#   normalize    : se True, normaliza os vetores antes da cosine similarity
MODEL_REGISTRY: dict[str, dict] = {
    "minilm": {
        "hf_id":                  "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix":           "",
        "doc_prefix":             "",
        "normalize":              True,
        # Spread baixo (~0.10–0.35) → threshold absoluto funciona bem
        "default_min_similarity": 0.30,
    },
    "e5-base": {
        "hf_id":                  "intfloat/multilingual-e5-base",
        "query_prefix":           "query: ",
        "doc_prefix":             "passage: ",
        "normalize":              True,
        # Spread estreito (~0.77–0.82) → threshold alto + score_gap
        "default_min_similarity": 0.79,
    },
    "e5-large": {
        "hf_id":                  "intfloat/multilingual-e5-large",
        "query_prefix":           "query: ",
        "doc_prefix":             "passage: ",
        "normalize":              True,
        "default_min_similarity": 0.79,
    },
    "bge-m3": {
        "hf_id":                  "BAAI/bge-m3",
        "query_prefix":           "",
        "doc_prefix":             "",
        "normalize":              True,
        "default_min_similarity": 0.50,
    },
}

# Slug seguro para nome de arquivo de cache
_SLUG_MAP = {
    "minilm":   "minilm",
    "e5-base":  "e5_base",
    "e5-large": "e5_large",
    "bge-m3":   "bge_m3",
}

# Nomes de colunas genéricas a omitir do documento de embedding
# (termos não discriminativos que diluem o sinal semântico)
_GENERIC_COLUMNS: frozenset[str] = frozenset({
    "CODIGO", "ID", "id", "ULTIMAATUALIZACAO",
    "created_at", "updated_at", "DATAINCLUSAOCARGA",
    "usuario_decisao_id",
})

# Máximo de colunas por fonte incluídas no documento de embedding
_MAX_EMBEDDING_COLS: int = 8


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class SemanticContextBuilder(ContextBuilder):
    """
    Context Builder com seleção semântica de fontes via embeddings.
    Herda toda a lógica de renderização Markdown do ContextBuilder.
    """

    def __init__(
        self,
        catalog_path: str,
        model_name: str = "e5-base",
        top_k: int = 3,
        min_similarity: Optional[float] = None,
        selection_strategy: str = "score_gap",
        cache_dir: Optional[str] = None,
    ):
        """
        Parâmetros:
          catalog_path       : caminho para catalog.json
          model_name         : chave do MODEL_REGISTRY ("minilm", "e5-base", "e5-large", "bge-m3")
          top_k              : número máximo de fontes a selecionar
          min_similarity     : threshold mínimo de cosine similarity (0–1)
                               (None = usa o default do modelo definido no MODEL_REGISTRY)
          selection_strategy : estratégia de corte:
                               "score_gap"  — corta no maior salto entre scores (padrão)
                               "threshold"  — clássico top_k acima do threshold
          cache_dir          : diretório para cache de embeddings
                               (padrão: <pasta do catalog.json>/embeddings_cache/)
        """
        super().__init__(catalog_path)

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Modelo '{model_name}' não encontrado. "
                f"Opções: {list(MODEL_REGISTRY.keys())}"
            )
        if selection_strategy not in ("score_gap", "threshold"):
            raise ValueError(
                f"Estratégia '{selection_strategy}' inválida. "
                f"Use 'score_gap' ou 'threshold'."
            )

        self.model_name         = model_name
        self.top_k              = top_k
        self._model_cfg         = MODEL_REGISTRY[model_name]
        # Se não informado, usa o threshold padrão calibrado por modelo
        self.min_similarity = (
            min_similarity
            if min_similarity is not None
            else self._model_cfg["default_min_similarity"]
        )
        self.selection_strategy = selection_strategy

        # Diretório de cache
        base = Path(catalog_path).parent
        self._cache_dir = Path(cache_dir) if cache_dir else base / "embeddings_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Estado interno — carregados sob demanda
        self._encoder   = None          # SentenceTransformer
        self._source_ids: list[str] = []
        self._matrix: Optional[np.ndarray] = None  # shape (n_sources, dim)

    # ── Carregamento do modelo ────────────────────────────────────────────────

    @property
    def encoder(self):
        """Lazy load do modelo de embedding."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            print(f"[SemanticCB] Carregando modelo: {self._model_cfg['hf_id']}")
            self._encoder = SentenceTransformer(self._model_cfg["hf_id"])
        return self._encoder

    # ── Índice de embeddings ──────────────────────────────────────────────────

    def _cache_path(self) -> Path:
        slug = _SLUG_MAP[self.model_name]
        return self._cache_dir / f"{slug}.pkl"

    def _catalog_hash(self) -> str:
        """SHA-256 do arquivo catalog.json para detectar mudanças."""
        raw = self.catalog_path.read_bytes()
        return hashlib.sha256(raw).hexdigest()

    def _load_cache(self) -> bool:
        """
        Tenta carregar o índice do cache.
        Retorna True se bem-sucedido e válido, False caso contrário.
        """
        path = self._cache_path()
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data.get("catalog_hash") != self._catalog_hash():
                print(f"[SemanticCB] Catálogo alterado — reconstruindo índice '{self.model_name}'")
                return False
            self._source_ids = data["source_ids"]
            self._matrix     = data["matrix"]
            return True
        except Exception as e:
            print(f"[SemanticCB] Cache inválido ({e}) — reconstruindo índice")
            return False

    def _save_cache(self):
        path = self._cache_path()
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "catalog_hash": self._catalog_hash(),
                    "source_ids":   self._source_ids,
                    "matrix":       self._matrix,
                },
                f,
            )

    def rebuild_index(self, force: bool = False, verbose: bool = False):
        """
        Constrói (ou reconstrói) o índice de embeddings.

        Parâmetros:
          force   : se True, ignora o cache e reconstrói sempre
          verbose : se True, imprime os documentos gerados por fonte
        """
        if not force and self._load_cache():
            if verbose:
                print(f"[SemanticCB] Índice carregado do cache ({self.model_name}): "
                      f"{len(self._source_ids)} fontes")
            return

        documents, source_ids = self._build_documents(verbose=verbose)

        doc_prefix = self._model_cfg["doc_prefix"]
        texts = [doc_prefix + doc for doc in documents]

        print(f"[SemanticCB] Computando embeddings para {len(texts)} fontes "
              f"com '{self.model_name}'...")
        matrix = self.encoder.encode(
            texts,
            normalize_embeddings=self._model_cfg["normalize"],
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        self._source_ids = source_ids
        self._matrix     = matrix
        self._save_cache()
        print(f"[SemanticCB] Índice construído e salvo. Shape: {matrix.shape}")

    # ── Construção dos documentos de embedding ────────────────────────────────

    def _build_documents(
        self, verbose: bool = False
    ) -> tuple[list[str], list[str]]:
        """
        Converte cada fonte do catálogo em um documento de texto rico,
        usado para gerar os embeddings de referência.

        Retorna:
          documents  : lista de strings (um por fonte)
          source_ids : lista de ids no formato "tipo.chave" (ex: "sql.materiais")
        """
        documents: list[str] = []
        source_ids: list[str] = []
        sources = self.catalog.get("sources", {})

        # ── Tabelas SQL ───────────────────────────────────────────────────────
        for table_name, meta in sources.get("sql", {}).get("tables", {}).items():
            doc = self._doc_from_source(
                name=f"Tabela SQL: {table_name}",
                meta=meta,
                columns=meta.get("columns", {}),
            )
            if verbose:
                print(f"\n--- sql.{table_name} ---\n{doc}")
            documents.append(doc)
            source_ids.append(f"sql.{table_name}")

        # ── Endpoints de API ──────────────────────────────────────────────────
        for ep_key, meta in sources.get("api", {}).get("endpoints", {}).items():
            # Para API: "colunas" são os parâmetros e campos de resposta
            params = meta.get("parameters", {}).get("query_params", {})
            response_fields = meta.get("response_schema", {}).get("result_object_fields", {})
            pseudo_columns = {**params, **response_fields}
            doc = self._doc_from_source(
                name=f"API endpoint: {ep_key} ({meta.get('method','GET')} {meta.get('path','')})",
                meta=meta,
                columns=pseudo_columns,
            )
            if verbose:
                print(f"\n--- api.{ep_key} ---\n{doc}")
            documents.append(doc)
            source_ids.append(f"api.{ep_key}")

        # ── Planilhas ─────────────────────────────────────────────────────────
        for file_key, meta in sources.get("spreadsheets", {}).get("files", {}).items():
            # Agrega colunas de todas as abas
            all_columns: dict = {}
            for sheet_meta in meta.get("sheets", {}).values():
                all_columns.update(sheet_meta.get("columns", {}))
            doc = self._doc_from_source(
                name=f"Planilha: {meta.get('filename', file_key)}",
                meta=meta,
                columns=all_columns,
            )
            if verbose:
                print(f"\n--- spreadsheets.{file_key} ---\n{doc}")
            documents.append(doc)
            source_ids.append(f"spreadsheets.{file_key}")

        return documents, source_ids

    @staticmethod
    def _clean(value: str, field_name: str) -> str:
        """
        Limpa campos [MANUAL] não preenchidos.
        Se o campo ainda tiver a marcação, substitui pelo nome do campo como fallback.
        """
        if not value or "[MANUAL]" in value:
            return field_name.replace("_", " ")
        return value.strip()

    def _doc_from_source(
        self, name: str, meta: dict, columns: dict
    ) -> str:
        """
        Monta o documento de texto que representa uma fonte de dados.
        Usado para gerar o embedding de referência dessa fonte.

        Lógica de prioridade:
          1. embedding_hint (manual, curto, focado) — se preenchido, usa exclusivamente
          2. Automático: description + business_role + top-_MAX_EMBEDDING_COLS colunas
             (sem when_to_use / when_NOT_to_use, que diluem o sinal semântico)
        """
        # Prioridade 1: embedding_hint manual curto e focado
        hint = meta.get("embedding_hint", "")
        if hint and "[MANUAL]" not in hint:
            return f"{name}\n{hint.strip()}"

        # Prioridade 2: documento focado construído automaticamente
        parts: list[str] = [name]

        # Apenas description + business_role (when_to_use dilui o sinal)
        for field in ("description", "business_role"):
            raw = meta.get(field, "")
            cleaned = self._clean(raw, field)
            if cleaned:
                parts.append(cleaned)

        # Colunas: top _MAX_EMBEDDING_COLS, excluindo nomes puramente genéricos
        if columns:
            col_parts: list[str] = []
            for col_name, col_meta in columns.items():
                if col_name in _GENERIC_COLUMNS:
                    continue
                if isinstance(col_meta, dict):
                    raw_desc = col_meta.get("description", "")
                    col_desc = self._clean(raw_desc, col_name)
                    col_parts.append(f"{col_name}: {col_desc}")
                else:
                    col_parts.append(col_name)
                if len(col_parts) >= _MAX_EMBEDDING_COLS:
                    break
            if col_parts:
                parts.append("Campos: " + "; ".join(col_parts))

        return "\n".join(parts)

    # ── Seleção semântica (substitui _select_sources do v1) ───────────────────

    def _cut_by_score_gap(
        self, ranked: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        """
        Encontra o maior gap entre scores consecutivos nos top-(top_k+1)
        candidatos e usa isso como ponto de corte natural.

        Exemplo (E5): scores [0.824, 0.803, 0.791, 0.771]
          gaps:  0.021, 0.012, 0.020
          maior gap em pos 0→1  →  retorna apenas [(idx_0, 0.824)]

        Garante ao menos 1 candidato retornado.
        """
        window = ranked[: self.top_k + 1]
        if len(window) <= 1:
            return window

        best_gap  = -1.0
        cut_after = 0
        for i in range(len(window) - 1):
            gap = float(window[i][1]) - float(window[i + 1][1])
            if gap > best_gap:
                best_gap  = gap
                cut_after = i

        return window[: min(cut_after + 1, self.top_k)]

    def _select_sources(self, query: str, verbose: bool = False) -> dict:
        """
        Seleciona fontes por similaridade semântica com a query.
        Substitui a seleção por keywords do ContextBuilder v1.
        """
        # Garante que o índice está pronto
        if self._matrix is None:
            self.rebuild_index(verbose=verbose)

        query_prefix = self._model_cfg["query_prefix"]
        q_vec = self.encoder.encode(
            query_prefix + query,
            normalize_embeddings=self._model_cfg["normalize"],
            convert_to_numpy=True,
        )

        # Cosine similarity: como os vetores já estão normalizados → produto interno
        similarities = self._matrix @ q_vec  # shape (n_sources,)

        ranked = sorted(
            enumerate(similarities),
            key=lambda x: x[1],
            reverse=True,
        )

        # Aplica estratégia de corte
        if self.selection_strategy == "score_gap":
            # Corte natural pelo maior gap + floor min_similarity + teto top_k
            candidates = self._cut_by_score_gap(ranked)
            candidates = [
                (idx, sc) for idx, sc in candidates
                if sc >= self.min_similarity
            ]
        else:  # "threshold" — comportamento clássico
            candidates = [
                (idx, sc) for idx, sc in ranked[: self.top_k]
                if sc >= self.min_similarity
            ]

        selected_ids: list[str] = []
        scores: dict[str, float] = {}
        for idx, score in candidates:
            selected_ids.append(self._source_ids[idx])
            scores[self._source_ids[idx]] = float(score)

        # Fallback: nenhuma fonte selecionada → inclui todas
        if not selected_ids:
            if verbose:
                print(f"[SemanticCB] Nenhuma fonte selecionada "
                      f"(strategy={self.selection_strategy}, "
                      f"threshold={self.min_similarity:.2f}) → modo completo")
            sources = self.catalog.get("sources", {})
            return {
                "sql":          list(sources.get("sql", {}).get("tables", {}).keys()),
                "api":          list(sources.get("api", {}).get("endpoints", {}).keys()),
                "spreadsheets": list(sources.get("spreadsheets", {}).get("files", {}).keys()),
            }

        if verbose:
            print(f"[SemanticCB] Query: \"{query}\"")
            print(f"  Estratégia : {self.selection_strategy}  |  "
                  f"threshold={self.min_similarity:.2f}  |  top_k={self.top_k}")
            print(f"  Fontes selecionadas:")
            for sid, sc in sorted(scores.items(), key=lambda x: -x[1]):
                print(f"    {sid:<35} sim={sc:.4f}")

        # Decompõe ids em {tipo: [chaves]}
        result: dict[str, list[str]] = {"sql": [], "api": [], "spreadsheets": []}
        for source_id in selected_ids:
            parts = source_id.split(".", 1)
            if len(parts) == 2:
                source_type, key = parts
                if source_type in result:
                    result[source_type].append(key)
        return result

    # ── Informações de diagnóstico ────────────────────────────────────────────

    def similarity_report(self, query: str) -> str:
        """
        Retorna um relatório textual com as similaridades de todas as fontes
        para a query fornecida. Útil para debugging e benchmarking.
        """
        if self._matrix is None:
            self.rebuild_index()

        query_prefix = self._model_cfg["query_prefix"]
        q_vec = self.encoder.encode(
            query_prefix + query,
            normalize_embeddings=self._model_cfg["normalize"],
            convert_to_numpy=True,
        )
        similarities = self._matrix @ q_vec

        lines = [
            f"Modelo     : {self.model_name} ({self._model_cfg['hf_id']})",
            f"Estratégia : {self.selection_strategy}",
            f"Query      : \"{query}\"",
            f"Top-k      : {self.top_k}  |  Threshold : {self.min_similarity:.2f}",
            "",
            f"{'Fonte':<40} {'Similaridade':>12}  Selec.",
            "-" * 60,
        ]
        ranked = sorted(enumerate(similarities), key=lambda x: -x[1])
        for idx, score in ranked:
            sid    = self._source_ids[idx]
            sel    = "✓" if score >= self.min_similarity else " "
            lines.append(f"{sid:<40} {score:>12.4f}  {sel}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

# 10 queries com gabarito — fontes esperadas por query
# Formato: (query, set_de_source_ids_esperados)
BENCHMARK_QUERIES: list[tuple[str, set[str]]] = [
    (
        "Quais distribuições estão pendentes para a unidade LOC002?",
        {"api.distribuicao_list", "sql.unidades"},
    ),
    (
        "Quantos equipamentos estão com situação física indisponível?",
        {"sql.materiais", "sql.catalogo", "sql.unidades"},
    ),
    (
        "Qual o valor total dos contratos vigentes de 2024?",
        {"spreadsheets.contratos"},
    ),
    (
        "Quais materiais foram entregues no depósito na última semana?",
        {"api.distribuicao_list"},
    ),
    (
        "Mostre o catálogo de equipamentos.",
        {"sql.catalogo"},
    ),
    (
        "Qual unidade tem mais equipamentos em manutenção?",
        {"sql.materiais", "sql.unidades"},
    ),
    (
        "Quais itens foram adquiridos pelo contrato número 12/2023?",
        {"spreadsheets.contratos"},
    ),
    (
        "Liste os equipamentos cautelados na unidade REGIAO SUL.",
        {"sql.materiais", "sql.unidades"},
    ),
    (
        "Qual a quantidade total de itens distribuídos por unidade de destino?",
        {"api.distribuicao_list", "sql.unidades"},
    ),
    (
        "Qual fabricante tem mais equipamentos registrados no sistema?",
        {"sql.catalogo", "sql.materiais"},
    ),
]


def run_benchmark(
    catalog_path: str,
    models: list[str] | None = None,
    top_k: int = 3,
    min_similarity: float | None = None,
    selection_strategy: str = "score_gap",
    verbose_sources: bool = False,
) -> None:
    """
    Executa o benchmark comparativo entre os modelos de embedding.

    Métricas por query:
      - Precision@k : fontes corretas selecionadas / fontes selecionadas
      - Recall@k    : fontes corretas selecionadas / fontes esperadas

    Parâmetros:
      catalog_path       : caminho para catalog.json
      models             : lista de chaves do MODEL_REGISTRY (None = todos)
      top_k              : máximo de fontes a selecionar
      min_similarity     : threshold de corte (None = usa default do modelo)
      selection_strategy : "score_gap" (padrão) ou "threshold"
      verbose_sources    : se True, imprime as fontes selecionadas por query
    """
    if models is None:
        models = list(MODEL_REGISTRY.keys())

    results: dict[str, dict] = {}  # model → {precision, recall, hits por query}

    for model_name in models:
        print(f"\n{'='*65}")
        print(f"  Modelo: {model_name} ({MODEL_REGISTRY[model_name]['hf_id']})")
        print(f"{'='*65}")

        cb = SemanticContextBuilder(
            catalog_path=catalog_path,
            model_name=model_name,
            top_k=top_k,
            min_similarity=min_similarity,
            selection_strategy=selection_strategy,
        )
        cb.rebuild_index(verbose=False)
        print(f"  threshold={cb.min_similarity:.2f}  |  strategy={cb.selection_strategy}")

        precisions: list[float] = []
        recalls:    list[float] = []

        for i, (query, expected) in enumerate(BENCHMARK_QUERIES, 1):
            selected_dict = cb._select_sources(query, verbose=False)

            # Reconstrói o set de source_ids selecionados
            selected_set: set[str] = set()
            for src_type, keys in selected_dict.items():
                for key in keys:
                    selected_set.add(f"{src_type}.{key}")

            # Métricas
            tp        = len(selected_set & expected)
            precision = tp / len(selected_set) if selected_set else 0.0
            recall    = tp / len(expected)      if expected    else 1.0
            precisions.append(precision)
            recalls.append(recall)

            status = "✓" if recall == 1.0 else ("~" if tp > 0 else "✗")
            print(f"\n  [{i:02d}] {status}  P={precision:.2f}  R={recall:.2f}")
            print(f"       Query   : {query[:70]}")
            print(f"       Esperado: {sorted(expected)}")
            if verbose_sources:
                print(f"       Selecion: {sorted(selected_set)}")

            # Imprime relatório de similaridade para queries com erro
            if recall < 1.0:
                report = cb.similarity_report(query)
                for line in report.split("\n"):
                    print(f"         {line}")

        macro_p = sum(precisions) / len(precisions)
        macro_r = sum(recalls)    / len(recalls)
        f1      = (2 * macro_p * macro_r / (macro_p + macro_r)
                   if (macro_p + macro_r) > 0 else 0.0)

        results[model_name] = {
            "precision": macro_p,
            "recall":    macro_r,
            "f1":        f1,
        }

        print(f"\n  ─── Resultado final [{model_name}] ───")
        print(f"  Macro Precision : {macro_p:.4f}")
        print(f"  Macro Recall    : {macro_r:.4f}")
        print(f"  Macro F1        : {f1:.4f}")

    # ── Tabela comparativa ────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  COMPARATIVO GERAL")
    print(f"{'='*65}")
    print(f"  {'Modelo':<12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}")
    for m, r in sorted(results.items(), key=lambda x: -x[1]["f1"]):
        print(f"  {m:<12}  {r['precision']:>10.4f}  {r['recall']:>8.4f}  {r['f1']:>8.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO / BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    catalog_path = sys.argv[1] if len(sys.argv) > 1 else "./output/catalog.json"

    # Aceita um segundo argumento para rodar apenas um modelo específico
    # Ex: python semantic_context_builder.py ./output/catalog.json e5-base
    model_filter = sys.argv[2] if len(sys.argv) > 2 else None
    models_to_run = [model_filter] if model_filter else None

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         SemanticContextBuilder — Benchmark Comparativo       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nCatálogo  : {catalog_path}")
    print(f"Modelos   : {models_to_run or list(MODEL_REGISTRY.keys())}")
    print(f"top_k     : 3  |  min_similarity: default por modelo  |  strategy: score_gap\n")

    run_benchmark(
        catalog_path=catalog_path,
        models=models_to_run,
        top_k=3,
        verbose_sources=True,
    )
