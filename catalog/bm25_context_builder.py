# catalog/bm25_context_builder.py
import re
from rank_bm25 import BM25Okapi
from catalog.semantic_context_builder_v2 import SemanticContextBuilderV2

class BM25ContextBuilder(SemanticContextBuilderV2):
    """
    Reaproveita _build_documents / _doc_from_source (mesmos textos usados
    pelos embeddings, incluindo embedding_hint por versão v1/v2/v3) e
    _cut_by_score_gap. Substitui apenas a etapa de scoring denso por BM25.
    """

    def __init__(self, catalog_path, top_k=3, min_similarity=0.30,
                 selection_strategy="score_gap", cache_dir=None):
        # model_name="minilm" é só placeholder para satisfazer o __init__ do
        # pai — o encoder do MiniLM nunca é carregado porque sobrescrevemos
        # rebuild_index() e _select_sources() por completo.
        super().__init__(
            catalog_path=catalog_path,
            model_name="minilm",
            top_k=top_k,
            min_similarity=min_similarity,
            selection_strategy=selection_strategy,
            cache_dir=cache_dir,
        )
        self._bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def rebuild_index(self, force: bool = False, verbose: bool = False):
        # BM25 é rápido o bastante para não precisar de cache em pickle
        documents, source_ids = self._build_documents(verbose=verbose)
        tokenized_docs = [self._tokenize(d) for d in documents]
        self._bm25 = BM25Okapi(tokenized_docs)
        self._source_ids = source_ids
        self._doc_texts = documents
        if verbose:
            print(f"[BM25] Índice construído: {len(source_ids)} fontes")

    def _select_sources(self, query: str, verbose: bool = False) -> dict:
        if self._bm25 is None:
            self.rebuild_index(verbose=verbose)

        raw_scores = self._bm25.get_scores(self._tokenize(query))
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        # Normalização por consulta: mantém a faixa [0,1] esperada pelo
        # score_gap/threshold e pela grade de calibração de run_benchmark.py
        norm_scores = [s / max_score for s in raw_scores]

        ranked = sorted(enumerate(norm_scores), key=lambda x: x[1], reverse=True)

        if self.selection_strategy == "score_gap":
            candidates = self._cut_by_score_gap(ranked)
            candidates = [(i, s) for i, s in candidates if s >= self.min_similarity]
        else:
            candidates = [(i, s) for i, s in ranked[:self.top_k] if s >= self.min_similarity]

        selected_ids = [self._source_ids[i] for i, _ in candidates]

        if not selected_ids:
            sources = self.catalog.get("sources", {})
            return {
                "sql": list(sources.get("sql", {}).get("tables", {}).keys()),
                "api": list(sources.get("api", {}).get("endpoints", {}).keys()),
                "spreadsheets": list(sources.get("spreadsheets", {}).get("files", {}).keys()),
            }

        result = {"sql": [], "api": [], "spreadsheets": []}
        for sid in selected_ids:
            t, key = sid.split(".", 1)
            if t in result:
                result[t].append(key)
        return result