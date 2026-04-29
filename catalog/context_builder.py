"""
catalog/context_builder.py
─────────────────────────────────────────────────────────────────────────────
Context Builder — transforma o catálogo JSON em contexto otimizado para o LLM.

Responsabilidades:
  1. Carregar o catálogo do disco
  2. Aplicar context_hints (regras manuais de roteamento)
  3. Serializar as fontes relevantes em Markdown comprimido
  4. Retornar o bloco de contexto pronto para injetar no prompt do agente

Estratégia de injeção (v1 — sem embedding):
  - Aplica context_hints para pré-selecionar fontes
  - Se nenhuma hint se aplica: injeta catálogo completo em modo comprimido
  - Formato: Markdown com seções por fonte (SQL / API / Planilha)

Extensão futura (v2 — com embedding):
  - Substituir _select_by_hints() por _select_by_embedding()
  - Trocar JSON local por ChromaDB/SQLite-vec
  - Interface pública não muda — o agente não precisa saber

Uso:
  cb = ContextBuilder("./output/catalog.json")
  context = cb.build_context("Quais distribuições estão pendentes para LOC002?")
  # → string Markdown pronta para o system prompt do LLM
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class ContextBuilder:

    def __init__(self, catalog_path: str):
        self.catalog_path = Path(catalog_path)
        self._catalog: Optional[dict] = None

    @property
    def catalog(self) -> dict:
        """Lazy load — carrega o catálogo uma vez e mantém em memória."""
        if self._catalog is None:
            if not self.catalog_path.exists():
                raise FileNotFoundError(
                    f"Catálogo não encontrado: {self.catalog_path}\n"
                    f"Execute os extratores primeiro."
                )
            with open(self.catalog_path, encoding="utf-8") as f:
                self._catalog = json.load(f)
        return self._catalog

    def reload(self):
        """Força recarregamento do catálogo do disco (útil em atualizações)."""
        self._catalog = None

    # ── Interface pública ─────────────────────────────────────────────────────

    def build_context(self, user_query: str,
                      max_sample_rows: int = 2,
                      verbose: bool = False) -> str:
        """
        Ponto de entrada principal.
        Retorna string Markdown pronta para injetar no prompt do LLM.

        Parâmetros:
          user_query     : pergunta do usuário em linguagem natural
          max_sample_rows: quantas linhas de amostra incluir por fonte
          verbose        : se True, imprime quais fontes foram selecionadas
        """
        selected = self._select_sources(user_query, verbose)
        return self._render_markdown(selected, user_query, max_sample_rows)

    def build_full_context(self, max_sample_rows: int = 1) -> str:
        """
        Retorna o catálogo completo em Markdown comprimido.
        Útil para o primeiro turno de uma conversa ou para debugging.
        """
        sources = self.catalog.get("sources", {})
        return self._render_markdown(
            {
                "sql":          list(sources.get("sql", {}).get("tables", {}).keys()),
                "api":          list(sources.get("api", {}).get("endpoints", {}).keys()),
                "spreadsheets": list(sources.get("spreadsheets", {}).get("files", {}).keys()),
            },
            query="(contexto completo)",
            max_sample_rows=max_sample_rows,
        )

    # ── Seleção de fontes por context_hints ───────────────────────────────────

    def _select_sources(self, query: str, verbose: bool = False) -> dict:
        """
        Aplica context_hints ao query para selecionar fontes relevantes.
        Retorna dict com listas de chaves por tipo de fonte.
        """
        query_lower = query.lower()
        hints = self.catalog.get("context_hints", {}).get("rules", [])

        selected = {"sql": set(), "api": set(), "spreadsheets": set()}
        matched_rules = []

        for rule in hints:
            keywords = [kw.lower() for kw in rule.get("trigger_keywords", [])
                        if "[MANUAL]" not in kw]
            if any(kw in query_lower for kw in keywords):
                matched_rules.append(rule["id"])
                for source_ref in rule.get("include_sources", []):
                    self._add_source_ref(selected, source_ref)
                for source_ref in rule.get("complement_with", []):
                    self._add_source_ref(selected, source_ref)

        # Se nenhuma hint se aplicou: inclui tudo (fallback_rule)
        if not any(selected.values()):
            if verbose:
                print("[ContextBuilder] Nenhuma hint aplicável → modo completo")
            sources = self.catalog.get("sources", {})
            selected["sql"]          = set(sources.get("sql", {}).get("tables", {}).keys())
            selected["api"]          = set(sources.get("api", {}).get("endpoints", {}).keys())
            selected["spreadsheets"] = set(sources.get("spreadsheets", {}).get("files", {}).keys())
        elif verbose:
            print(f"[ContextBuilder] Hints aplicadas: {matched_rules}")
            print(f"  SQL          : {selected['sql']}")
            print(f"  API          : {selected['api']}")
            print(f"  Spreadsheets : {selected['spreadsheets']}")

        return {k: list(v) for k, v in selected.items()}

    @staticmethod
    def _add_source_ref(selected: dict, ref: str):
        """Converte 'sql.materiais' → selected['sql'].add('materiais')."""
        parts = ref.split(".", 1)
        if len(parts) == 2:
            source_type, key = parts
            if source_type in selected:
                selected[source_type].add(key)

    # ── Renderização em Markdown ──────────────────────────────────────────────

    def _render_markdown(self, selected: dict, query: str,
                         max_sample_rows: int) -> str:
        """
        Serializa as fontes selecionadas em Markdown otimizado para LLM.

        Princípios de compressão aplicados:
          - Omite colunas com null_rate = 0 e sem FK (reduz ruído)
          - Limita amostra a max_sample_rows
          - Usa formato tabular compacto para colunas
          - Remove campos [MANUAL] não preenchidos do output
        """
        blocks = [
            "# CONTEXTO DO CATÁLOGO DE DADOS\n",
            f"_Fontes selecionadas para responder: \"{query}\"_\n",
        ]

        sources = self.catalog.get("sources", {})

        # ── SQL ───────────────────────────────────────────────────────────────
        sql_tables    = sources.get("sql", {}).get("tables", {})
        selected_sql  = selected.get("sql", [])
        if selected_sql:
            blocks.append("## 📊 BANCO DE DADOS (SQL — MySQL)\n")
            for table_name in selected_sql:
                if table_name not in sql_tables:
                    continue
                blocks.append(self._render_sql_table(
                    table_name, sql_tables[table_name], max_sample_rows))

        # ── API ───────────────────────────────────────────────────────────────
        api_endpoints    = sources.get("api", {}).get("endpoints", {})
        selected_api     = selected.get("api", [])
        api_base_url     = sources.get("api", {}).get("base_url", "")
        if selected_api:
            blocks.append("## 🔌 API REST\n")
            for ep_key in selected_api:
                if ep_key not in api_endpoints:
                    continue
                blocks.append(self._render_api_endpoint(
                    ep_key, api_endpoints[ep_key], api_base_url))

        # ── Planilhas ─────────────────────────────────────────────────────────
        spreadsheet_files = sources.get("spreadsheets", {}).get("files", {})
        selected_sheets   = selected.get("spreadsheets", [])
        if selected_sheets:
            blocks.append("## 📄 PLANILHAS\n")
            for file_key in selected_sheets:
                if file_key not in spreadsheet_files:
                    continue
                blocks.append(self._render_spreadsheet(
                    file_key, spreadsheet_files[file_key], max_sample_rows))

        # ── Rodapé com instruções para o LLM ─────────────────────────────────
        blocks.append(self._render_footer())

        return "\n".join(blocks)

    # ── Renderizadores por tipo de fonte ──────────────────────────────────────

    def _render_sql_table(self, name: str, meta: dict,
                          max_sample: int) -> str:
        lines = [f"### Tabela: `{name}`"]

        desc = meta.get("description", "")
        role = meta.get("business_role", "")
        when = meta.get("when_to_use", "")
        if desc and "[MANUAL]" not in desc:
            lines.append(f"**Descrição:** {desc}")
        if role and "[MANUAL]" not in role:
            lines.append(f"**Papel:** {role}")
        if when and "[MANUAL]" not in when:
            lines.append(f"**Quando usar:** {when}")

        lines.append(f"**Linhas:** {meta.get('row_count', '?'):,}  |  "
                     f"**Colunas:** {meta.get('column_count', '?')}")

        # Tabela de colunas comprimida
        cols = meta.get("columns", {})
        if cols:
            lines.append("\n| Coluna | Tipo | PK | Nulos | Descrição |")
            lines.append("|--------|------|----|-------|-----------|")
            for col_name, col_meta in cols.items():
                pk   = "✓" if col_meta.get("is_pk") else ""
                null = f"{col_meta.get('null_rate', 0):.0%}" if col_meta.get("null_rate") else "0%"
                desc = col_meta.get("description", "")
                desc = "" if "[MANUAL]" in desc else desc
                ctype = col_meta.get("type", "")[:20]
                lines.append(f"| `{col_name}` | {ctype} | {pk} | {null} | {desc} |")

        # FKs
        fks = meta.get("foreign_keys", [])
        if fks:
            lines.append("\n**Relacionamentos:**")
            for fk in fks:
                lines.append(f"- `{fk['columns']}` → `{fk['references_table']}.{fk['references_columns']}`")

        # Amostra
        sample = meta.get("sample_rows", [])[:max_sample]
        if sample:
            lines.append(f"\n**Amostra ({len(sample)} linha(s)):**")
            lines.append("```json")
            lines.append(json.dumps(sample, ensure_ascii=False,
                                    indent=2, default=str))
            lines.append("```")

        lines.append("")
        return "\n".join(lines)

    def _render_api_endpoint(self, key: str, meta: dict,
                             base_url: str) -> str:
        lines = [f"### Endpoint: `{key}`"]

        path   = meta.get("path", "")
        method = meta.get("method", "GET")
        lines.append(f"**URL:** `{method} {base_url}{path}`")

        for field in ("description", "business_role", "when_to_use", "when_NOT_to_use"):
            val = meta.get(field, "")
            if val and "[MANUAL]" not in val:
                label = {
                    "description":   "Descrição",
                    "business_role": "Papel",
                    "when_to_use":   "Quando usar",
                    "when_NOT_to_use": "Quando NÃO usar",
                }[field]
                lines.append(f"**{label}:** {val}")

        # Parâmetros
        params = (meta.get("parameters", {}).get("query_params", {}))
        real_params = {k: v for k, v in params.items() if "[MANUAL]" not in k}
        if real_params:
            lines.append("\n**Parâmetros:**")
            for p_name, p_meta in real_params.items():
                req  = "obrigatório" if p_meta.get("required") else "opcional"
                desc = p_meta.get("description", "")
                lines.append(f"- `{p_name}` ({p_meta.get('type','?')}, {req}): {desc}")

        # Campos da resposta
        response_schema = meta.get("response_schema", {})
        fields = response_schema.get("result_object_fields", {})
        real_fields = {k: v for k, v in fields.items() if "[MANUAL]" not in k}
        if real_fields:
            lines.append("\n**Campos retornados:**")
            lines.append("| Campo | Tipo | Exemplo | Descrição |")
            lines.append("|-------|------|---------|-----------|")
            for f_name, f_meta in real_fields.items():
                f_type = f_meta.get("type", "?")
                f_ex   = str(f_meta.get("example", ""))[:30]
                f_desc = f_meta.get("description", "")
                f_desc = "" if "[MANUAL]" in f_desc else f_desc
                lines.append(f"| `{f_name}` | {f_type} | {f_ex} | {f_desc} |")

        # Confiança base
        conf = meta.get("data_quality", {}).get("confidence_base")
        if conf:
            lines.append(f"\n**Confiança base:** {conf:.0%}")

        lines.append("")
        return "\n".join(lines)

    def _render_spreadsheet(self, key: str, meta: dict,
                             max_sample: int) -> str:
        lines = [f"### Planilha: `{meta.get('filename', key)}`"]

        for field in ("description", "business_role", "when_to_use", "when_NOT_to_use"):
            val = meta.get(field, "")
            if val and "[MANUAL]" not in val:
                label = {
                    "description":     "Descrição",
                    "business_role":   "Papel",
                    "when_to_use":     "Quando usar",
                    "when_NOT_to_use": "Quando NÃO usar",
                }[field]
                lines.append(f"**{label}:** {val}")

        for sheet_name, sheet_meta in meta.get("sheets", {}).items():
            lines.append(f"\n**Aba:** `{sheet_name}`  |  "
                         f"**Linhas:** {sheet_meta.get('row_count', '?'):,}  |  "
                         f"**Colunas:** {sheet_meta.get('column_count', '?')}")

            cols = sheet_meta.get("columns", {})
            if cols:
                lines.append("\n| Coluna | Tipo | Nulos | Descrição |")
                lines.append("|--------|------|-------|-----------|")
                for col_name, col_meta in cols.items():
                    null = f"{col_meta.get('null_rate', 0):.0%}"
                    desc = col_meta.get("description", "")
                    desc = "" if "[MANUAL]" in desc else desc
                    ctype = col_meta.get("type", "")[:20]
                    lines.append(f"| `{col_name}` | {ctype} | {null} | {desc} |")

            sample = sheet_meta.get("sample_rows", [])[:max_sample]
            if sample:
                lines.append(f"\n**Amostra:**")
                lines.append("```json")
                lines.append(json.dumps(sample, ensure_ascii=False,
                                        indent=2, default=str))
                lines.append("```")

        conf = meta.get("data_quality", {}).get("confidence_base")
        if conf:
            lines.append(f"\n**Confiança base:** {conf:.0%}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_footer() -> str:
        return (
            "---\n"
            "## INSTRUÇÕES PARA USO DO CATÁLOGO\n\n"
            "- Use as tabelas SQL para dados estruturados de materiais, catalogo de equipamentos e unidades\n"
            "- Use a API REST para dados de recebimento e distribuição\n"
            "- Use a planilha de contratos para informações sobre itens adquiridos e de fornecedores\n"
            "- Se uma fonte retornar dados incompletos, sinalize na resposta ao usuário\n"
            "- Priorize SQL como fonte authoritative quando houver sobreposição de dados\n"
            "- Ao gerar queries SQL, use apenas operações SELECT\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO / TESTE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    catalog_path = sys.argv[1] if len(sys.argv) > 1 else "./output/catalog.json"

    print(f"Carregando catálogo: {catalog_path}\n")
    cb = ContextBuilder(catalog_path)

    test_queries = [
        "Quais materiais estão disponíveis para distribuição?",
        "Mostre as distribuições pendentes para LOC002",
        "Qual o valor total dos contratos vigentes?",
        "Quais equipamentos estão com a situação física: \"indisponível\" ?",
    ]

    for query in test_queries:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        context = cb.build_context(query, verbose=True)
        # Mostra só as primeiras linhas para não poluir o terminal
        preview = "\n".join(context.split("\n")[:25])
        print(preview)
        print(f"... [{len(context)} chars total]\n")
