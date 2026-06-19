"""
Componentes do pipeline agêntico logístico.

Cada componente é uma classe independente com uma única
responsabilidade. O pipeline é orquestrado em agent.py.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from core.catalog import get_schema_for_prompt, CATALOG
from core.llm_client import LLMClient, LLMConfig
from core.models import (
    ExecutionPlan,
    ExecutionStep,
    OperationType,
    SourceType,
    StepResult,
    StepStatus,
)
from prompts.templates import (
    PLANNER_SYSTEM,
    PLANNER_USER_TEMPLATE,
    QUERY_GENERATOR_SYSTEM,
    QUERY_GENERATOR_USER_TEMPLATE,
    SCB_SYSTEM,
    SCB_USER_TEMPLATE,
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_USER_TEMPLATE,
)
from tools.sql_tool import execute_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. SCB — Semantic Context Builder
# ---------------------------------------------------------------------------

class SCBComponent:
    """
    Seleciona as fontes relevantes para uma consulta (zero-shot).

    No protótipo usa Llama 3.2 3B via Ollama — configuração validada
    no artigo do SCB com F1=0.775.
    """

    def __init__(self, config: LLMConfig):
        self.client = LLMClient(config)

    def select_sources(self, query: str) -> list[str]:
        """
        Retorna lista de source_ids relevantes para a consulta.
        Fallback: todas as fontes do catálogo se o modelo falhar.
        """
        # Serializa hints do catálogo comprimido (apenas embedding_hint)
        catalog_hints = "\n".join(
            f"- {sid}: {src.embedding_hint}"
            for sid, src in CATALOG.items()
        )
        user = SCB_USER_TEMPLATE.format(
            query=query,
            catalog_hints=catalog_hints,
        )

        try:
            result = self.client.chat_json(
                system=SCB_SYSTEM,
                user=user,
                json_schema={"type": "object", "properties": {
                    "sources": {"type": "array", "items": {"type": "string"}}
                }, "required": ["sources"]},
            )
            sources = result.get("sources", [])
            # Filtra IDs que existem no catálogo
            valid = [s for s in sources if s in CATALOG]
            if valid:
                logger.info(f"SCB selecionou: {valid}")
                return valid
        except Exception as e:
            logger.warning(f"SCB falhou ({e}), usando fallback (todas as fontes)")

        # Fallback: todas as fontes disponíveis
        return list(CATALOG.keys())


# ---------------------------------------------------------------------------
# 2. Planner
# ---------------------------------------------------------------------------

# Schema JSON para structured output do Planner
_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":          {"type": "integer"},
                    "source_id":   {"type": "string"},
                    "source_type": {"type": "string",
                                   "enum": ["sql", "api", "excel"]},
                    "operation":   {"type": "string",
                                   "enum": ["filter", "aggregation",
                                            "lookup", "comparison", "list"]},
                    "reasoning":   {"type": "string"},
                    "depends_on":  {"type": ["integer", "null"]},
                },
                "required": ["id", "source_id", "source_type",
                             "operation", "reasoning"],
            },
        }
    },
    "required": ["steps"],
}


class PlannerComponent:
    """
    Gera o plano de execução (NL + JSON estruturado) para uma consulta.
    """

    def __init__(self, config: LLMConfig):
        self.client = LLMClient(config)

    def plan(self, query: str, selected_sources: list[str]) -> ExecutionPlan:
        """
        Retorna ExecutionPlan com os passos para responder a consulta.
        """
        # Resume as fontes selecionadas para o prompt
        sources_summary = "\n".join(
            f"- {sid}: {CATALOG[sid].embedding_hint}"
            for sid in selected_sources
            if sid in CATALOG
        )
        user = PLANNER_USER_TEMPLATE.format(
            query=query,
            sources_summary=sources_summary,
        )

        raw = self.client.chat_json(
            system=PLANNER_SYSTEM,
            user=user,
            json_schema=_PLAN_SCHEMA,
        )

        steps = []
        for step_data in raw.get("steps", []):
            steps.append(ExecutionStep(
                id=step_data["id"],
                source_id=step_data["source_id"],
                source_type=SourceType(step_data["source_type"]),
                operation=OperationType(step_data["operation"]),
                reasoning=step_data["reasoning"],
                depends_on=step_data.get("depends_on"),
            ))

        logger.info(f"Planner gerou {len(steps)} passo(s)")
        return ExecutionPlan(query=query, steps=steps)


# ---------------------------------------------------------------------------
# 3. Executor
# ---------------------------------------------------------------------------

class ExecutorComponent:
    """
    Executa cada passo do plano em duas sub-camadas:
    1. Query Generator (LLM): traduz o step em parâmetros da tool
    2. Tool Runner (código): executa a tool com os parâmetros gerados
    """

    def __init__(self, query_gen_config: LLMConfig, max_retries: int = 1):
        self.qg_client = LLMClient(query_gen_config)
        self.max_retries = max_retries

    def execute_plan(self, plan: ExecutionPlan) -> list[StepResult]:
        """Executa todos os passos do plano, respeitando dependências."""
        results: list[StepResult] = []
        results_by_id: dict[int, StepResult] = {}

        for step in plan.steps:
            # Aguarda passo do qual depende (se houver)
            if step.depends_on and step.depends_on not in results_by_id:
                logger.warning(
                    f"Passo {step.id} depende do passo {step.depends_on} "
                    f"que ainda não foi executado. Executando assim mesmo."
                )

            result = self._execute_step(step, results_by_id)
            results.append(result)
            results_by_id[step.id] = result

        return results

    def _execute_step(
        self,
        step: ExecutionStep,
        previous_results: dict[int, StepResult],
    ) -> StepResult:
        """Executa um único step com retry em caso de erro."""

        if step.source_type == SourceType.SQL:
            return self._execute_sql_step(step, previous_results)

        # Placeholder para API e Excel — a implementar nas próximas iterações
        logger.warning(
            f"Tipo de fonte '{step.source_type}' ainda não implementado "
            f"no protótipo. Retornando status pending."
        )
        return StepResult(
            step_id=step.id,
            source_id=step.source_id,
            status=StepStatus.ERROR,
            error_message=f"Executor para '{step.source_type}' não implementado.",
        )

    def _execute_sql_step(
        self,
        step: ExecutionStep,
        previous_results: dict[int, StepResult],
    ) -> StepResult:
        """
        Sub-camada 1: Query Generator gera o SQL.
        Sub-camada 2: sql_tool executa.
        Retry em caso de erro de execução.
        """
        schema_text = get_schema_for_prompt(step.source_id)

        # Contexto adicional de passos anteriores se houver dependência
        context = ""
        if step.depends_on and step.depends_on in previous_results:
            prev = previous_results[step.depends_on]
            if prev.data:
                context = (
                    f"\n\nCONTEXTO DO PASSO {step.depends_on}:\n"
                    f"{json.dumps(prev.data[:3], ensure_ascii=False)}"
                )

        error_context = ""
        for attempt in range(self.max_retries + 1):
            # --- Sub-camada 1: Query Generator (LLM) ---
            user = QUERY_GENERATOR_USER_TEMPLATE.format(
                reasoning=step.reasoning + context,
                operation=step.operation.value,
                schema=schema_text,
            ) + error_context

            try:
                qg_result = self.qg_client.chat_json(
                    system=QUERY_GENERATOR_SYSTEM,
                    user=user,
                    json_schema={
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                )
                sql = qg_result.get("sql", "").strip()
                if not sql:
                    raise ValueError("Query Generator retornou SQL vazio.")
            except Exception as e:
                logger.error(f"Query Generator falhou (tentativa {attempt}): {e}")
                return StepResult(
                    step_id=step.id,
                    source_id=step.source_id,
                    status=StepStatus.ERROR,
                    error_message=f"Geração de query falhou: {e}",
                )

            logger.info(f"SQL gerado (tentativa {attempt}): {sql}")

            # --- Sub-camada 2: sql_tool (código determinístico) ---
            from core.catalog import get_source
            connection_key = get_source(step.source_id).connection_key
            tool_result = execute_sql(sql, connection_key)

            status_map = {
                "success": StepStatus.SUCCESS,
                "empty":   StepStatus.EMPTY,
                "error":   StepStatus.ERROR,
            }
            status = status_map.get(tool_result["status"], StepStatus.ERROR)

            if status == StepStatus.ERROR and attempt < self.max_retries:
                # Inclui o erro no próximo prompt para que o modelo corrija
                error_context = (
                    f"\n\nATENÇÃO: A query anterior falhou com o erro:\n"
                    f"{tool_result['error_message']}\n"
                    f"Query anterior: {sql}\n"
                    f"Corrija a query considerando este erro."
                )
                logger.warning(
                    f"SQL falhou, tentando novamente "
                    f"({attempt + 1}/{self.max_retries}): "
                    f"{tool_result['error_message']}"
                )
                continue

            return StepResult(
                step_id=step.id,
                source_id=step.source_id,
                status=status,
                data=tool_result["rows"],
                generated_query=sql,
                error_message=tool_result["error_message"],
            )

        # Não deve chegar aqui, mas por segurança:
        return StepResult(
            step_id=step.id,
            source_id=step.source_id,
            status=StepStatus.ERROR,
            error_message="Número máximo de tentativas atingido.",
        )


# ---------------------------------------------------------------------------
# 4. Synthesizer
# ---------------------------------------------------------------------------

class SynthesizerComponent:
    """
    Sintetiza os resultados brutos em resposta em linguagem natural,
    citando as fontes consultadas.
    """

    def __init__(self, config: LLMConfig):
        self.client = LLMClient(config)

    def synthesize(
        self,
        query: str,
        plan: ExecutionPlan,
        step_results: list[StepResult],
    ) -> tuple[str, list[str]]:
        """
        Retorna (resposta_NL, fontes_citadas).
        """
        # Monta resumo dos resultados para o prompt
        results_parts = []
        cited_sources = []

        for step, result in zip(plan.steps, step_results):
            if result.status == StepStatus.SUCCESS:
                cited_sources.append(result.source_id)
                # Limita dados para não explodir o contexto
                data_preview = json.dumps(
                    result.data[:10], ensure_ascii=False, indent=2
                )
                results_parts.append(
                    f"Passo {step.id} ({step.source_id}):\n"
                    f"Query: {result.generated_query}\n"
                    f"Resultado ({result.row_count} registros):\n{data_preview}"
                )
            elif result.status == StepStatus.EMPTY:
                results_parts.append(
                    f"Passo {step.id} ({step.source_id}): "
                    f"consulta retornou resultado vazio."
                )
            else:
                results_parts.append(
                    f"Passo {step.id} ({step.source_id}): "
                    f"falha na consulta — {result.error_message}"
                )

        results_summary = "\n\n".join(results_parts)

        user = SYNTHESIZER_USER_TEMPLATE.format(
            query=query,
            results_summary=results_summary,
        )

        answer = self.client.chat(
            system=SYNTHESIZER_SYSTEM,
            user=user,
        )

        return answer, list(set(cited_sources))
