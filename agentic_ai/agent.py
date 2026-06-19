"""
Pipeline agêntico logístico — orquestrador principal.

Este módulo coordena os quatro componentes (SCB → Planner → Executor →
Synthesizer) e é o único ponto de entrada para consultas externas.

Uso básico:
    from agent import LogisticsAgent, AgentRegime

    agent = LogisticsAgent(regime=AgentRegime.LOCAL_LIGHT)
    response = agent.query("Quantos notebooks estão em falta na 3ª RM?")
    print(response.to_display())
"""
from __future__ import annotations

import logging
import time
from enum import Enum

from core.components import (
    ExecutorComponent,
    PlannerComponent,
    SCBComponent,
    SynthesizerComponent,
)
from core.llm_client import make_local_light_configs, make_oracle_configs
from core.models import AgentResponse

logger = logging.getLogger(__name__)


class AgentRegime(str, Enum):
    LOCAL_LIGHT = "local_light"   # Llama 3.2 3B (SCB) + Qwen2.5 7B
    ORACLE = "oracle"             # Llama 3.2 3B (SCB) + GPT-4o


class LogisticsAgent:
    """
    Agente de apoio à decisão logística.

    Encapsula o pipeline completo SCB → Planner → Executor → Synthesizer.
    O regime (local vs oracle) é configurado na inicialização e determina
    qual modelo é usado em cada componente.
    """

    def __init__(self, regime: AgentRegime = AgentRegime.LOCAL_LIGHT):
        self.regime = regime
        configs = (
            make_local_light_configs()
            if regime == AgentRegime.LOCAL_LIGHT
            else make_oracle_configs()
        )

        self.scb = SCBComponent(configs["scb"])
        self.planner = PlannerComponent(configs["planner"])
        self.executor = ExecutorComponent(
            query_gen_config=configs["query_generator"],
            max_retries=1,
        )
        self.synthesizer = SynthesizerComponent(configs["synthesizer"])

        logger.info(f"LogisticsAgent inicializado — regime: {regime.value}")

    def query(self, user_query: str) -> AgentResponse:
        """
        Processa uma consulta end-to-end e retorna AgentResponse.

        O AgentResponse contém:
        - Camada 1: fontes selecionadas pelo SCB
        - Camada 2: plano de execução + resultados brutos
        - Camada 3: resposta em NL + fontes citadas
        """
        logger.info(f"Consulta recebida: {user_query!r}")
        t_start = time.perf_counter()

        # ── Camada 1: SCB ──────────────────────────────────────────────
        t0 = time.perf_counter()
        selected_sources = self.scb.select_sources(user_query)
        t_scb = time.perf_counter() - t0
        logger.info(f"SCB concluído em {t_scb:.2f}s → {selected_sources}")

        # ── Camada 2a: Planner ─────────────────────────────────────────
        t0 = time.perf_counter()
        plan = self.planner.plan(user_query, selected_sources)
        t_planner = time.perf_counter() - t0
        logger.info(f"Planner concluído em {t_planner:.2f}s → {len(plan.steps)} passo(s)")

        # ── Camada 2b: Executor ────────────────────────────────────────
        t0 = time.perf_counter()
        step_results = self.executor.execute_plan(plan)
        t_executor = time.perf_counter() - t0
        logger.info(f"Executor concluído em {t_executor:.2f}s")

        # ── Camada 3: Synthesizer ──────────────────────────────────────
        t0 = time.perf_counter()
        answer, cited_sources = self.synthesizer.synthesize(
            user_query, plan, step_results
        )
        t_synthesizer = time.perf_counter() - t0
        logger.info(f"Synthesizer concluído em {t_synthesizer:.2f}s")

        t_total = time.perf_counter() - t_start
        logger.info(
            f"Pipeline completo em {t_total:.2f}s "
            f"(SCB:{t_scb:.2f} Planner:{t_planner:.2f} "
            f"Executor:{t_executor:.2f} Synth:{t_synthesizer:.2f})"
        )

        # Verifica se algum passo falhou (para o decisor)
        from core.models import StepStatus
        has_unanswered = any(
            r.status in (StepStatus.ERROR, StepStatus.EMPTY)
            for r in step_results
        )

        return AgentResponse(
            query=user_query,
            selected_sources=selected_sources,
            execution_plan=plan,
            step_results=step_results,
            answer=answer,
            cited_sources=cited_sources,
            has_unanswered_steps=has_unanswered,
        )
