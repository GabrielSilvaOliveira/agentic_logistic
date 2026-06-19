"""
Modelos de dados do pipeline agêntico logístico.
Todas as estruturas que transitam entre componentes são tipadas aqui.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    SQL = "sql"
    API = "api"
    EXCEL = "excel"


class OperationType(str, Enum):
    FILTER = "filter"
    AGGREGATION = "aggregation"
    LOOKUP = "lookup"
    COMPARISON = "comparison"
    LIST = "list"


class StepStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    EMPTY = "empty"       # query executou, resultado vazio
    ERROR = "error"       # query falhou


# ---------------------------------------------------------------------------
# Plano de execução (output do Planner)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionStep:
    """Um passo do plano gerado pelo Planner."""
    id: int
    source_id: str                  # ex: "sql.vw_comparativo_previsto_existente"
    source_type: SourceType
    operation: OperationType
    reasoning: str                  # raciocínio em NL do Planner
    depends_on: int | None = None   # id do step do qual este depende


@dataclass
class ExecutionPlan:
    """Plano completo gerado pelo Planner para uma consulta."""
    query: str
    steps: list[ExecutionStep]


# ---------------------------------------------------------------------------
# Resultado de execução (output do Executor)
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Resultado da execução de um step pelo Executor."""
    step_id: int
    source_id: str
    status: StepStatus
    data: list[dict[str, Any]] = field(default_factory=list)
    generated_query: str = ""       # SQL gerado, endpoint chamado, etc.
    error_message: str = ""
    row_count: int = 0

    def __post_init__(self):
        self.row_count = len(self.data)


# ---------------------------------------------------------------------------
# Resposta final (output do Synthesizer)
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """Resposta completa do agente ao decisor."""
    query: str

    # Camada 1 — seleção (do SCB)
    selected_sources: list[str]

    # Camada 2 — execução (do Executor)
    execution_plan: ExecutionPlan
    step_results: list[StepResult]

    # Camada 3 — resposta final (do Synthesizer)
    answer: str
    cited_sources: list[str]
    has_unanswered_steps: bool = False

    def to_display(self) -> str:
        """Formata a resposta nas três camadas para exibição."""
        lines = []

        lines.append("=" * 60)
        lines.append(f"CONSULTA: {self.query}")
        lines.append("=" * 60)

        lines.append("\n── CAMADA 1: FONTES SELECIONADAS ──")
        for src in self.selected_sources:
            lines.append(f"  • {src}")

        lines.append("\n── CAMADA 2: PLANO DE EXECUÇÃO ──")
        for step in self.execution_plan.steps:
            result = next(
                (r for r in self.step_results if r.step_id == step.id), None
            )
            status_icon = {
                StepStatus.SUCCESS: "✓",
                StepStatus.EMPTY:   "∅",
                StepStatus.ERROR:   "✗",
                StepStatus.PENDING: "?",
            }.get(result.status if result else StepStatus.PENDING, "?")

            lines.append(f"\n  Passo {step.id} [{status_icon}]")
            lines.append(f"  Fonte:     {step.source_id}")
            lines.append(f"  Operação:  {step.operation.value}")
            lines.append(f"  Raciocínio: {step.reasoning}")
            if result:
                lines.append(f"  Query:     {result.generated_query}")
                lines.append(f"  Registros: {result.row_count}")
                if result.error_message:
                    lines.append(f"  Erro:      {result.error_message}")

        lines.append("\n── CAMADA 3: RESPOSTA FINAL ──")
        lines.append(f"\n{self.answer}")
        lines.append(
            f"\nFonte(s) consultada(s): {', '.join(self.cited_sources)}"
        )
        lines.append("=" * 60)

        return "\n".join(lines)
