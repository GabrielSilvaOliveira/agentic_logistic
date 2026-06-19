"""
Teste manual do protótipo — valida cada componente independentemente.

Execute com:
    cd scb_agent
    DB_URL=mysql+pymysql://user:pass@host/db python tests/test_pipeline.py

Saída esperada: cada componente imprime seu resultado para inspeção manual.
Não usa pytest para evitar dependências adicionais no protótipo.
"""
import logging
import os
import sys

# Adiciona o diretório agentic_ai/ ao path para que core/, prompts/ e
# tools/ sejam resolvidos independente do cwd de onde o script é chamado
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("test_pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def ok(msg: str):
    print(f"  ✓ {msg}")


def fail(msg: str):
    print(f"  ✗ {msg}")


# ---------------------------------------------------------------------------
# Teste 1: Catálogo
# ---------------------------------------------------------------------------

def test_catalog():
    section("TESTE 1: Catálogo")
    from core.catalog import CATALOG, get_schema_for_prompt

    assert len(CATALOG) >= 1, "Catálogo vazio"
    ok(f"{len(CATALOG)} fonte(s) no catálogo")

    schema_text = get_schema_for_prompt("sql.materiais")
    assert "CODIGO" in schema_text
    ok("Schema serializado corretamente")
    print(f"\n  Prévia do schema:\n  {schema_text[:300]}...")


# ---------------------------------------------------------------------------
# Teste 2: sql_tool (requer DB_URL configurado)
# ---------------------------------------------------------------------------

def test_sql_tool():
    section("TESTE 2: sql_tool")
    from tools.sql_tool import execute_sql

    db_url = os.getenv("DB_URL")
    if not db_url:
        print("  ⚠ DB_URL não configurado — pulando teste de banco real")
        print("  Testando com query inválida para verificar validação...")

        result = execute_sql("DROP TABLE materiais")
        assert result["status"] == "error"
        ok("Validação de segurança (DROP rejeitado)")

        result = execute_sql("SELECT * FROM tabela_inexistente_xyz")
        assert result["status"] == "error"
        ok("Erro de execução capturado corretamente")
        return

    # Com banco real
    result = execute_sql(
        "SELECT COUNT(*) as total FROM materiais"
    )
    if result["status"] == "success":
        ok(f"Query executada: {result['rows']}")
    elif result["status"] == "error":
        fail(f"Erro: {result['error_message']}")
    else:
        print(f"  Status: {result['status']}")


# ---------------------------------------------------------------------------
# Teste 3: SCB (requer Ollama rodando)
# ---------------------------------------------------------------------------

def test_scb():
    section("TESTE 3: SCB — Seleção semântica")
    from core.llm_client import LLMBackend, LLMConfig
    from core.components import SCBComponent

    config = LLMConfig(
        backend=LLMBackend.OLLAMA,
        model="llama3.2:3b",
        temperature=0.0,
        max_tokens=256,
    )

    try:
        scb = SCBComponent(config)
        query = "Quantos notebooks estão em falta na 3ª Região Militar?"
        sources = scb.select_sources(query)
        ok(f"Fontes selecionadas: {sources}")
        assert isinstance(sources, list)
        assert len(sources) >= 1
    except Exception as e:
        print(f"  ⚠ Ollama não disponível ou modelo não carregado: {e}")
        print("  Configure o Ollama e carregue o modelo llama3.2:3b")


# ---------------------------------------------------------------------------
# Teste 4: Planner (requer Ollama rodando com Qwen2.5:7b)
# ---------------------------------------------------------------------------

def test_planner():
    section("TESTE 4: Planner — Geração de plano")
    from core.llm_client import LLMBackend, LLMConfig
    from core.components import PlannerComponent

    config = LLMConfig(
        backend=LLMBackend.OLLAMA,
        model="qwen2.5:7b",
        temperature=0.0,
        max_tokens=1024,
    )

    try:
        planner = PlannerComponent(config)
        query = "Quantos materiais estão com situação física degradada na unidade LOC031?"
        sources = ["sql.materiais"]
        plan = planner.plan(query, sources)

        ok(f"{len(plan.steps)} passo(s) gerado(s)")
        for step in plan.steps:
            print(f"  Passo {step.id}: {step.operation.value} em {step.source_id}")
            print(f"    Raciocínio: {step.reasoning[:100]}...")

    except Exception as e:
        print(f"  ⚠ Ollama não disponível ou modelo não carregado: {e}")
        print("  Configure o Ollama e carregue o modelo qwen2.5:7b")


# ---------------------------------------------------------------------------
# Teste 5: Pipeline completo (requer todos os serviços)
# ---------------------------------------------------------------------------

def test_full_pipeline():
    section("TESTE 5: Pipeline completo")
    from agent import AgentRegime, LogisticsAgent

    try:
        agent = LogisticsAgent(regime=AgentRegime.LOCAL_LIGHT)
        query = "Quantos notebooks estão em falta na 3ª Região Militar?"

        logger.info("Iniciando pipeline completo...")
        response = agent.query(query)

        print(response.to_display())

        # Asserções básicas
        assert response.query == query
        assert len(response.selected_sources) >= 1
        assert len(response.execution_plan.steps) >= 1
        assert response.answer  # não vazio

        ok("Pipeline completo executado com sucesso")

    except Exception as e:
        print(f"  ⚠ Pipeline falhou: {e}")
        logger.exception("Erro no pipeline completo")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nPROTÓTIPO MÍNIMO — SCB Framework Agêntico Logístico")
    print("Executando testes de validação de componentes...\n")

    tests = [
        ("Catálogo",         test_catalog),
        ("sql_tool",         test_sql_tool),
        ("SCB",              test_scb),
        ("Planner",          test_planner),
        ("Pipeline completo",test_full_pipeline),
    ]

    # Permite rodar um teste específico: python test_pipeline.py catalog
    target = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in tests:
        if target and target.lower() not in name.lower():
            continue
        try:
            fn()
        except AssertionError as e:
            fail(f"Asserção falhou: {e}")
        except Exception as e:
            fail(f"Exceção inesperada: {e}")
            logger.exception(f"Erro em {name}")

    print("\n" + "=" * 60)
    print("  Testes concluídos. Verifique os resultados acima.")
    print("=" * 60 + "\n")
