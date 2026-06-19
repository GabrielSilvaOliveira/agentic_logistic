# Agentic Logistic

Projeto de dissertação de mestrado sobre Agentic AI para suporte a decisao logistica a partir de dados heterogeneos (SQL, APIs, planilhas e documentos).

## Objetivo

Construir um agente capaz de:

1. Planejar consultas com base em linguagem natural.
2. Selecionar e executar ferramentas apropriadas por fonte de dados.
3. Consolidar respostas com rastreabilidade, explicabilidade e avaliacao de qualidade.

## Estado Atual do Repositorio

O repositorio possui dois blocos principais no momento:

1. Estrutura base do projeto (modulos em uso).
2. Novo modulo `agentic_ai/` (pipeline minimo funcional), que sera incorporado gradualmente aos demais componentes.

## Estrutura Atual (resumo)

```text
agentic_logistic/
├── main.py
├── test_gemini.py
├── agent/
│   ├── planner.py
│   ├── execution.py
│   └── reflection.py
├── llm/
│   ├── base_llm.py
│   ├── gemini_provider.py
│   └── azure_provider.py
├── tools/
│   ├── base_tool.py
│   ├── sql_tool.py
│   ├── api_tool.py
│   ├── spreadsheet_tool.py
│   └── document_tool.py
├── catalog/
│   ├── build_catalog.py
│   ├── build_catalog_v2.py
│   ├── semantic_context_builder.py
│   ├── semantic_context_builder_v2.py
│   ├── extractors/
│   ├── output/
│   └── results/
├── data_masking/
├── data_synthesis/
└── agentic_ai/                # modulo novo em convergencia
     ├── agent.py
     ├── core/
     │   ├── models.py
     │   ├── llm_client.py
     │   ├── catalog.py        # le catalog/output/catalog_v3.json
     │   └── components.py
     ├── prompts/
     │   └── templates.py
     ├── tools/
     │   └── sql_tool.py
     ├── test_pipeline.py
     └── requirements.txt
```

## Arquitetura de Referencia

```text
Usuario
  -> Interface/API
  -> Planner
  -> Tool Router
       -> SQL Tool
       -> API Tool
       -> Spreadsheet Tool
       -> Document Tool
  -> Contexto Semantico (Catalogo)
  -> Executor
  -> Reflexao / Validacao
  -> Resposta com rastreabilidade
```

## Sobre o modulo novo agentic_ai

O diretorio `agentic_ai/` representa uma linha de evolucao recente com pipeline end-to-end mais enxuto (SCB -> Planner -> Executor -> Synthesizer).

Neste momento:

1. Ele permanece como modulo separado para testes e comparacoes.
2. A intencao e unificar suas pecas com a estrutura principal do repositorio.
3. O README principal passa a refletir essa fase de transicao.

## Proximos passos de integracao

1. Consolidar modelos e contratos de dados entre `agent/`, `tools/` e `agentic_ai/`.
2. Unificar cliente de LLM e estrategia de provedores.
3. Centralizar catalogo semantico e camada de execucao SQL.
4. Padronizar testes de pipeline e benchmarks em um unico fluxo.

## Observacao

Este README descreve o estado real do repositorio hoje e o direcionamento de convergencia do modulo novo, sem alterar ainda a organizacao fisica dos codigos.