# Projeto para dissertação de Mestrado
## Agentic AI for decision support in logistic from heterogeneous data
### Estrutura inicial
agentic_logistics/

├── agent/
│   ├── planner.py
│   ├── executor.py
│   └── reflection.py
│
├── llm/
│   ├── base_llm.py
│   ├── gemini_provider.py
│   ├── openai_provider.py
│   └── azure_provider.py
│
├── tools/
│   ├── base_tool.py
│   ├── sql_tool.py
│   ├── api_tool.py
│   ├── spreadsheet_tool.py
│   └── document_tool.py
│
├── catalog/
│   ├── data_catalog.py
│   ├── schema_registry.py
│   └── metadata_index.py
│
├── memory/
│   ├── episodic_memory.py
│   ├── semantic_memory.py
│   └── procedural_memory.py
│
├── observability/
│   ├── execution_logger.py
│   ├── reasoning_trace.py
│   └── execution_graph.py
│
├── data/
│   └── synthetic_dataset/
│
├── config/
│   └── connections.yaml
│
├──state/
│     ├── execution_state.py
│     ├── state_manager.py
│     └── state_transitions.py
└── main.py

### Arquitetura Inicial
                Usuário
                    │
                    ▼
            Interface / API
                    │
                    ▼
                Planner
                    │
                    ▼
               Tool Router
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Database     APIs        Planilhas
   Connector    Connector   Connector
        │           │           │
        └──────► Data Catalog ◄─┘
                    │
                    ▼
             Execution Engine
                    │
                    ▼
      Camada Analítica e Explicabilidade
      ├ Execution Logs
      ├ Reasoning Trace
      ├ Execution Graph
      └ Feedback Learning
                    │
                    ▼
                 Memória
       (episódica, semântica, procedural)