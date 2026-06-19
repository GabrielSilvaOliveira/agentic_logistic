# SCB Framework Agêntico Logístico — Protótipo Mínimo

Pipeline end-to-end: SCB → Planner → Executor (SQL) → Synthesizer

## Estrutura

```
scb_agent/
├── agent.py              # Orquestrador — ponto de entrada único
├── requirements.txt
├── core/
│   ├── models.py         # Tipos de dados do pipeline (dataclasses)
│   ├── catalog.py        # Catálogo semântico de fontes
│   ├── llm_client.py     # Cliente unificado Ollama / Azure OpenAI
│   └── components.py     # SCB, Planner, Executor, Synthesizer
├── prompts/
│   └── templates.py      # Todos os prompts — separados do código
├── tools/
│   └── sql_tool.py       # Executor SQL determinístico (sem LLM)
└── test_pipeline.py      # Testes de validação por componente
```

`core/catalog.py` não tem dados próprios: ele carrega
`catalog/output/catalog_v3.json` (mesmo catálogo de 16 fontes usado em
`catalog/run_llm_benchmark.py` para o artigo do SCB), garantindo que o
pipeline e o benchmark trabalhem sobre a mesma fonte de verdade.

## Setup

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente
```bash
# Banco de dados (MySQL do ambiente logístico)
export DB_URL="mysql+pymysql://usuario:senha@host:3306/nome_banco"

# Azure OpenAI (para regime oracle)
export AZURE_OPENAI_ENDPOINT="https://seu-recurso.openai.azure.com"
export AZURE_OPENAI_KEY="sua-chave-aqui"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
```

### 3. Modelos Ollama necessários
```bash
ollama pull llama3.2:3b    # SCB (configuração validada no artigo)
ollama pull qwen2.5:7b     # Planner + Executor + Synthesizer (local-light)
```

## Uso

### Linha de comando
```bash
cd scb_agent

# Regime local (sem custo de API)
python -c "
from agent import LogisticsAgent, AgentRegime
agent = LogisticsAgent(regime=AgentRegime.LOCAL_LIGHT)
r = agent.query('Quantos notebooks estão em falta na 3ª Região Militar?')
print(r.to_display())
"

# Regime oracle (GPT-4o)
python -c "
from agent import LogisticsAgent, AgentRegime
agent = LogisticsAgent(regime=AgentRegime.ORACLE)
r = agent.query('Quantos notebooks estão em falta na 3ª Região Militar?')
print(r.to_display())
"
```

### Testes por componente
```bash
# Todos os testes
python test_pipeline.py

# Componente específico
python test_pipeline.py catalog
python test_pipeline.py sql
python test_pipeline.py scb
python test_pipeline.py planner
python test_pipeline.py pipeline
```

## Catálogo

`core/catalog.py` lê `catalog/output/catalog_v3.json` (16 fontes: 3 tabelas
SQL, 7 endpoints de API, 3 planilhas) e expõe `CATALOG`, `get_source()` e
`get_schema_for_prompt()`. Para fontes SQL, `fields` vem das colunas do
catálogo e `sample_values` vem dos `sample_rows` (dados mascarados) já
presentes no JSON — não é necessário editar nada manualmente.

## Próximos passos (após validação do protótipo)

1. Adicionar `api_tool` em `tools/api_tool.py` — mesma interface que sql_tool
2. Adicionar `excel_tool` em `tools/excel_tool.py`
3. Integrar ao Django via view ou management command
4. Implementar o benchmark end-to-end (80 consultas + gabarito) sobre o pipeline completo

## Regimes de comparação (dissertação)

| Regime       | SCB          | Planner/Synth | Hardware mínimo  |
|--------------|--------------|---------------|------------------|
| local-light  | Llama 3.2 3B | Qwen2.5 7B    | 8 GB RAM / CPU   |
| local-robust | Llama 3.2 3B | Qwen2.5 14B   | 12 GB VRAM (GPU) |
| oracle       | Llama 3.2 3B | GPT-4o (Azure)| API key          |
