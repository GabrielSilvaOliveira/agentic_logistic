# Historico e descricao das aplicacoes em `catalog/`

Data de referencia: 2026-04-29

## 1) Objetivo do diretorio

O diretorio `catalog/` concentra os componentes que:
- extraem metadados de SQL, API e planilhas;
- consolidam esses metadados em um `catalog.json` unico;
- constroem contexto em Markdown para o agente;
- selecionam fontes por regras (`context_hints`) e por similaridade semantica (embeddings).

## 2) Historico evolutivo (resumo)

### Fase inicial (v1)
- Estruturacao de extratores dedicados para cada fonte de dados:
  - SQL (`extractors/sql_extractor.py`)
  - API (`extractors/api_extractor.py`)
  - Planilhas (`extractors/spreadsheet_extractor.py`)
- Criacao de um orquestrador unico (`build_catalog.py`) para gerar o catalogo final.
- Adocao do `context_builder.py` com selecao por `context_hints` e fallback para contexto completo.

### Fase atual (v2 semantica)
- Introducao do `semantic_context_builder.py` para selecao por embeddings.
- Inclusao de cache de embeddings por modelo, com invalidacao por hash do `catalog.json`.
- Inclusao de benchmark comparativo entre modelos (MiniLM, E5-base, E5-large, BGE-M3).

### Melhorias recentes (aplicadas)
- Campo `embedding_hint` adicionado como prioridade na geracao dos documentos de embedding.
- Refatoracao de `_doc_from_source()` para documento automatico mais focado:
  - usa `description + business_role`;
  - remove `when_to_use/when_NOT_to_use` da versao automatica;
  - limita e filtra colunas para reduzir ruido semantico.
- Nova estrategia de corte `score_gap` (alem de `threshold`).
- Threshold padrao por modelo:
  - MiniLM: ~0.30
  - E5-base/E5-large: ~0.79
  - BGE-M3: ~0.50 (default inicial de calibracao)
- `build_catalog.py` passou a injetar placeholder de `embedding_hint` para todas as fontes (sem sobrescrever o que ja foi preenchido).

## 3) Descricao das aplicacoes

## 3.1 `build_catalog.py`
Funcao principal:
- Orquestra a extracao SQL, API e planilhas em sequencia.
- Consolida tudo em um unico `catalog.json`.

Responsabilidades:
- carregar template de schema (`catalog_schema_template.json`);
- executar extratores e fazer merge incremental;
- atualizar metadados (`updated_at`);
- injetar `embedding_hint` placeholder quando ausente.

Entrada tipica:
- `build_config.json`

Saida tipica:
- `output/catalog.json`

## 3.2 `context_builder.py` (v1)
Funcao principal:
- Selecionar fontes com base em `context_hints` e renderizar contexto em Markdown.

Responsabilidades:
- leitura lazy do catalogo;
- roteamento por palavras-chave (`trigger_keywords`);
- fallback para incluir todas as fontes quando nenhuma regra casar;
- renderizacao comprimida por tipo de fonte (SQL/API/Planilhas).

Ponto forte:
- previsibilidade e controle manual via regras.

Limitacao:
- depende da qualidade/cobertura de palavras-chave nas regras.

## 3.3 `semantic_context_builder.py` (v2)
Funcao principal:
- Selecionar fontes por similaridade semantica entre query e documentos de cada fonte.

Responsabilidades:
- construir documentos de embedding por fonte;
- aplicar prioridade `embedding_hint` quando preenchido;
- calcular embeddings de query e fontes;
- selecionar fontes com `score_gap` ou `threshold`;
- manter cache de embeddings por modelo.

Diferenciais atuais:
- thresholds calibrados por modelo;
- estrategia `score_gap` para cenarios com scores muito proximos;
- benchmark integrado para comparar modelos com queries esperadas.

## 3.4 `extractors/sql_extractor.py`
Funcao principal:
- Extracao automatica de metadados de tabelas MySQL.

Extrai:
- colunas, tipos, PK/FK;
- contagem de linhas;
- taxa de nulos;
- cardinalidade;
- amostras de dados.

Tambem:
- preserva campos manuais no merge quando ja preenchidos.

## 3.5 `extractors/api_extractor.py`
Funcao principal:
- Extracao semi-automatica de schema de endpoint REST.

Extrai:
- estrutura da resposta;
- campos do objeto de resultado;
- inferencia de tipos e exemplos.

Tambem:
- fallback offline quando endpoint nao responde;
- gera placeholders manuais para semantica de negocio.

## 3.6 `extractors/spreadsheet_extractor.py`
Funcao principal:
- Extracao automatica de metadados de Excel/CSV.

Extrai:
- abas, colunas e tipos semanticos;
- taxa de nulos e cardinalidade;
- estatisticas numericas;
- intervalos de data;
- amostras de linhas.

Tambem:
- gera placeholders manuais para contexto de negocio.

## 4) Arquivos de suporte

- `build_config.json`: configuracao do processo de build do catalogo.
- `catalog_schema_template.json`: estrutura base do catalogo.
- `output/catalog.json`: catalogo final consolidado.
- `output/embeddings_cache/*.pkl`: cache de embeddings por modelo.

## 5) Fluxo operacional recomendado

1. Executar `build_catalog.py` com `build_config.json`.
2. Revisar e preencher campos manuais no `output/catalog.json`.
3. Preencher `embedding_hint` em cada fonte (texto curto e focado).
4. Ajustar `context_hints` (regras de negocio) para complementar a busca semantica.
5. Rodar benchmark do `semantic_context_builder.py` para calibrar modelo/estrategia.
6. Integrar o builder selecionado (v1 ou v2) no agente.

## 6) Observacoes de manutencao

- O contexto para LLM (resposta) e o texto para embedding (selecao) sao artefatos diferentes.
- Evitar textos muito longos no `embedding_hint`; prefira termos discriminativos.
- Recalibrar thresholds quando aumentar o numero de fontes no catalogo.
- Sempre invalidar/regerar cache quando houver alteracao estrutural importante no catalogo.
