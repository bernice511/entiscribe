# PDF Named Entity Extractor

Upload PDFs, define what entity types to look for, extract them with an LLM, and check the
extraction quality — either against a gold-standard CSV, an LLM groundedness judge, or both at
once in a single report.

## Workflow

1. **Upload** one or more PDFs. Text is extracted (`pypdf`), chunked, and embedded into a local
   Chroma vector store. Uploading more files is additive — nothing already stored is removed.
2. **Define entity types** in the sidebar: pick from presets (Person, Organization, Location,
   Date, Amount, Product) and/or add your own name + description.
3. **Extract**: for each stored file, the relevant chunks are retrieved and an LLM fills in a
   schema built from your entity definitions.
4. **Evaluate**: optionally upload a gold-standard CSV (`file_name` column + one column per entity
   type) to get precision/recall/F1 per entity. An LLM groundedness score (is each extracted value
   actually supported by the source text?) is always computed. Both are merged into one report.

## Why no API key

All LLM calls shell out to the `claude` CLI (`claude -p ... --output-format json --json-schema ...`)
using your existing Claude Code login, instead of a keyed `ChatOpenAI`/`ChatAnthropic` client.
Embeddings run locally via chromadb's bundled ONNX `all-MiniLM-L6-v2` model (no torch/transformers).
Nothing in this app needs a secret.

## Architecture

- `src/ingestion/pdf.py` — PDF → text (single code path, page numbers kept)
- `src/ingestion/vector_store.py` — additive Chroma wrapper (`langchain_chroma` + a local ONNX
  embedding function)
- `src/llm/claude_cli.py` — subprocess wrapper around the `claude` CLI
- `src/entities/schema.py` — preset entity types + dynamic pydantic model per extraction run
- `src/pipeline/*_graph.py` — three LangGraph pipelines: ingest (load → chunk → embed/store),
  extract (retrieve → extract), evaluate (score groundedness → compare to gold → merge)
- `src/evaluation/metrics.py` — gold-standard precision/recall/F1 (multilabel, via scikit-learn)
- `src/evaluation/groundedness.py` — LLM-judge groundedness score per extracted value
- `src/ui/*`, `src/session.py`, `src/app.py` — Streamlit app

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run src/app.py
```

## Test

```bash
pytest
```
