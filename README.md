# LangGraph MCP Agent (Studio-ready)

Lean agent wired to LangGraph’s Postgres checkpointer/memory and MCP tools, with a simple console runner and LangGraph Studio integration.

## Features

- Compiled graph exposed for Studio (`app/graph.py:graph` via `langgraph.json`).
- Postgres-backed checkpoints and memory using `AsyncPostgresSaver` and `AsyncPostgresStore`.
- MCP tool loading from `config.yaml` (Multi-Server MCP client).
- Minimal console runtime for local testing.

## Project Layout

- `agents/graph.py` — graph builder (`build_graph`) with MCP tooling and summarization.
- `app/graph.py` — compiled graph for LangGraph CLI/Studio.
- `runtime/console.py` — interactive console loop using the compiled graph.
- `tools/mcp/client.py` — MCP client and tool loader; uses `config.yaml`.
- `config/loader.py`, `config.yaml` — configuration loading for MCP clients.
- `.env.example` — environment variable template.
- `compose.yaml` — optional local Postgres stack.

## Prerequisites

- Python 3.13+
- OpenAI API key
- Optional: LangSmith API key for Studio traces
- Optional: Docker (if you want to run local Postgres via compose)

## Setup

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

2. Install the package

```bash
pip install -e .
```

3. Configure environment

```bash
cp .env.example .env
```

Fill in at least:

- `OPENAI_API_KEY`
- `PGVECTOR_CONN` (e.g. `postgresql+psycopg://agentic_app:change_this_password@localhost:5442/VectorDB`)
- Optional: `LANGSMITH_API_KEY` for Studio observability

4. Start Postgres locally for checkpointing and memory

```bash
docker compose up -d
```

Compose reads environment variables from `.env` for container setup. You can override:

- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` (default `VectorDB`), `PG_PORT` (default `5442`).

Then set `PGVECTOR_CONN` to match, for example:
`postgresql+psycopg://agentic_app:change_this_password_now@localhost:5442/VectorDB`.

## Run in Studio

Install LangGraph CLI if needed:

```bash
pip install --upgrade "langgraph-cli[inmem]"
```

Start the local server (reads `langgraph.json`):

```bash
langgraph dev agent
```

[Open Studio](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)

## Run in Console

```bash
python runtime/console.py
```

Type your message, or `q`/`quit`/`exit` to quit.

## Notes

- The graph binds MCP tools from the endpoints listed in `config.yaml`.
- The app writes checkpoints and memory to the Postgres database from `PGVECTOR_CONN`.
