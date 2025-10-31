import os
import asyncio
from dotenv import load_dotenv

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

# Import the legacy graph builder
from agents.graph import build_graph


load_dotenv()


async def _compile_graph():
    """Build and compile the legacy graph with Postgres checkpointer + store.

    Exposes a compiled graph object for LangGraph CLI / Studio.
    """
    pg_conn = os.environ["PGVECTOR_CONN"]
    if pg_conn.startswith("postgresql+psycopg://"):
        pg_conn = pg_conn.replace("postgresql+psycopg://", "postgresql://", 1)

    graph_builder = await build_graph()

    # Initialize Postgres-backed checkpointer and store (call setup() explicitly)
    checkpointer = AsyncPostgresSaver.from_conn_string(pg_conn)
    await checkpointer.setup()

    store = AsyncPostgresStore.from_conn_string(pg_conn)
    await store.setup()

    return graph_builder.compile(checkpointer=checkpointer, store=store)


# Compiled graph object used by LangGraph CLI (langgraph dev)
graph = asyncio.run(_compile_graph())
