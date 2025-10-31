import asyncio
import os
import sys
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.graph import build_graph


load_dotenv()


async def main():
    pg_conn = os.environ["PGVECTOR_CONN"]
    if pg_conn.startswith("postgresql+psycopg://"):
        pg_conn = pg_conn.replace("postgresql+psycopg://", "postgresql://", 1)

    graph_builder = await build_graph()

    async with (
        AsyncPostgresSaver.from_conn_string(pg_conn) as checkpointer,
        AsyncPostgresStore.from_conn_string(pg_conn) as store,
    ):
        graph = graph_builder.compile(checkpointer=checkpointer, store=store)
        print(graph.get_graph(xray=True).draw_mermaid())

        user_id = os.getenv("DEMO_USER_ID", "123456")
        thread_id = f"thread_{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}
        context = {"thread_id": thread_id, "user_id": user_id}

        print("Console Agent ready. Type 'quit' to exit.")
        print(f"Running with user_id: {user_id} and thread_id: {thread_id}")

        while True:
            try:
                text = input("User: ")
                if text.lower() in {"q", "quit", "exit"}:
                    print("Goodbye!")
                    break
                final_state = await graph.ainvoke(
                    {"messages": [HumanMessage(content=text)]},
                    config=config,
                    context=context,
                )
                last_msg = final_state["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    print(f"AI: {last_msg.content}")
                else:
                    content = getattr(last_msg, "content", None)
                    if content:
                        print(f"AI: {content}")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    asyncio.run(main())
