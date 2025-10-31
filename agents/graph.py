import uuid
import datetime
from dataclasses import dataclass
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, ToolMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model

from tools.mcp.client import get_mcp_tools
from constants import NAMESPACE_MEMORIES

# --- Init model provider ---
llm = init_chat_model("openai:gpt-4.1", temperature=0)


# --- Schema Definitions ---
class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str


# Define the context schema
@dataclass
class ContextSchema:
    thread_id: str
    user_id: str


# --- Graph Routing Function ---


def route_to_summary_or_chatbot(state: State) -> Literal["summary_node", "chatbot"]:
    """
    Checks the number of messages in the state. If the conversation is getting long,
    it routes to the summarization node. Otherwise, it routes to the main chatbot node.
    """
    if len(state["messages"]) > 10:
        print("--- Conversation is long, routing to summary node ---")
        return "summary_node"
    else:
        print("--- Conversation is short, routing to chatbot node ---")
        return "chatbot"


# --- Agent Builder Function ---


async def build_graph():
    """Prepares the uncompiled graph for the agent."""

    all_tools = await get_mcp_tools()
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(all_tools)

    # --- Node Definitions ---

    async def chatbot_node(
        state: State, *, store: BaseStore, runtime: Runtime[ContextSchema]
    ):
        """Main chatbot node with RAG from Postgres store and conversation summary."""
        print("--- Calling Chatbot Node ---")
        user_id = runtime.context.user_id
        namespace = ("memories", user_id)
        last_message = state["messages"][-1]

        # RAG Search for conversation history
        memories = await store.asearch(
            namespace, query=str(last_message.content), limit=3
        )
        info_parts = []
        for d in memories:
            if isinstance(d.value, dict) and "data" in d.value and "date" in d.value:
                info_parts.append(f"Memory from {d.value['date']}: {d.value['data']}")
            else:
                info_parts.append(str(d.value))
        info = "\n".join(info_parts)

        # Get conversation summary
        summary = state.get("summary", "")
        summary_prompt = (
            f"Here is a summary of the conversation so far:\n{summary}"
            if summary
            else ""
        )

        system_prompt = (
            f"You are a helpful research assistant. {summary_prompt}\n"
            f"Here is some information from your memory that might be relevant:\n{info}\n\n"
            "Use the available MCP tools to find or manipulate information as needed."
        )

        response = await llm_with_tools.ainvoke(
            [("system", system_prompt)] + state["messages"]
        )
        return {"messages": [response]}

    tool_node = ToolNode(tools=all_tools)

    async def save_tool_result_node(
        state: State, *, store: BaseStore, runtime: Runtime[ContextSchema]
    ):
        """Saves the output of a tool call to the vector store."""
        print("--- Calling Save Tool Result Node ---")
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            tool_output = last_message.content
            tool_name = last_message.name
            current_time = datetime.datetime.now().isoformat()
            descriptive_text = (
                f"The agent used the tool '{tool_name}'.\nThe result was: {tool_output}"
            )
            data_to_save = {"date": current_time, "data": descriptive_text}
            user_id = runtime.context.user_id
            namespace = (NAMESPACE_MEMORIES, user_id)
            await store.aput(namespace, str(uuid.uuid4()), data_to_save)
            print(f"--- Saved structured tool result to memory for user {user_id} ---")
        return {}

    async def summarize_conversation_node(state: State):
        """Summarizes the conversation history."""
        print("--- Calling Summarization Node ---")
        summary = state.get("summary", "")

        summary_prompt = (
            f"This is a summary of the conversation to date:\n{summary}\n\nExtend the summary by taking into account the new messages above:"
            if summary
            else "Create a summary of the conversation above:"
        )

        messages_with_prompt = state["messages"] + [
            HumanMessage(content=summary_prompt)
        ]
        response = await llm.ainvoke(messages_with_prompt)

        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    # --- Graph Wiring ---
    graph_builder = StateGraph(State, context_schema=ContextSchema)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("save_tool_result", save_tool_result_node)
    graph_builder.add_node("summary_node", summarize_conversation_node)

    graph_builder.set_conditional_entry_point(
        route_to_summary_or_chatbot,
        {"summary_node": "summary_node", "chatbot": "chatbot"},
    )
    graph_builder.add_edge("summary_node", "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot", tools_condition, {"tools": "tools", END: END}
    )
    graph_builder.add_edge("tools", "save_tool_result")
    graph_builder.add_edge("save_tool_result", "chatbot")

    return graph_builder
