from typing import Dict, Optional, List

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from config.loader import load_config


load_dotenv()

# Module-level caches to avoid re-initializing MCP on every call
_mcp_client: Optional[MultiServerMCPClient] = None
_mcp_tools_cache: Optional[List] = None
_per_server_cache: Dict[str, List] = {}


async def get_mcp_tools():
    """Initializes the MultiServerMCPClient and returns the available tools."""
    global _mcp_client, _mcp_tools_cache
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache
    config = load_config()
    print("--- Initializing MCP Tools ---")
    _mcp_client = MultiServerMCPClient(config["mcp_clients"])
    mcp_tools = await _mcp_client.get_tools()
    print(f"--- Found {len(mcp_tools)} MCP Tools ---")
    _mcp_tools_cache = mcp_tools
    return mcp_tools


async def get_mcp_tools_for(server_name: str):
    """Return only the tools for a single MCP server defined in config.yaml."""
    global _per_server_cache
    if server_name in _per_server_cache:
        return _per_server_cache[server_name]
    config = load_config()
    all_clients: Dict = config.get("mcp_clients", {})
    if server_name not in all_clients:
        raise ValueError(
            f"MCP server '{server_name}' not found in config.yaml. Available: {list(all_clients.keys())}"
        )
    single_cfg = {server_name: all_clients[server_name]}
    print(f"--- Initializing MCP Tools for server: {server_name} ---")
    mcp_client = MultiServerMCPClient(single_cfg)
    tools = await mcp_client.get_tools()
    print(f"--- Found {len(tools)} tools for {server_name} ---")
    _per_server_cache[server_name] = tools
    return tools
