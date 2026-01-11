"""
server_core.py

This module attempts to register tools with MCP FastMCP if the installed SDK supports it.
If MCP FastMCP / tool decorator are not available, we provide a fallback Dispatcher
object with a `run_tool(name, params)` coroutine that just calls the corresponding function.

This lets the rest of the code call server_core.run_tool(...) uniformly.
"""

import asyncio
import logging
import importlib
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Tool registry for fallback
_TOOL_REGISTRY = {}

def register_tool(name: str, func):
    """Registers a function in the fallback registry (and returns func)."""
    _TOOL_REGISTRY[name] = func
    return func

# Try to import MCP FastMCP and its decorator
try:
    # Prefer the modern fastmcp API
    from mcp.server.fastmcp import FastMCP, tool as mcp_tool
    MCP_AVAILABLE = True
    logger.info("MCP FastMCP imported successfully — MCP integration enabled.")
except Exception as e:
    FastMCP = None
    mcp_tool = None
    MCP_AVAILABLE = False
    logger.warning("MCP FastMCP unavailable; running in fallback dispatcher mode. Error: %s", e)

# If MCP available, create fastmcp instance and expose helper decorator/wrapper
if MCP_AVAILABLE:
    mcp = FastMCP("primehire-mcp")
    def register_mcp_tool(name=None, description=None):
        """
        Decorator wrapper to register both with MCP (if available) and fallback registry
        Usage:
            @register_mcp_tool("zoho.fetch_candidates")
            async def fetch_candidates(...): ...
        """
        def _dec(fn):
            tool_name = name or fn.__name__
            # register with fastmcp
            try:
                # If the SDK provides @mcp.tool usage, use it by decorating a wrapper.
                # Use mcp_tool if available (some SDKs provide it)
                if mcp_tool:
                    decorated = mcp_tool(name=tool_name, description=description)(fn)
                else:
                    # Some versions expect mcp.add_tool
                    mcp.add_tool(fn)
                    decorated = fn
                logger.info("Registered MCP tool: %s", tool_name)
            except Exception as e:
                logger.warning("Could not register tool with MCP: %s — falling back. Err: %s", tool_name, e)
                decorated = fn
            # always register in fallback registry so we can run even if MCP is absent
            register_tool(tool_name, fn)
            return decorated
        return _dec
    async def run_tool(name: str, params: Dict[str, Any]):
        """Run a tool via MCP if possible, else call local function."""
        # If mcp has run_tool API, use it
        try:
            # some fastmcp versions expose run_tool or run_tool_async; try both
            if hasattr(mcp, "run_tool"):
                return await mcp.run_tool(name, params)
            elif hasattr(mcp, "run"):
                # older variants might require method call object; fallback to registry
                pass
        except Exception as e:
            logger.debug("mcp.run_tool failed, falling back to local registry: %s", e)

        # Fallback to direct call
        fn = _TOOL_REGISTRY.get(name)
        if not fn:
            raise RuntimeError(f"Tool not found: {name}")
        # If fn is async
        if asyncio.iscoroutinefunction(fn):
            return await fn(**(params or {}))
        else:
            # run in threadpool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: fn(**(params or {})))

else:
    # MCP not available: fallback dispatcher only
    def register_mcp_tool(name=None, description=None):
        def _dec(fn):
            tool_name = name or fn.__name__
            register_tool(tool_name, fn)
            logger.info("Registered fallback tool: %s", tool_name)
            return fn
        return _dec

    async def run_tool(name: str, params: Dict[str, Any]):
        fn = _TOOL_REGISTRY.get(name)
        if not fn:
            raise RuntimeError(f"Tool not found: {name}")
        import asyncio, inspect
        if inspect.iscoroutinefunction(fn):
            return await fn(**(params or {}))
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: fn(**(params or {})))
