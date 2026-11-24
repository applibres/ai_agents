"""
math_server.py — MCP Server #1 (Math tools)
===========================================
A tiny MCP server that exposes two tools: add and multiply.
You do NOT run this file yourself: the client (lab_mcp_client.py)
starts it automatically as a subprocess using the "stdio" transport.
"""

from mcp.server.fastmcp import FastMCP

# 1) Create the server and give it a name.
mcp = FastMCP("Math")


# 2) Declare tools with the @mcp.tool() decorator.
#    - The function NAME becomes the tool name.
#    - The DOCSTRING becomes the tool description the LLM reads.
#    - The TYPE HINTS become the tool's input schema.
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


# 3) Start the server. "stdio" = it talks to the client through
#    standard input/output (no network, no ports).
if __name__ == "__main__":
    mcp.run(transport="stdio")
