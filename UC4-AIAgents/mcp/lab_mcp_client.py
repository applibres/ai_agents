"""
lab_mcp_client.py — Lab: Using MultiServerMCPClient with LangChain
==================================================================

GOAL
----
Connect ONE LangChain agent to TWO MCP servers (math + weather) using
`MultiServerMCPClient`, and let the LLM decide which tool to call.

ARCHITECTURE
------------
                         ┌──────────────────┐
                    ┌───►│  math_server.py  │  tools: add, multiply
   ┌────────────┐   │    └──────────────────┘
   │  LangChain │   │
   │   Agent    ├───┤  MultiServerMCPClient
   │ (Ollama)   │   │
   └────────────┘   │    ┌──────────────────┐
                    └───►│ weather_server.py│  tool: get_weather
                         └──────────────────┘

SETUP
-----
    pip install langchain langchain-mcp-adapters langchain-ollama mcp
    ollama pull qwen2.5:7b      # any Ollama model that supports tool calling

Put math_server.py, weather_server.py and this file in the SAME folder, then:
    python lab_mcp_client.py
"""

import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

# Folder where this script lives (used to find the server files).
HERE = Path(__file__).parent


async def main():
    # ------------------------------------------------------------------
    # STEP 1 — Describe the servers in a dictionary.
    # ------------------------------------------------------------------
    # Each key is a name YOU choose for the server.
    # For "stdio" servers we say HOW to launch them:
    #   command   -> the program to run (sys.executable = this same Python)
    #   args      -> the script to run (absolute path is safest)
    #   transport -> "stdio" = the client starts the server as a subprocess
    servers = {
        "math": {
            "command": sys.executable,
            "args": [str(HERE / "math_server.py")],
            "transport": "stdio",
        },
        "weather": {
            "command": sys.executable,
            "args": [str(HERE / "weather_server.py")],
            "transport": "stdio",
        },
    }

    # ------------------------------------------------------------------
    # STEP 2 — Create the client.
    # ------------------------------------------------------------------
    # Creating it does not connect yet; it only stores the configuration.
    client = MultiServerMCPClient(servers)

    # ------------------------------------------------------------------
    # STEP 3 — Discover the tools from ALL servers.
    # ------------------------------------------------------------------
    # get_tools() connects to each server, asks "which tools do you have?"
    # and converts every MCP tool into a regular LangChain tool.
    tools = await client.get_tools()

    print("=== Tools discovered ===")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    # ------------------------------------------------------------------
    # STEP 4 — (Optional) Call a tool directly, WITHOUT an LLM.
    # ------------------------------------------------------------------
    # Useful to check that the servers work before adding the agent.
    add_tool = next(t for t in tools if t.name == "add")
    result = await add_tool.ainvoke({"a": 3, "b": 3})
    print("\n=== Direct tool call ===")
    print("add(3, 3) =", result)

    # ------------------------------------------------------------------
    # STEP 5 — Create the LLM and the agent.
    # ------------------------------------------------------------------
    # The agent receives the MCP tools exactly like any other LangChain tool.
    llm = ChatOllama(model="deepseek-r1:8b", temperature=0)
    agent = create_agent(llm, tools)

    # ------------------------------------------------------------------
    # STEP 6 — Ask questions. The LLM decides which server/tool to use.
    # ------------------------------------------------------------------
    questions = [
        "What is (3 + 5) multiplied by 12?",   # -> math server
        "What's the weather in urcuqui?",      # -> weather server
    ]

    for question in questions:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )

        print(f"\n=== Question: {question} ===")
        # Show which tools the agent called (to SEE the MCP calls happen)
        for msg in response["messages"]:
            for call in getattr(msg, "tool_calls", []) or []:
                print(f"  [tool call] {call['name']}({call['args']})")
        # The last message is the final answer
        print("Answer:", response["messages"][-1].content)


# get_tools() and ainvoke() are async, so we run main() with asyncio.
if __name__ == "__main__":
    asyncio.run(main())
