"""
weather_server.py — MCP Server #2 (Weather tool)
================================================
A second, independent MCP server. It returns FAKE weather data so the
lab works offline. The point is to show that ONE client can talk to
SEVERAL servers at the same time.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

# Fake data (no internet needed)
FAKE_WEATHER = {
    "quito": "15 °C, cloudy",
    "guayaquil": "30 °C, sunny",
    "urcuqui": "18 °C, light rain",
}


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city in Ecuador."""
    return FAKE_WEATHER.get(city.lower(), f"No data for {city}")


if __name__ == "__main__":
    mcp.run(transport="stdio")
