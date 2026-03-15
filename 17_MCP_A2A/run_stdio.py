"""Run Cat Shop MCP server in stdio mode (for Claude local MCP)."""
from app import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
