#!/usr/bin/env python3
"""
Standalone Airbnb MCP search script.
Bypasses the agent to avoid token limits - calls MCP tool directly and summarizes.

Usage: python3 airbnb_search.py "<user_query>"
"""

import sys
import json
import os
import asyncio
import re
from dotenv import load_dotenv

_project_root = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(os.path.join(_project_root, ".env"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


def get_groq_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            import toml
            secrets_path = os.path.join(_project_root, ".streamlit", "secrets.toml")
            if os.path.exists(secrets_path):
                secrets = toml.load(secrets_path)
                key = secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    return key


def parse_query(query: str) -> dict:
    """Extract search params from query."""
    params = {"location": "Wayanad, Kerala", "adults": 2}
    # Try to extract dates like "next 3 days"
    from datetime import datetime, timedelta
    today = datetime.now()
    match = re.search(r"next\s+(\d+)\s+days?", query, re.IGNORECASE)
    if match:
        days = int(match.group(1))
        checkin = today.strftime("%Y-%m-%d")
        checkout = (today + timedelta(days=days)).strftime("%Y-%m-%d")
        params["checkin"] = checkin
        params["checkout"] = checkout
    # Try to extract location
    loc_match = re.search(r"(?:for|to|in)\s+([A-Za-z\s,]+?)(?:\s*$|\s+with|\s+for)", query)
    if loc_match:
        params["location"] = loc_match.group(1).strip()
    return params


async def run_airbnb_search(query: str) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        return "❌ Error: GROQ_API_KEY not found."

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=1.0,
        max_tokens=4096,
        api_key=api_key,
    )

    search_params = parse_query(query)
    location = search_params.get("location", "Wayanad, Kerala")

    print(f"🔍 Searching: {location}, params={search_params}", file=sys.stderr)

    import subprocess as sp
    tmp_script = os.path.join(os.path.dirname(__file__), ".tmp_mcp_runner.py")

    # Write the MCP runner script with proper JSON encoding
    location_json = json.dumps(location)
    checkin = search_params.get("checkin", "")
    checkout = search_params.get("checkout", "")
    adults = search_params.get("adults", 2)

    tmp_script_content = f'''import asyncio
import sys
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    location = {location_json}
    checkin = {json.dumps(checkin)}
    checkout = {json.dumps(checkout)}
    adults = {adults}

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool_args = {{"location": location, "adults": adults}}
            if checkin and checkout:
                tool_args["checkin"] = checkin
                tool_args["checkout"] = checkout

            result = await session.call_tool("airbnb_search", tool_args)
            for content in result.content:
                if hasattr(content, "text"):
                    print(content.text)
                else:
                    print(str(content))

if __name__ == "__main__":
    asyncio.run(main())
'''

    with open(tmp_script, 'w') as f:
        f.write(tmp_script_content)

    try:
        # Step 1: Get raw search results
        result = await asyncio.to_thread(
            sp.run,
            [sys.executable, tmp_script],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"⚠️ MCP tool failed:\n{result.stderr}", file=sys.stderr)
            return f"❌ Airbnb search failed: {result.stderr[:300]}"

        raw_results = result.stdout

        # Step 2: Summarize with LLM
        response = await llm.ainvoke([
            SystemMessage(content=(
                "Summarize these Airbnb search results. "
                "Show top 5 properties with: name, rating, price/night, type, key amenities, link. "
                "Be concise. Use markdown format."
            )),
            HumanMessage(content=f"Query: {query}\n\nResults:\n{raw_results[:8000]}")
        ])

        return response.content

    finally:
        try:
            os.remove(tmp_script)
        except:
            pass


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No query provided"}))
        sys.exit(1)

    query = sys.argv[1]

    try:
        result = asyncio.run(run_airbnb_search(query))
        print(json.dumps({"success": True, "result": result}))
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}"
        print(json.dumps({"success": False, "error": error_msg, "traceback": traceback.format_exc()}), file=sys.stderr)
        print(json.dumps({"success": False, "error": error_msg}))
        sys.exit(1)


if __name__ == "__main__":
    main()
