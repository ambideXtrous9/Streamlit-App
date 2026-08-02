from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
import os
import re
import datetime
import requests
import json
import uuid
import time
import asyncio
import sys
import subprocess
import concurrent.futures
from typing import Any, Annotated, List, Dict
from typing_extensions import TypedDict
from langchain_core.tools import tool, Tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, Field
from langfuse.langchain import CallbackHandler

# ── Node.js availability check ─────────────────────────────────────
def _check_node() -> bool:
    for cmd in (["node", "-v"], ["npx", "--version"]):
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    return True

_NPX_AVAILABLE = _check_node()

if not _NPX_AVAILABLE:
    if sys.platform.startswith("linux"):
        print("⚠️ Node.js not found, installing via apt-get...")
        try:
            subprocess.run(["apt-get", "update", "-qq"], check=False, capture_output=True, timeout=120)
            subprocess.run(["apt-get", "install", "-y", "-qq", "nodejs", "npm"], check=False, capture_output=True, timeout=180)
            _NPX_AVAILABLE = _check_node()
        except Exception as e:
            print(f"apt-get fallback error: {e}")

if not _NPX_AVAILABLE:
    print("⚠️ Airbnb MCP requires Node.js/npx. Install it to enable this feature.")
else:
    try:
        node_version = subprocess.run(["node", "-v"], capture_output=True, text=True, timeout=5)
        print(f"✅ Node.js {node_version.stdout.strip()} available — Airbnb MCP enabled")
    except Exception:
        print("✅ Node.js available — Airbnb MCP enabled")

# Langfuse handler
try:
    langfuse_handler = CallbackHandler()
except Exception:
    langfuse_handler = None


class ArticleResponse(TypedDict):
    topic: str
    summary: str
    knowledge: Annotated[list[AnyMessage], add_messages]


if hasattr(st, "secrets"):
    for sec_key in ["GROQ_API_KEY", "WEATHER_API_KEY", "OPENROUTER_API_KEY", "QDRANT_API_KEY", "COHERE_API_KEY"]:
        if sec_key in st.secrets:
            os.environ[sec_key] = st.secrets[sec_key]

from llm_utils import build_llm
llm = build_llm(temperature=0.0)


# ── Smart Query Parsing Helper ─────────────────────────────────────
def parse_trip_query(query: str):
    """
    Parses user query to extract location name, checkin date, checkout date, and stay duration.
    Defaults to tomorrow for checkin if not explicitly specified.
    """
    today = datetime.date.today()
    checkin = today + datetime.timedelta(days=1)
    
    # Extract duration in days if mentioned (e.g. '3 days', '5 day')
    days_match = re.search(r'(\d+)\s*days?', query, re.IGNORECASE)
    duration = int(days_match.group(1)) if days_match else 3
    checkout = checkin + datetime.timedelta(days=duration)

    # Heuristic for location extraction
    stop_words = r'(?i)\b(\d+\s*days?|trip|itinerary|to|from|tomorrow|next|week|for|in|stay|hotel|airbnb|booking|cheap|best)\b'
    cleaned = re.sub(stop_words, ' ', query)
    words = [w.strip().capitalize() for w in cleaned.split() if len(w.strip()) > 2]
    location = ' '.join(words) if words else 'Munnar'

    return location, checkin.strftime('%Y-%m-%d'), checkout.strftime('%Y-%m-%d'), duration


# ── Airbnb Agent ──────────────────────────────────────────────────
async def airbnbAgent(state: Dict[str, Any]):
    topic = state.get("topic", "")
    location, checkin_str, checkout_str, duration = parse_trip_query(topic)
    print(f"🏠 Airbnb Agent searching: location='{location}', checkin='{checkin_str}', checkout='{checkout_str}'")
    start = time.time()

    if not _NPX_AVAILABLE:
        ai_content = "⚠️ Airbnb search unavailable — Node.js/npx runtime not available."
    else:
        ai_content = ""
        try:
            mcp_env = dict(os.environ)
            mcp_env["AIRBNB_BASE_URL"] = "https://www.airbnb.co.in"

            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
                env=mcp_env,
            )

            with open(os.devnull, "w") as err_log:
                async with stdio_client(server_params, errlog=err_log) as (read, write):
                    async with ClientSession(read, write) as session:
                        print("Initializing MCP connection...")
                        await asyncio.wait_for(session.initialize(), timeout=25)

                        orig_call_tool = session.call_tool

                        async def custom_call_tool(name, arguments=None, **kwargs):
                            try:
                                result = await asyncio.wait_for(
                                    orig_call_tool(name, arguments=arguments, **kwargs),
                                    timeout=30
                                )
                            except asyncio.TimeoutError:
                                print(f"⚠️ MCP tool '{name}' timed out")
                                return None
                            if name == "airbnb_search" and hasattr(result, "content") and result.content:
                                for c in result.content:
                                    if hasattr(c, "text") and c.text:
                                        try:
                                            data = json.loads(c.text)
                                            if isinstance(data, dict) and "searchResults" in data and isinstance(data["searchResults"], list):
                                                data["searchResults"] = data["searchResults"][:5]
                                                data.pop("paginationInfo", None)
                                                c.text = json.dumps(data)
                                        except Exception as ex:
                                            print(f"JSON truncation note: {ex}")
                            return result

                        session.call_tool = custom_call_tool

                        print("Loading MCP tools...")
                        tools = await asyncio.wait_for(load_mcp_tools(session), timeout=15)
                        # Filter to only search tool to prevent extra MCP calls that crash TaskGroup
                        tools = [t for t in tools if t.name == "airbnb_search"]
                        print(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")

                        class CleanAirbnbSearch(BaseModel):
                            location: str = Field(description="Location to search for (city, state, etc.)")
                            checkin: str = Field(default=checkin_str, description="Check-in date (YYYY-MM-DD)")
                            checkout: str = Field(default=checkout_str, description="Check-out date (YYYY-MM-DD)")
                            adults: float = Field(default=2, description="Number of adults")
                            children: float = Field(default=0, description="Number of children")

                        for t in tools:
                            if t.name == "airbnb_search":
                                t.args_schema = CleanAirbnbSearch

                        prompt_text = (
                            f"Extracted Destination: {location}\n"
                            f"Check-in Date: {checkin_str}\n"
                            f"Check-out Date: {checkout_str}\n"
                            f"Duration: {duration} nights\n\n"
                            f"Instructions:\n"
                            f"1. Search for available stays in '{location}' from {checkin_str} to {checkout_str} using airbnb_search.\n"
                            f"2. For EACH listing found, extract ALL available details:\n"
                            f"   - Full property name and type (hotel, villa, cottage, apartment)\n"
                            f"   - Star rating and review count\n"
                            f"   - Complete address and neighborhood\n"
                            f"   - Price per night (INR & USD), total price, taxes & fees breakdown\n"
                            f"   - Room categories (beds, bedrooms, bathrooms)\n"
                            f"   - ALL amenities (WiFi, pool, gym, spa, kitchen, parking, AC, heating, washer, balcony, mountain/sea view etc.)\n"
                            f"   - Direct booking URL\n"
                            f"   - Host details (superhost status, response rate)\n"
                            f"   - Check-in/Check-out times and cancellation policy if available\n"
                            f"   - Distance to city center and key landmarks if mentioned\n"
                            f"   - Guest review highlights and standout features\n"
                            f"3. Present ALL extracted data in a structured format. Do NOT skip any available field.\n"
                            f"4. Use ONLY the airbnb_search tool. Do NOT call any other tool.\n"
                        )

                        agent = create_react_agent(
                            llm,
                            tools,
                            prompt=prompt_text,
                        )

                        print(f"Invoking Airbnb react agent for query: {topic}")
                        response = await asyncio.wait_for(
                            agent.ainvoke({"messages": [{"role": "user", "content": topic}]}),
                            timeout=60
                        )

                        ai_content = response["messages"][-1].content
                        print(f"Final Airbnb agent response: {ai_content[:500]}")
        except (Exception, BaseException) as e:
            error_name = type(e).__name__
            print(f"⚠️ Airbnb Agent error ({error_name}): {e}")
            if not ai_content:
                ai_content = (
                    f"⚠️ Could not load live Airbnb listings for {location} ({checkin_str} to {checkout_str}).\n"
                    f"Note: Standard boutique homestays & mountain cottages are recommended for this destination."
                )

    print(f"✅ Airbnb Agent completed in {time.time() - start:.2f}s")
    return {"knowledge": [f"[Info from AirBnb Search]\n{ai_content}\n\n"]}


# ── Weather helpers ───────────────────────────────────────────────
def extract_weather(data: dict) -> str:
    lines = []
    loc = data.get("location", {})
    location = f"{loc.get('name')}, {loc.get('region')}, {loc.get('country')}"
    lines.append(f"📍 Location: {location}")

    current = data.get("current", {})
    lines.append("\n🌤️ Current Weather:")
    lines.append(f"  Temp: {current.get('temp_c')}°C (Feels like {current.get('feelslike_c')}°C)")
    lines.append(f"  Condition: {current.get('condition', {}).get('text')}")
    lines.append(f"  Humidity: {current.get('humidity')}%")
    lines.append(f"  Wind Gust: {current.get('gust_kph')} kph")
    lines.append(f"  Pressure: {current.get('pressure_mb')} mb")

    forecast = data.get("forecast", {}).get("forecastday", [])
    lines.append("\n📅 Forecast:")
    for day in forecast:
        d = day.get("date")
        det = day.get("day", {})
        lines.append(f"  Date: {d}")
        lines.append(f"    Condition: {det.get('condition', {}).get('text')}")
        lines.append(f"    Max Temp: {det.get('maxtemp_c')}°C")
        lines.append(f"    Min Temp: {det.get('mintemp_c')}°C")
        lines.append(f"    Avg Humidity: {det.get('avghumidity')}%")
        lines.append(f"    Max Wind: {det.get('maxwind_kph')} kph")
        lines.append("-" * 40)

    return "\n".join(lines)


# ── Weather tool ──────────────────────────────────────────────────
class WeatherArgs(BaseModel):
    location: str = Field(description="City name or coordinates")
    days: Any = Field(default=3, description="Number of days to forecast")


@tool("WeatherForecast", args_schema=WeatherArgs)
def get_forecast(location: str, days: int = 3):
    """Fetch weather forecast for a given location using WeatherAPI."""
    days = int(days)
    print(f"🌤️ WeatherForecast tool: {location}, {days} days")
    API_KEY = st.secrets.get("WEATHER_API_KEY", "")
    if not API_KEY:
        return "Weather API key missing."
    url = (
        f"http://api.weatherapi.com/v1/forecast.json"
        f"?key={API_KEY}&q={location}&days={days}&aqi=no&alerts=yes"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return extract_weather(response.json())
    except requests.RequestException as e:
        print(f"Error fetching forecast: {e}")
        return f"Weather forecast unavailable for {location}."


# ── Weather Agent ─────────────────────────────────────────────────
async def weatherAgent(state: Dict[str, Any]):
    topic = state.get("topic", "")
    location, _, _, duration = parse_trip_query(topic)
    print(f"Weather Agent searching: location='{location}', days={duration}")
    
    weather_agent = create_react_agent(
        llm,
        [get_forecast],
        prompt=(
            "You are a Weather Assistant.\n"
            "Use the **WeatherForecast tool** to fetch the forecast.\n"
            "Generate a **Weather Report** with Current Conditions, Forecast Summary, and Tour Recommendation."
        ),
    )

    start = time.time()
    try:
        result = await asyncio.wait_for(
            weather_agent.ainvoke({"messages": [{"role": "user", "content": topic}]}),
            timeout=25
        )
        ai_content = result["messages"][-1].content
    except Exception as e:
        print(f"⚠️ Weather Agent LLM exception ({e}), executing direct tool fallback...")
        try:
            direct_report = get_forecast(location, days=duration)
            ai_content = direct_report if direct_report else f"Weather report unavailable: {e}"
        except Exception as ex:
            ai_content = f"Weather report unavailable: {ex}"
                
    print(f"✅ Weather Agent completed in {time.time() - start:.2f}s")
    return {"knowledge": [f"[Info from Weather Search]\n{ai_content}\n\n"]}


# ── Tour Agent (final summarizer) ─────────────────────────────────
touragentprompt = """
You are an expert Travel & Tour Guide Assistant. Create a comprehensive, stunning Tour Guide Plan based on the user request and gathered intelligence reports.

Format the output strictly using GitHub Flavored Markdown. Use ONLY Markdown — NO raw HTML tags like <br>. Use newlines for line breaks.

## 🎯 Search Summary
- **Location:** [location] | **Dates:** [checkin] → [checkout]
- **Guests:** [adults]A, [children]C, [infants]I, [pets]P
- **Room:** [room type] | **Stars:** [rating] | **Amenities:** [amenities]
- **Results:** [number] hotels/stays found

---

## 🏨 Hotel / Stay Listings

For EACH stay found, create a detailed section:

### 🏠 [Hotel/Stay Name]
| Detail | Info |
|--------|------|
| ⭐ Rating | [rating]/5 ([reviews] reviews) |
| 📍 Address | [full address, neighborhood] |
| 💰 Rate | ₹[price]/night (≈ $[usd]) + ₹[tax] fees |
| 💵 Total | ₹[total] for [nights] nights |
| 🏠 Rooms | [bedrooms, beds, bathrooms] |
| 📏 Distance | [city center distance] • [airport/station distance] |
| 🔗 Booking | [Direct Booking URL] |
| 🏷️ Host | [host name, superhost badge, response rate] |

**Key Amenities:** WiFi, Pool, Gym, Spa, Kitchen, Parking, AC, Heating, Washer, Balcony, Mountain View, etc.

**Booking Details:** Check-in [time], Check-out [time], Cancellation [policy], Payment [methods]

**Guest Highlights:** [Top review quotes or standout features]

---

(Repeat for each stay)

## 📈 Quick Comparison
| Stay | Rating | Price/Night | Key Features | Booking |
|------|--------|-------------|--------------|----------|
| [S1] | [rating]⭐ | ₹[price] | [2 highlights] | [URL] |
| [S2] | [rating]⭐ | ₹[price] | [2 highlights] | [URL] |

---

## 🏆 Final Picks
- **🥇 Best Value:** [stay + reason]
- **💎 Luxury Pick:** [stay + premium features]
- **💰 Budget Pick:** [stay + savings]
- **📍 Best Location:** [stay + location benefit]
- **✨ Best Amenities:** [stay + standout amenities]

---

## 🌤️ Weather Forecast & Climate Guide
- Summary of current conditions, temperature range, wind, and rain likelihood.
- Day-by-day forecast table:

| Day | Date | Condition | High | Low | Humidity | Wind |
|-----|------|-----------|------|-----|----------|------|
| Day 1 | [date] | [condition] | [max]°C | [min]°C | [humidity]% | [wind] kph |

---

## 🧭 Day-by-Day Itinerary
For each day, provide:
- Morning, Afternoon, Evening activities based on weather
- Restaurant / dining suggestions
- Transport tips

---

## 🎒 Packing Essentials
- Clothing recommendations based on weather forecast
- Essential travel items for the destination
- Special items (trekking gear, sunscreen, rain gear, etc.)
"""


async def tourAgent(state: Dict[str, Any]):
    context = f"User Query: {state['topic']}\n\nGathered Intelligence:\n{state['knowledge']}"
    start_time = time.time()

    try:
        response = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=touragentprompt),
                HumanMessage(content=context)
            ]),
            timeout=45
        )
        summary_content = response.content
    except Exception as e:
        print(f"⚠️ tourAgent LLM call exception: {e}")
        knowledge_text = "\n\n".join([str(k) for k in state.get('knowledge', [])])
        summary_content = (
            f"## 🎯 Trip Summary for {state.get('topic', 'Your Trip')}\n\n"
            f"{knowledge_text}"
        )

    print(f"✅ Tour Agent synthesized final report in {time.time() - start_time:.2f}s")
    return {"summary": summary_content}


# ── LangGraph Pipeline ─────────────────────────────────────────────
async_graph = StateGraph(ArticleResponse)
async_graph.add_node("weatherAgent", weatherAgent)
async_graph.add_node("airbnbAgent", airbnbAgent)
async_graph.add_node("tourAgent", tourAgent)

async_graph.add_edge(START, "weatherAgent")
async_graph.add_edge(START, "airbnbAgent")
async_graph.add_edge("weatherAgent", "tourAgent")
async_graph.add_edge("airbnbAgent", "tourAgent")
async_graph.add_edge("tourAgent", END)

app = async_graph.compile(checkpointer=InMemorySaver())


# ── Thread-Safe Sync Wrapper for Streamlit ─────────────────────────
async def _run_app_async(topic: str, thread_id: str, callbacks: list):
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "run_name": "tour_agent",
    }

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        text_placeholder = st.empty()
        full_text = ""

        start_time = time.time()
        current_label = "🚀 Processing Tour Guide Request..."
        status_placeholder.caption(f"{current_label} (0.0s)")

        async for event in app.astream_events(input={"topic": topic}, config=config, version="v2"):
            event_type = event.get("event")
            metadata = event.get("metadata", {})
            node = metadata.get("langgraph_node")

            if event_type == "on_chain_start" and node:
                if node == "weatherAgent":
                    current_label = "🌤️ Weather Agent checking forecast..."
                elif node == "airbnbAgent":
                    current_label = "🏠 Airbnb Agent searching stays..."
                elif node == "tourAgent":
                    current_label = "🧭 Tour Agent synthesizing final itinerary..."

                elapsed = time.time() - start_time
                status_placeholder.caption(f"{current_label} ({elapsed:.1f}s)")

            if event_type == "on_chain_end" and node:
                print(f"✅ Node '{node}' completed")

            if (
                event_type == "on_chat_model_stream"
                and node == "tourAgent"
            ):
                chunk = event["data"]["chunk"].content
                full_text += chunk
                elapsed = time.time() - start_time
                status_placeholder.caption(f"{current_label} ({elapsed:.1f}s)")
                text_placeholder.markdown(full_text, unsafe_allow_html=True)
                await asyncio.sleep(0.01)

        status_placeholder.empty()

    return full_text


def sync_app(topic: str, thread_id: str, callbacks: list):
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(_run_app_async(topic, thread_id, callbacks))).result()
    else:
        return asyncio.run(_run_app_async(topic, thread_id, callbacks))


def tourChat():
    if not st.session_state.get('logged_in'):
        st.warning("Please log in to access this feature.")
        return

    session_key = "tour_agent_messages"
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    for message in st.session_state[session_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    selected_prompt = None
    if not st.session_state[session_key]:
        st.write("💡 **Suggested Travel Queries:**")
        cols = st.columns(2)
        suggestions = [
            ("🌴 Munnar 3-Day Trip", "3 days trip to Munnar from tomorrow"),
            ("⛰️ Manali Weekend Getaway", "4 days weekend getaway to Manali for 2 adults"),
            ("🌊 Goa Beach Vacation", "3 days relaxing beach vacation in Goa with weather forecast"),
            ("🏰 Jaipur Cultural Tour", "5 days heritage and culture tour in Jaipur starting next Monday")
        ]
        for idx, (label, prompt_text) in enumerate(suggestions):
            with cols[idx % 2]:
                if st.button(label, key=f"tour_sug_{idx}", use_container_width=True):
                    selected_prompt = prompt_text

    user_input = st.chat_input("Ask about your trip (e.g. 3 days trip to Munnar from tomorrow)...")
    prompt = selected_prompt or user_input

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state[session_key].append({"role": "user", "content": prompt})

        thread_id = str(uuid.uuid4())
        callbacks = [langfuse_handler] if langfuse_handler else []
        response = sync_app(prompt, thread_id, callbacks)
        st.session_state[session_key].append({"role": "assistant", "content": response})
