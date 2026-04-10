from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
import os
from langchain_core.tools import tool
from typing import Any
from langgraph.graph import StateGraph, END, START
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
import requests
import uuid
from langfuse.langchain import CallbackHandler
import time
import asyncio
import sys
import subprocess
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, Field

# ── Node.js availability check ─────────────────────────────────────
def _check_node():
    for cmd in (["node", "-v"], ["npx", "--version"]):
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    return True

_NPX_AVAILABLE = _check_node()

if not _NPX_AVAILABLE:
    import sys as _sys
    if _sys.platform.startswith("linux"):
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
    node_version = subprocess.run(["node", "-v"], capture_output=True, text=True, timeout=5)
    print(f"✅ Node.js {node_version.stdout.strip()} available — Airbnb MCP enabled")

# Langfuse handler
try:
    langfuse_handler = CallbackHandler()
except Exception:
    langfuse_handler = None


class ArticleResponse(TypedDict):
    topic: str
    summary: str
    knowledge: Annotated[list[AnyMessage], add_messages]


os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

model_name = "llama-3.3-70b-versatile"
temperature = 0.0

llm = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    seed=42,
    tags=["TourAgentExpert"]
)


# ── Airbnb Agent ──────────────────────────────────────────────────
async def airbnbAgent(state):
    print(f"🏠 Airbnb Agent: {state.get('topic', 'unknown')}")
    start = time.time()

    if not _NPX_AVAILABLE:
        ai_content = "⚠️ Airbnb unavailable — Node.js/npx not installed."
    else:
        try:
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
            )

            # Fix: pass a real file to stdio_client's errlog to avoid fileno error
            with open(os.devnull, "w") as err_log:
                async with stdio_client(server_params, errlog=err_log) as (read, write):
                    async with ClientSession(read, write) as session:
                        print("Initializing connection...")
                        await session.initialize()

                        print("Loading tools...")
                        tools = await load_mcp_tools(session)
                        print(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")

                        for t in tools:
                            print(f"Tool: {t.name}, args_schema: {t.args_schema}")

                        agent = create_react_agent(
                            llm,
                            tools,
                            prompt=(
                                """
                                **ADVANCED HOTEL SEARCH FORMAT**

                                ## 🎯 Search Summary
                                - **Location:** [location] | **Dates:** [checkin] → [checkout]
                                - **Guests:** [adults]A, [children]C, [infants]I, [pets]P
                                - **Room:** [room type] | **Stars:** [rating] | **Amenities:** [amenities]
                                - **Results:** [number] hotels

                                ## 🏨 Hotel Listings
                                ### [Hotel Name]
                                | Detail | Info |
                                |--------|------|
                                | ⭐ Rating | [rating]/5 ([reviews]) |
                                | 📍 Address | [full address] |
                                | 💰 Rate | $[price]/night (+$[tax]) |
                                | 🏠 Rooms | [categories] |
                                | 📏 Distance | [city center] • [airport] |
                                | 🔗 Booking | [URL] |
                                | 📞 Contact | [phone] • [website] |

                                **Amenities:** [pool/gym/spa, dining, transport, business, pets, WiFi, services]
                                **Booking:** Check-in [time], Check-out [time], Cancellation [policy], Payment [methods], Breakfast [info], Parking [info], Extra Beds [policy]

                                **Match Analysis:** Budget [fit], Amenities [X/Y matched], Location [score], Guest Reviews [highlights]
                                **Recommendations:** Best for [use case], Offers [promos], Tips [advice]

                                -- repeat per hotel --

                                ## 📈 Comparison
                                | Hotel | Rating | Price | Features | Link |
                                |-------|--------|-------|----------|------|
                                | [H1] | [rating]⭐ | $[price] | [2 highlights] | [URL] |
                                | [H2] | [rating]⭐ | $[price] | [2 highlights] | [URL] |

                                ## 🏆 Final Picks
                                - **Best Value:** [hotel + reason]
                                - **Luxury:** [hotel + features]
                                - **Budget:** [hotel + savings]
                                - **Location:** [hotel + benefit]
                                - **Amenities:** [hotel + standout]
                                """
                            ),
                        )

                        print(f"Invoking agent with query: {state['topic']}")
                        response = await agent.ainvoke({"messages": [{"role": "user", "content": state['topic']}]})

                        # Debug: print ALL messages to see tool calls and results
                        for msg in response.get("messages", []):
                            if hasattr(msg, "type"):
                                print(f"  [{msg.type}] {msg.content[:500]}")

                        ai_content = response["messages"][-1].content
                        print(f"Final agent response: {ai_content[:1000]}")
        except Exception as e:
            ai_content = f"⚠️ Airbnb agent failed: {type(e).__name__}: {e}"

    print(f"✅ Airbnb Agent done in {time.time() - start:.2f}s")

    with st.spinner("Airbnb Agent in Progress…", show_time=True):
        pass

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
    API_KEY = st.secrets["WEATHER_API_KEY"]
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
        return None


# ── Weather Agent ─────────────────────────────────────────────────
async def weatherAgent(state):
    print("Weather Agent..")
    weather_agent = create_react_agent(
        llm,
        [get_forecast],
        prompt=(
            "You are a Weather Assistant.\n"
            "Use the **WeatherForecast tool** to fetch the forecast.\n"
            "Generate a **Weather Report** strictly following this format:\n\n"
            "## Weather Report for <Location> (Next <Days> Days)\n\n"
            "**Current Conditions:** <Temp>°C, <Condition>\n\n"
            "**Forecast Summary:**\n"
            "- **<Date>:** <Condition>, <Max>/<Min>°C\n"
            "(repeat for each day)\n\n"
            "**Tour Recommendation:**\n"
            "Based on weather, state if it's a good time to visit.\n"
            "Give practical advice: clothing, precautions, activities."
        ),
    )

    start = time.time()
    with st.spinner("Weather Agent in Progress…", show_time=True):
        result = weather_agent.invoke({"messages": [{"role": "user", "content": state['topic']}]})
    ai_content = result["messages"][-1].content
    print(f"✅ Weather Agent done in {time.time() - start:.2f}s")
    return {"knowledge": [f"[Info from Weather Search]\n{ai_content}\n\n"]}


# ── Tour Agent (final summarizer) ─────────────────────────────────
touragentprompt = """
You are a Travel & Tour Guide Assistant. Suggest a tour plan based on the user query and the Airbnb and Weather reports.

**Strictly follow the Markdown Output Format below.**

---

## 🎯 Search Summary
- **Location:** [location] | **Dates:** [checkin] → [checkout]
- **Guests:** [adults]A, [children]C, [infants]I, [pets]P
- **Room:** [room type] | **Stars:** [rating] | **Amenities:** [amenities]
- **Results:** [number] hotels

---

## 🏨 Hotel Listings

### [Hotel Name]
| Detail | Info |
|--------|------|
| ⭐ Rating | [rating]/5 ([reviews]) |
| 📍 Address | [full address] |
| 💰 Rate | $[price]/night (+$[tax]) |
| 🏠 Rooms | [categories] |
| 📏 Distance | [city center] • [airport] |
| 🔗 Booking | [URL] |
| 📞 Contact | [phone] • [website] |

---

### 🏆 Final Picks
- **Best Value:** [hotel + reason]
- **Luxury:** [hotel + features]
- **Budget:** [hotel + savings]
- **Location:** [hotel + benefit]
- **Amenities:** [hotel + standout]

---

### 🌤️ Weather Summary
- **Location:** [location]
- **Duration:** [days] Days ([checkin] → [checkout])
- **Current:** 🌡️ [current_temp]°C, [condition] | Feels like [feelslike]°C
- **Humidity:** [humidity]% | **Wind:** [wind] kph | **Pressure:** [pressure] mb

---

### 📅 3-Day Forecast
| Date       | Condition | 🌡️ Max Temp | 🌡️ Min Temp | 💧 Humidity | 💨 Max Wind |
|------------|-----------|-------------|-------------|-------------|-------------|
| [date1]    | [cond1]   | [max1]°C    | [min1]°C    | [hum1]%     | [wind1] kph |
| [date2]    | [cond2]   | [max2]°C    | [min2]°C    | [hum2]%     | [wind2] kph |
| [date3]    | [cond3]   | [max3]°C    | [min3]°C    | [hum3]%     | [wind3] kph |

---

### 🧭 Travel & Stay Insights
- **Clothing:** [clothing advice]
- **Precautions:** [safety advice]
- **Activities:** [indoor/outdoor suggestions]
- **Stay Match:** Recommend [cozy apartments / mountain-view homes / family stays] based on forecast

---

### 🏡 Stay Match with Weather
- **If Rainy/Cloudy:** Cozy Airbnbs (like {PropertyX}) are recommended since you'll spend more time indoors. Look for ones with indoor seating, tea/coffee facilities, or scenic balconies to enjoy the misty views.
- **If Sunny/Clear:** Open-view Airbnbs (like {PropertyY}) are better, offering outdoor seating and mountain/sunset views.
- **If Mixed Weather:** Balanced stays (like {PropertyZ}) with both indoor comfort and outdoor access give flexibility.

### 🌟 Alternative Travel Note
{AlternativeAdvice}
---

"""


async def tourAgent(state):
    context = f"Based on User Query : {state['topic']} \nanalyze below Reports on Airbnb and Weather: {state['knowledge']}"

    start_time = time.time()

    with st.spinner("Tour Agent in Progress…", show_time=True):
        response = llm.invoke([
            SystemMessage(content=touragentprompt),
            HumanMessage(content=context)
        ])

    end_time = time.time()
    tour_time = end_time - start_time

    return {"summary": response.content}


# ── Graph ─────────────────────────────────────────────────────────
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


def sync_app(topic, thread_id, callbacks):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_app():
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks,
            "run_name": "tour_agent",
        }

        message_box = st.chat_message("assistant")
        text_placeholder = message_box.empty()
        full_text = ""

        async for event in app.astream_events(input={"topic": topic}, config=config, version="v2"):
            if event["event"] == "on_chain_end" and event.get("metadata", {}).get("langgraph_node"):
                node = event["metadata"]["langgraph_node"]
                print(f"✅ Node '{node}' completed")

            if (
                event["event"] == "on_chat_model_stream"
                and event["metadata"].get("langgraph_node") == "tourAgent"
            ):
                chunk = event["data"]["chunk"].content
                full_text += chunk
                text_placeholder.markdown(full_text)
                await asyncio.sleep(0.01)

        return full_text

    try:
        result = loop.run_until_complete(run_app())
        return result
    finally:
        loop.close()


def tourChat():
    if not st.session_state.get('logged_in'):
        st.warning("Please log in to access this feature.")
        return

    session_key = "tour_agent_messages"
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    for message in st.session_state[session_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state[session_key].append({"role": "user", "content": prompt})

        thread_id = str(uuid.uuid4())
        callbacks = [langfuse_handler] if langfuse_handler else []
        response = sync_app(prompt, thread_id, callbacks)
        st.session_state[session_key].append({"role": "assistant", "content": response})
