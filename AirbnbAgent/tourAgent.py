from langchain_groq import ChatGroq
from langchain.agents.factory import create_agent
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
import json

# ── Node.js availability check ─────────────────────────────────────
# - Docker: Node.js 20.x is installed via apt during build
# - Local dev: Uses system Node.js (brew/nvm)
# - Falls back to apt install on Linux if not found

def _check_node():
    """Check if node/npx works."""
    for cmd in (["node", "-v"], ["npx", "--version"]):
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    return True

# Fast path: already available
_NPX_AVAILABLE = _check_node()

# Fallback: install Node.js via apt on Linux only if not already available
if not _NPX_AVAILABLE:
    import sys as _sys
    if _sys.platform.startswith("linux"):
        print("⚠️ Node.js not found, installing via apt-get...")
        try:
            r1 = subprocess.run(
                ["apt-get", "update", "-qq"],
                check=False, capture_output=True, timeout=120
            )
            print(f"  apt-get update: rc={r1.returncode}")
            r2 = subprocess.run(
                ["apt-get", "install", "-y", "-qq", "nodejs", "npm"],
                check=False, capture_output=True, timeout=180
            )
            print(f"  apt-get install nodejs npm: rc={r2.returncode}")
            if r2.returncode != 0:
                err = r2.stderr.decode()[:500]
                print(f"  apt-get stderr: {err}")
            _NPX_AVAILABLE = _check_node()
        except Exception as e:
            print(f"apt-get fallback error: {e}")

if not _NPX_AVAILABLE:
    print("⚠️ Airbnb MCP requires Node.js/npx. Install it to enable this feature.")
else:
    node_version = subprocess.run(["node", "-v"], capture_output=True, text=True, timeout=5)
    print(f"✅ Node.js {node_version.stdout.strip()} available — Airbnb MCP enabled")

# Langfuse handler: graceful if not configured
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
    seed = 42,
    tags=["TourAgentExpert"]
)   


#llm = ChatOllama(model="qwen3:4b")  # Use a model available via Ollama


async def airbnbAgent(state):
    print(f"🏠 Airbnb Agent starting for: {state.get('topic', 'unknown')}")

    if not _NPX_AVAILABLE:
        msg = "⚠️ Airbnb search is unavailable because Node.js/npx is not functional on this system."
        print(f"⚠️ {msg}")
        return {"knowledge": [msg]}

    # Use the standalone script via subprocess to avoid Streamlit stdio wrapping issues
    script_path = os.path.join(os.path.dirname(__file__), "airbnb_search.py")
    venv_python = os.path.join(os.path.dirname(__file__), "..", ".rp360", "bin", "python")
    
    # Ensure we use the venv Python that has all dependencies
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    
    print(f"🔧 Running Airbnb search via subprocess: {python_exec} {script_path}")
    
    try:
        start_time = time.time()
        
        # Run the standalone script as subprocess
        result = await asyncio.to_thread(
            subprocess.run,
            [python_exec, script_path, state['topic']],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        end_time = time.time()
        print(f"✅ Airbnb subprocess completed in {end_time - start_time:.2f}s")
        
        # Log stderr for debugging
        if result.stderr:
            print(f"📝 Airbnb stderr: {result.stderr[:500]}")
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            print(f"⚠️ Airbnb script failed (rc={result.returncode}): {error_msg[:300]}")
            return {"knowledge": [f"⚠️ Airbnb search failed: {error_msg[:200]}"]}
        
        # Parse JSON output from stdout
        try:
            output = json.loads(result.stdout)
            if output.get("success"):
                ai_content = output["result"]
                return {"knowledge": [f"[Info from AirBnb Search]\n{ai_content}\n\n"]}
            else:
                error_msg = output.get("error", "Unknown error")
                return {"knowledge": [f"⚠️ Airbnb search error: {error_msg}"]}
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON output: {e}")
            print(f"  stdout: {result.stdout[:200]}")
            return {"knowledge": [f"⚠️ Failed to parse Airbnb search results: {result.stdout[:200]}"]}
        
    except subprocess.TimeoutExpired:
        print("⚠️ Airbnb search timed out after 120s")
        return {"knowledge": ["⚠️ Airbnb search timed out. The search took too long to complete."]}
    except Exception as e:
        print(f"⚠️ Airbnb subprocess failed: {type(e).__name__}: {e}")
        return {"knowledge": [f"⚠️ Airbnb search failed ({type(e).__name__}: {e})."]}



def extract_weather(data: dict) -> str:
    """
    Extracts location, current weather, and 3-day forecast info from WeatherAPI JSON.
    Returns as a formatted string instead of printing.
    """
    lines = []

    # --- Location info ---
    loc = data.get("location", {})
    location = f"{loc.get('name')}, {loc.get('region')}, {loc.get('country')}"
    lines.append(f"📍 Location: {location}")

    # --- Current weather ---
    current = data.get("current", {})
    condition = current.get("condition", {}).get("text")
    temp_c = current.get("temp_c")
    feelslike_c = current.get("feelslike_c")
    humidity = current.get("humidity")
    gust_kph = current.get("gust_kph")
    pressure_mb = current.get("pressure_mb")

    lines.append("\n🌤️ Current Weather:")
    lines.append(f"  Temp: {temp_c}°C (Feels like {feelslike_c}°C)")
    lines.append(f"  Condition: {condition}")
    lines.append(f"  Humidity: {humidity}%")
    lines.append(f"  Wind Gust: {gust_kph} kph")
    lines.append(f"  Pressure: {pressure_mb} mb")

    # --- Forecast info ---
    forecast = data.get("forecast", {}).get("forecastday", [])
    lines.append("\n📅 Forecast:")
    for day in forecast:
        d = day.get("date")
        details = day.get("day", {})
        lines.append(f"  Date: {d}")
        lines.append(f"    Condition: {details.get('condition', {}).get('text')}")
        lines.append(f"    Max Temp: {details.get('maxtemp_c')}°C")
        lines.append(f"    Min Temp: {details.get('mintemp_c')}°C")
        lines.append(f"    Avg Humidity: {details.get('avghumidity')}%")
        lines.append(f"    Max Wind: {details.get('maxwind_kph')} kph")
        lines.append("-" * 40)

    return "\n".join(lines)



from pydantic import BaseModel, Field

class WeatherArgs(BaseModel):
    location: str = Field(description="City name or coordinates")
    days: Any = Field(default=3, description="Number of days to forecast (integer, e.g. 3)")

@tool("WeatherForecast", args_schema=WeatherArgs)
def get_forecast(location: str, days: int = 3):
    """Fetch weather forecast for a given location using WeatherAPI."""
    days = int(days)  # Ensure int even if LLM passes string
    print("WeatherForecast tool")
    API_KEY = st.secrets["WEATHER_API_KEY"]
    
    params = {
        "key": API_KEY,
        "q": location,
        "days": days,
        "aqi": "no",
        "alerts": "yes"
    }

    BASE_URL = f"http://api.weatherapi.com/v1/forecast.json?key={params['key']}&q={params['q']}&days={params['days']}&aqi={params['aqi']}&alerts={params['alerts']}"

    try:
        response = requests.get(BASE_URL, timeout=10)
        response.raise_for_status()  # raise error for bad status
        data = response.json()
        forecast = extract_weather(data)
        return forecast
    except requests.RequestException as e:
        print(f"Error fetching forecast: {e}")
        return None


async def weatherAgent(state):
    print("Weather Agent..")
    weather_agent = create_agent(
            model=llm,
            tools=[get_forecast],
            system_prompt = (
                """
                You are a Weather Assistant.

                - Use the **WeatherForecast tool** to fetch the forecast for the location mentioned in the query.  
                - Then generate a **Weather Report** strictly following the Markdown format below.  
                - Do not add extra sections, explanations, or text outside the format.  
                - Fill in <placeholders> with actual values.  

                ## Weather Report for <Location> (Next <Days> Days)

                **Current Conditions:** <CurrentTemp>°C with <CurrentCondition> (<Rain/Heatwave/Clear/Other summary>)

                **Forecast Summary:**
                - **<Date 1>:** <Condition>, <MaxTemp>°C / <MinTemp>°C (<Rain/Heatwave/Clear/Other summary>)
                - **<Date 2>:** <Condition>, <MaxTemp>°C / <MinTemp>°C (<Rain/Heatwave/Clear/Other summary>)
                - **<Date 3>:** <Condition>, <MaxTemp>°C / <MinTemp>°C (<Rain/Heatwave/Clear/Other summary>)
                ... (repeat for all forecast days)

                **Tour Recommendation:**  
                Based on the weather forecast, state clearly if it is a good time to visit <Location>. 
                Give practical advice: clothing, precautions, indoor/outdoor activity suggestions.
                """
            ),
            name="Weather Forecast Assistant",
        )

    start_time = time.time()
    
    with st.spinner("Weather Agent in Progress…", show_time=True):
        result = weather_agent.invoke({"messages": [{"role": "user", "content": state['topic']}]})
    ai_content = result["messages"][-1].content

    end_time = time.time()
    weather_time = end_time - start_time
    
    # with st.chat_message("Agent"):
    #     st.markdown(f"**✅ Weather Agent Time :** {weather_time:.2f} seconds\n")
    
    return {"knowledge": [f"[Info from Weather Search]\n{ai_content}\n\n"]}



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
- **If Rainy/Cloudy:** Cozy Airbnbs (like {PropertyX}) are recommended since you’ll spend more time indoors. Look for ones with indoor seating, tea/coffee facilities, or scenic balconies to enjoy the misty views.  
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
    # with st.chat_message("Agent"):
    #     st.markdown(f"**✅ Tour Agent Time :** {tour_time:.2f} seconds\n")

    return {"summary": response.content}


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
            # Debug: log node completion
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
                await asyncio.sleep(0.01)  # push updates

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
