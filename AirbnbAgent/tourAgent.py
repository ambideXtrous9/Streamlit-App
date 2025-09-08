from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import streamlit as st
import os
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import tool
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
from streamlit.runtime.scriptrunner import get_script_run_ctx
import asyncio

langfuse_handler = CallbackHandler()

class ArticleResponse(TypedDict):
    topic: str
    summary: str
    knowledge: Annotated[list[AnyMessage], add_messages]
    
   
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


model_name = "moonshotai/kimi-k2-instruct" #"qwen/qwen3-32b" #
temperature = 0.0


llm = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    seed = 42,
    tags=["TourAgentExpert"]
)   


#llm = ChatOllama(model="qwen3:4b")  # Use a model available via Ollama


async def airbnbAgent(state):
    # Get the current Streamlit context
    ctx = get_script_run_ctx()
    
    server_params = StdioServerParameters(
            command= "./nodev20/bin/npx",
            args= [
                "-y",
                "@openbnb/mcp-server-airbnb",
                "--ignore-robots-txt"
            ],
        )
    
    
    async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("Initializing connection...")
                await session.initialize()

                # Get tools
                print("Loading tools...")
                tools = await load_mcp_tools(session)
                
                
                agent = create_react_agent(
                    model=llm,
                    tools=tools,
                    prompt = (
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
                    )
                )

                start_time = time.time()
    
                with st.spinner("Airbnb Agent Node in Progress…", show_time=True):
                    response = await agent.ainvoke({"messages": [{"role": "user", "content": state['topic']}]})

                ai_content = response["messages"][-1].content

                end_time = time.time()
                airbnb_time = end_time - start_time
                
                # with st.chat_message("Agent"):
                #     st.markdown(f"**✅ AirBnb Agent Time :** {airbnb_time:.2f} seconds\n")
                
                return {"knowledge": [f"[Info from AirBnb Search]\n{ai_content}\n\n"]}
                


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



@tool("WeatherForecast")
def get_forecast(location: str, days: int = 3):
    """
    Fetch weather forecast for a given location using WeatherAPI.

    Args:
        location (str): City name or coordinates (e.g., "London" or "51.5072,-0.1276").
        days (int): Number of days to forecast (default = 3).

    Returns:
        dict: Forecast data if successful, None otherwise.
    """
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
    weather_agent = create_react_agent(
            model=llm,
            tools=[get_forecast],
            prompt = (
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
        config = {"thread_id": thread_id, "callbacks": [callbacks], "run_name": "tour_agent"}
        
        message_box = st.chat_message("assistant")
        text_placeholder = message_box.empty()
        full_text = ""

        async for event in app.astream_events(input={"topic": topic}, config=config, version="v2"):
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
        response = sync_app(prompt, thread_id, langfuse_handler)
        st.session_state[session_key].append({"role": "assistant", "content": response})
