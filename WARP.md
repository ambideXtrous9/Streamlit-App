# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common commands

- Install dependencies
  ```bash path=null start=null
  pip install -r requirements.txt
  ```
- Run the Streamlit app (default port 8501)
  ```bash path=null start=null
  streamlit run app.py
  ```
- Open a specific feature page (via query param)
  ```bash path=null start=null
  streamlit run app.py -- "?page=tourAgent"
  # pages: home | stockscreener | newsqa | tourAgent | yolologo | image_classifer | clusterplay | Social | login
  ```
- Run with Docker Compose (app served on :8051)
  ```bash path=null start=null
  docker compose up --build
  ```

Notes
- Linting: no linter configuration found in repo (ruff/flake8/isort/mypy not present).
- Tests: no test suite found (pytest/tox not present).

## Required secrets and local data

The app expects Streamlit secrets and some on-disk assets:

- Streamlit secrets (required for features):
  ```toml path=null start=null
  # .streamlit/secrets.toml
  GROQ_API_KEY = "..."              # used by LangChain ChatGroq across agents
  WEATHER_API_KEY = "..."           # used by Airbnb Tour Agent weather tool
  LANGFUSE_PUBLIC_KEY = "..."       # observability in app.py
  LANGFUSE_SECRET_KEY = "..."
  ```
- Local artifacts bundled in the repo and used at runtime:
  - HPVdb/: FAISS index (HarryAgent retrieval)
  - ImageClassifier/*.ckpt: model checkpoints
  - LogoYolo/LogoYolobest.pt: YOLO weights
  - SQLite files created on first run:
    - users.db (auth) • checkpoints.sqlite (LangGraph checkpoints)

## Architecture overview

High level
- Single Streamlit app (entry: app.py) with sidebar-driven navigation and URL query param persistence.
- Feature pages are routed through navigate.py and implemented in subpackages:
  - StockScreener (AI equity research)
  - HarryAgent (LLM multi-agent graph with retrieval and critique loop)
  - AirbnbAgent (MCP-powered hotel search + weather + tour synthesis)
  - LogoYolo (YOLOv8.1 logo detection)
  - ImageClassifier (multi-backbone logo classifier demo)
  - Clustering (toy clustering visualizations)

Entry and navigation
- app.py
  - Initializes Langfuse client; sets Streamlit config; disables file watcher.
  - Imports and calls SideBar() (sidebar.py) and navigator() (navigate.py).
  - Session state: logged_in, username; persists via query params (logged_in_user, page).
- sidebar.py
  - Renders navigation buttons; updates session state and query params; guards protected pages by redirecting to login when not authenticated.
- navigate.py
  - Switches pages using st.session_state['page'] or URL query param 'page'.
  - Routes to functions in feature modules (e.g., StockScan, ChatBot, tourChat, model_card).

Authentication
- login.py + auth_utils.py
  - Simple SQLite-backed auth with bcrypt; creates users table and a default admin user.
  - Writes users.db in repo root; login/signup flows set session state and persist username in URL.

HarryAgent (Harry Potter × Indian Mythology)
- Files: HarryAgent/HpAgent.py, HarryAgent/RouterAgent.py, HarryAgent/chatbot.py
- Graph (LangGraph) with state AgentState and nodes:
  - classify (RouterAgent.AgentClassifyNode via classify_node): routes generic/harry/exit
  - researcher: create_agent with retrieve_context tool over FAISS (HPVdb) + CrossEncoder reranking
  - mythologist: create_agent with DuckDuckGoSearchResults tool to relate topic to Indian mythology
  - writer: create_agent to compose article (ReAct-style prompt)
  - critic: reviews draft; conditional edge loops back to mythologist until approved
- Checkpointing/observability
  - SqliteSaver(checkpoints.sqlite) for graph checkpoints
  - Langfuse CallbackHandler passed in chatbot.ChatBot() invocation

Airbnb Tour Agent (MCP + Weather)
- Files: AirbnbAgent/tourAgent.py, AirbnbAgent/Implementation.md
- Parallel agents composed in LangGraph (START → weatherAgent and airbnbAgent → tourAgent → END):
  - airbnbAgent: launches MCP stdio session to @openbnb/mcp-server-airbnb via npx; tools loaded by load_mcp_tools(); uses ChatGroq LLM and a structured hotel listing prompt
  - weatherAgent: exposes @tool("WeatherForecast") using WeatherAPI; formats a markdown weather report
  - tourAgent: synthesizes Airbnb + weather knowledge into a strict markdown trip plan
- Streaming to UI: sync_app() streams only tourAgent node deltas via app.astream_events(version="v2") and renders incrementally.
- Node setup:
  - PATH is prepended with repo-local nodev20/bin; subprocess checks for node/npx; Docker also installs Node 20.

Stock Screener
- Files: StockScreener/screener.py (+ CSV lists, mlpchart)
- Data sources: yfinance price history; screener.in HTML scraping for fundamentals/shareholding; GNews for news.
- Analytics: computes technical indicators (EMA/SMA/RSI/MACD), breakout detection, valuation/financial tables, multibagger heuristics.
- LLM: ChatGroq used by a stock_agent (create_agent) to write broker-style report; functions decorated with @observe for Langfuse.

Computer Vision demos
- LogoYolo/inference.py: ultralytics.YOLO with bundled weights for logo detection; integrated in util.YoloforLogo()
- ImageClassifier/classifier.py: loads four classifier checkpoints (Xception, InceptionV3, MobileNetV2, EfficientNet); runs inference on uploaded image and displays per-model metrics.

Clustering
- Clustering/clusterapp.py: generates synthetic circular clusters; visualizes KMeans/DBSCAN and K-distance graph via seaborn/matplotlib.

## Feature-specific run hints

- Tour Agent only (ensure secrets + Node available)
  ```bash path=null start=null
  streamlit run app.py -- "?page=tourAgent"
  # If MCP server fails, ensure Node 20 + npx are on PATH or use the bundled nodev20/.
  ```
- HarryAgent only
  ```bash path=null start=null
  streamlit run app.py -- "?page=newsqa"
  ```
- Stock Screener only
  ```bash path=null start=null
  streamlit run app.py -- "?page=stockscreener"
  ```

## Docker

- Compose builds and serves on http://localhost:8051.
- The Dockerfile installs Node 20 for MCP; if build fails due to sudo not found in slim image, replace the Node setup with a non-sudo variant or rely on the repo’s nodev20/ directory.
  ```bash path=null start=null
  docker compose up --build
  ```

## References from README
- Key features: Stock Screener, HarryAgent, Logo Detection, Clustering, Image Classifier
- Quick start (local): pip install -r requirements.txt && streamlit run app.py
