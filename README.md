# Streamlit Applications Hub

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue?logo=streamlit)](https://ambidextrous.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![image](https://github.com/user-attachments/assets/1f390447-0618-4523-9655-9b99f0ec93e3)


## 🚀 About This Project

A curated hub of interactive Streamlit applications showcasing practical AI and data science. Each app is designed for clarity, fast iteration, and hands-on exploration.

## 📋 Key Features

- 📊 **Stock Screener**: AI-assisted equity research with indicators, charts, and reports
- 🗺️ **Airbnb Tour Agent (MCP + Weather)**: Trip plans from Airbnb listings + weather via MCP
- 📖 **HarryAgent**: LLM multi-agent research and writing with retrieval + critique loop
- 🎯 **Logo Detection**: YOLO-based logo recognition
- 📈 **Clustering**: Interactive clustering demos (KMeans/DBSCAN)
- 🧠 **Image Classifier**: Multi-backbone image classification

## 📁 Project Structure

```
Streamlit-AI-Portfolio/
├── StockScreener/        # Stock market analysis tools
├── HarryAgent/           # Harry Potter X Mythology
├── LogoYolo/            # Logo detection system
├── Clustering/          # Clustering algorithms
├── HPVdb/              # Harry Potter Books
├── app.py              # Main Streamlit application
└── requirements.txt    # Project dependencies
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/ambideXtrous9/Streamlit-App.git

cd Streamlit-App

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🎮 Usage

- Launch: `streamlit run app.py` → open `http://localhost:8501`
- Navigate via the sidebar between applications
- Open a specific page directly:

```bash
streamlit run app.py -- "?page=stockscreener"
# pages: home | stockscreener | newsqa | tourAgent | yolologo | image_classifer | clusterplay | login
```

## 📚 Project Details

### 📊 AI Stock Screener
- Real-time price data via yfinance
- Fundamentals/shareholding scraped from screener.in
- News context via GNews
- Technical indicators: EMA, SMA, RSI, MACD; breakout heuristics
- Interactive tables and charts (mlpchart)
- AI stock research report powered by ChatGroq

### 🗺️ Airbnb Tour Agent (MCP + Weather)
- LangGraph composition: parallel weather agent + Airbnb MCP agent → tour synthesis
- MCP stdio to `@openbnb/mcp-server-airbnb` via `npx`
- Weather tool formats a concise forecast report
- Streams tour synthesis updates to the UI
- Uses system Node 20 or bundled `nodev20/` for reliable MCP execution

### 📖 HarryAgent (LLM multi-agent, RAG + critique)
- Thematic blend: Harry Potter × Indian Mythology
- LangGraph workflow: classify → researcher → mythologist → writer → critic (with loop)
- Retrieval via FAISS index in `HPVdb/` with CrossEncoder reranking
- Checkpointing in SQLite; observability via Langfuse

### 🧠 Image Classifier
- Multiple backbones: Xception, InceptionV3, MobileNetV2, EfficientNet
- Loads checkpoints (`*.ckpt`) and runs per-model inference
- Displays per-model metrics for comparison

### 🎯 Logo Detection
- Ultralyics YOLO-based logo recognition with bundled weights (`LogoYolobest.pt`)
- Integrated into the app via shared utilities
- Real-time inference on uploaded images

### 📈 Clustering
- KMeans and DBSCAN demos with synthetic data
- Interactive visualizations with seaborn/matplotlib
- K-distance graph to explore cluster structure




## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - For the amazing framework
- [Python](https://www.python.org/) - The language that makes it all possible
- [Machine Learning Community](https://www.kaggle.com/) - For continuous learning and inspiration

## 📞 Contact

For any questions or collaborations, feel free to reach out!

---

🌟 Star this repository if you find it useful!

[![GitHub stars](https://img.shields.io/github/stars/ambideXtrous9/Streamlit-App.svg?style=social&label=Star)](https://github.com/ambideXtrous9/Streamlit-App)
[![GitHub forks](https://img.shields.io/github/forks/ambideXtrous9/Streamlit-App.svg?style=social&label=Fork)](https://github.com/ambideXtrous9/Streamlit-App/fork)
[![GitHub watchers](https://img.shields.io/github/watchers/ambideXtrous9/Streamlit-App.svg?style=social&label=Watch)](https://github.com/ambideXtrous9/Streamlit-App/watchers)
[![GitHub followers](https://img.shields.io/github/followers/ambideXtrous9.svg?style=social&label=Follow)](https://github.com/ambideXtrous9/followers)
