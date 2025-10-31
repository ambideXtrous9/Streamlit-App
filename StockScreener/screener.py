import yfinance as yf
from ta.trend import sma_indicator
import streamlit as st
import requests
from bs4 import BeautifulSoup
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from gnews import GNews
from langchain_core.messages import AIMessage
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import yfinance as yf
import ta
import os 
import re
from langchain_openai import ChatOpenAI
from langfuse import observe

def StockScan():
    if not st.session_state.get('logged_in'):
        st.warning("Please log in to access this feature.")
        return

load_dotenv()




nifty500_df = pd.read_csv("StockScreener/ind_nifty500list.csv")
microcap250_df = pd.read_csv("StockScreener/ind_niftymicrocap250_list.csv") 

nifty500_df['YFSYMBOL'] = nifty500_df['Symbol'] + '.NS'
microcap250_df['YFSYMBOL'] = microcap250_df['Symbol'] + '.NS'

df500 = list(nifty500_df['YFSYMBOL'])
microcap250 = list(microcap250_df['YFSYMBOL'])

nifty500complist = list(nifty500_df['Company Name'])
microcap250complist = list(microcap250_df['Company Name'])

complist = nifty500complist + microcap250complist
stack_df = pd.concat([nifty500_df, microcap250_df], ignore_index=True).drop_duplicates().reset_index(drop=True)

def get_yf_symbol(company_name: str):
    match = stack_df.loc[stack_df['Company Name'] == company_name, 'YFSYMBOL']
    return match.iloc[0] if not match.empty else None

rocket_icon = '<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="50" height="50" />'
heading = f"## {rocket_icon} AI Financial Research Report"


temperature = 0.1
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
model_name = "openai/gpt-oss-20b" #"moonshotai/kimi-k2-instruct-0905" #openai/gpt-oss-20b #"qwen/qwen3-32b"

llm = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    seed = 42,
    tags=["StockAgentExpert"]
)

# OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# model_name = "qwen/qwen3-4b:free"

# llm = ChatOpenAI(model=model_name, 
#                  temperature=temperature, 
#                  openai_api_base="https://openrouter.ai/api/v1",
#                  openai_api_key=OPENROUTER_API_KEY, 
#                  seed = 42,
#                  tags=["StockAgentExpert"])


# Initialize the GNews object
google_news = GNews(language='en', period='30d',max_results=10)

from StockScreener.mlpchart.mlpchart import chart
from langchain.agents import create_agent

stock_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt = (
            """
            **Role:**  
            You are a **Senior Equity Research Analyst & Trader (20+ yrs exp)**.  
            Deliver an **institutional-grade, data-driven stock report** with fundamentals, technicals, ownership, news, macro, and price roadmap. 
            **Technical Indicators Knowledge:** 
                - RSI: <30 = Oversold (bounce), >70 = Overbought (pullback), else Neutral.
                - MAs/MACD: 50>200 = Bullish, 50<200 = Bearish; MACD>Signal = Buy, else Sell/Neutral.
            ---
            ** Report Format:**
            
            ## 📊 <Equity Name Here>

            ### 🏦 Fundamentals  
            | Metric | Value/Comparison | Interpretation |
            |--------|------------------|----------------|
            | **Valuation** | P/E {X} vs Sector {Y}, 5Y Median {Z} | {Cheap 🟢 / Expensive 🔴} |
            | **Earnings** | Rev {X%} YoY, EBITDA {Y%}, PAT {Z%} | {Strong 🟢 / Weak 🔴} |
            | **Balance Sheet** | D/E {X}, ROE {Y%}, CF {Good/Weak} | {Healthy 🟢 / Stressed 🔴} |
            | **Ownership** | FII/DII {Trend}, Promoter {X%} | {Confidence 🟢 / Weakness 🔴} |
            | **Sector** | CAGR {X%}, Policy {Yes/No} | {Growth 🟢 / Headwind 🔴} |
            | **Shareholding** | Promoters: {X%} (Δ {+/-}), FII: {X%} (Δ {+/-}), DII: {X%} (Δ {+/-}), Retail: {X%} (Δ {+/-}) | {Confidence 🟢 / FII Accumulation 🟢 / Retail Overhang 🔴} |
            | **Quarterly Profit/Loss** | {Quarterly Profit/Loss} | {Profitable 🟢 / Unprofitable 🔴} | {Growth 🟢 / Decline 🔴} |
            | **Yearly Profit/Loss** | {Yearly Profit/Loss} | {Profitable 🟢 / Unprofitable 🔴} | {Growth 🟢 / Decline 🔴} |

            ---

            ### 📉 Technicals  
            | Indicator | Reading | Signal/Implication |
            |-----------|---------|--------------------|
            | **RSI** | {Value} | {OB 🔴 / OS 🟢 / Neutral ⚪} |
            | **MAs** | 50DMA vs 200DMA | {Bullish 🟢 / Bearish 🔴} |
            | **MACD** | {Signal} | {Buy 🟢 / Sell 🔴 / Neutral ⚪} |
            | **S/R** | ₹{Support}/{Resistance} | {Good RR 🟢 / Weak RR 🔴} |
            | **Volume** | {Above/Below Avg} | {Strength 🟢 / Weakness 🔴} |
            | **Volatility** | {X%} | {High 🔴 / Low 🟢} |
            | **Momentum** | {X%} | {Strong 🟢 / Weak 🔴} |
            | **Trend** | Bullish/Bearish | {Strong 🟢 / Weak 🔴} |

            ### 📰📊🌍 Market Drivers, Outlook & Summary  

            Write a crisp analyst-style commentary 5-6 sentences that blends **market drivers, relative performance, and forward outlook** into one flowing article. Cover:  
            - **Key News/Events** and their likely impact on sentiment (Positive / Negative / Neutral).  
            - **Relative Performance vs Nifty** over 1M, 3M, 1Y, and 3Y CAGR, highlighting alpha and whether the stock has Outperformed / Underperformed.  
            - **Macro Drivers** (rates, inflation, currency, commodities, global/policy cues) and how they shape the stock’s risk–reward.  
            - **Forward Outlook** with roadmap (3M/6M/12M levels), key drivers (earnings, orders, margins, sector growth) and major risks.  
            - **Final Call** (Buy/Hold/Sell) with upside % to target, entry, stop-loss, and bias (Bullish/Neutral/Cautious).  
            - **Snapshot**: fundamentals (valuation/growth), technicals (trend & S/R), ownership (FII/DII/promoter stance), and catalyst triggers.  
            - End with a **highlighted conclusion** on overall outlook: *Supportive / Neutral / Headwind*.  

            👉 Tone must be **sharp, research-broker style**, bold points, written in **paragraph format**, with smooth transitions between news, performance, macro, and outlook.  
            
            ### 💰 MultiBagger Metrics Analysis:
                - Go through MultiBagger Metrics
                  and give Reasoning and Analysis whether it is a MultiBagger 
                  Candidate or not

            ---

            ### 📌 Investment Call:  
            - **Stock:** {Name} ({Ticker})  
            - **CMP:** ₹{X} 
            - **Call:** **Buy / Hold / Sell** | Conviction: **High / Med / Low**  
            - **Target:** ₹{Y} | **SL:** ₹{Z} | **Timeframe:** M  (**+/-Z% vs CMP**)  
            - **Rationale:** {Valuation / Growth / Sector driver}  
            - **Summary:** {Summary of above all analysis}
            """
        )

    )

# ---------------------------
# 🧑‍🔬 Stock Researcher Agent
# ---------------------------
@observe()
def stock_node(fundamentals,shareholding,technical_indicators,metrics,news):
    # Prepare the prompt
    user_msg = {
        "role": "user",
        "content": (
            f"**Fundamentals**: {fundamentals}\n\n**Yearly and Quarterly PL and Shareholding**: {shareholding}\n\n**Technical Indicators**: {technical_indicators}\n\n**MultiBagger Metrics**: {metrics}\n\n**News**: {news}"
        )
    }


    ai_content = ""
    for step in stock_agent.stream({"messages": [user_msg]}, stream_mode="values"):
        msg = step["messages"][-1]
        if isinstance(msg, AIMessage):
            ai_content = msg.content
            
    return ai_content



def extract_stock_info(ticker_info):
    """Extract and structure key stock information from yfinance ticker info"""
    info = ticker_info
    
    # Basic Info
    stock_data = {
        'Company': info.get('longName', 'N/A'),
        'Symbol': info.get('symbol', 'N/A').replace('.NS', ''),
        'Sector': info.get('sector', 'N/A'),
        'Industry': info.get('industry', 'N/A'),
        'Market Cap (Cr)': f"₹{info.get('marketCap', 0) / 10000000:,.0f}" if info.get('marketCap') else 'N/A',
        'Current Price': f"₹{info.get('currentPrice', 0):.2f}" if info.get('currentPrice') else 'N/A',
        'Previous Close': f"₹{info.get('previousClose', 0):.2f}" if info.get('previousClose') else 'N/A',
        'Day Range': f"₹{info.get('dayLow', 0):.2f} - ₹{info.get('dayHigh', 0):.2f}" if all(k in info for k in ['dayLow', 'dayHigh']) else 'N/A',
        '52-Week Range': f"₹{info.get('fiftyTwoWeekLow', 0):.2f} - ₹{info.get('fiftyTwoWeekHigh', 0):.2f}" if all(k in info for k in ['fiftyTwoWeekLow', 'fiftyTwoWeekHigh']) else 'N/A',
        'Volume': f"{info.get('volume', 0):,}" if info.get('volume') else 'N/A',
        'Avg. Volume': f"{info.get('averageVolume', 0):,}" if info.get('averageVolume') else 'N/A',
    }
    
    # Valuation Metrics
    valuation = {
        'P/E (TTM)': f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else 'N/A',
        'Forward P/E': f"{info.get('forwardPE', 0):.2f}" if info.get('forwardPE') else 'N/A',
        'PEG Ratio': f"{info.get('pegRatio', 0):.2f}" if info.get('pegRatio') else 'N/A',
        'Price/Book': f"{info.get('priceToBook', 0):.2f}" if info.get('priceToBook') else 'N/A',
        'Price/Sales': f"{info.get('priceToSalesTrailing12Months', 0):.2f}" if info.get('priceToSalesTrailing12Months') else 'N/A',
        'Enterprise Value (Cr)': f"₹{info.get('enterpriseValue', 0) / 10000000:,.0f}" if info.get('enterpriseValue') else 'N/A',
    }
    
    # Financial Health
    financials = {
        'ROE': f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else 'N/A',
        'ROA': f"{(info.get('netIncomeToCommon', 0) / info.get('totalAssets', 1) * 100):.2f}%" if all(k in info for k in ['netIncomeToCommon', 'totalAssets']) and info['totalAssets'] else 'N/A',
        'Debt/Equity': f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else 'N/A',
        'Current Ratio': f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else 'N/A',
        'Quick Ratio': f"{info.get('quickRatio', 0):.2f}" if info.get('quickRatio') else 'N/A',
        'Interest Coverage': f"{info.get('interestCoverage', 0):.2f}" if info.get('interestCoverage') else 'N/A',
    }
    
    # Growth & Margins
    growth = {
        'Revenue Growth (YoY)': f"{info.get('revenueGrowth', 0) * 100:.2f}%" if info.get('revenueGrowth') else 'N/A',
        'Earnings Growth (YoY)': f"{info.get('earningsGrowth', 0) * 100:.2f}%" if info.get('earningsGrowth') else 'N/A',
        'EBITDA Margin': f"{info.get('ebitdaMargins', 0) * 100:.2f}%" if info.get('ebitdaMargins') else 'N/A',
        'Operating Margin': f"{info.get('operatingMargins', 0) * 100:.2f}%" if info.get('operatingMargins') else 'N/A',
        'Net Margin': f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else 'N/A',
        'Free Cash Flow (Cr)': f"₹{info.get('freeCashflow', 0) / 10000000:,.0f}" if info.get('freeCashflow') else 'N/A',
    }
    
    # Dividends
    dividends = {
        'Dividend Yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else '0.00%',
        'Payout Ratio': f"{info.get('payoutRatio', 0) * 100:.2f}%" if info.get('payoutRatio') else 'N/A',
        'Last Dividend': f"₹{info.get('lastDividendValue', 0):.2f}" if info.get('lastDividendValue') else 'N/A',
        'Ex-Dividend Date': pd.to_datetime(info.get('exDividendDate'), unit='s').strftime('%d-%b-%Y') if info.get('exDividendDate') else 'N/A',
    }
    
    # Ownership
    ownership = {
        'Insider Ownership': f"{info.get('heldPercentInsiders', 0) * 100:.2f}%" if info.get('heldPercentInsiders') else 'N/A',
        'Institutional Ownership': f"{info.get('heldPercentInstitutions', 0) * 100:.2f}%" if info.get('heldPercentInstitutions') else 'N/A',
        'Float Short': f"{info.get('shortPercentOfFloat', 0) * 100:.2f}%" if info.get('shortPercentOfFloat') else 'N/A',
    }
    
    # Analyst Recommendations
    recommendations = {
        'Mean Recommendation': info.get('recommendationMean', 'N/A'),
        'Recommendation': info.get('recommendationKey', 'N/A').title(),
        'Target High': f"₹{info.get('targetHighPrice', 0):.2f}" if info.get('targetHighPrice') else 'N/A',
        'Target Mean': f"₹{info.get('targetMeanPrice', 0):.2f}" if info.get('targetMeanPrice') else 'N/A',
        'Target Low': f"₹{info.get('targetLowPrice', 0):.2f}" if info.get('targetLowPrice') else 'N/A',
        'No. of Analysts': info.get('numberOfAnalystOpinions', 'N/A'),
    }
    
    # Multibagger Potential Indicators
    multibagger = {
        'Earnings Growth (5Y)': '15-25%+ (Target)',
        'Revenue Growth (YoY)': growth['Revenue Growth (YoY)'],
        'Debt/Equity': f"{info.get('debtToEquity', 0):.2f} (Target: <0.5)" if info.get('debtToEquity') else 'N/A',
        'Free Cash Flow': 'Positive' if info.get('freeCashflow', 0) > 0 else 'Negative',
        'ROE': f"{info.get('returnOnEquity', 0) * 100:.2f}% (Target: >15-20%)" if info.get('returnOnEquity') else 'N/A',
        'P/E Ratio': f"{info.get('trailingPE', 0):.2f} (Target: <15-20)" if info.get('trailingPE') else 'N/A',
        'PEG Ratio': f"{info.get('pegRatio', 0):.2f} (Target: <1.0)" if info.get('pegRatio') else 'N/A',
        'Net Margin': growth['Net Margin'],
        'Sector Growth': 'Check Sector Trends',
        'Management Quality': 'Review Leadership & Insider Trades',
    }
    
    return {
        'stock_data': stock_data,
        'valuation': valuation,
        'financials': financials,
        'growth': growth,
        'dividends': dividends,
        'ownership': ownership,
        'recommendations': recommendations,
        'multibagger': multibagger
    }

def display_stock_info(stock_info):
    """Display stock information in Streamlit"""
    st.markdown("---")
    st.markdown("## 📊 Stock Information")
    
    # Stock Data
    st.markdown("### Company Overview")
    stock_data = stock_info['stock_data']
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Company", stock_data['Company'])
        st.metric("Sector", stock_data['Sector'])
        st.metric("Industry", stock_data['Industry'])
        st.metric("Market Cap", stock_data['Market Cap (Cr)'])
    
    with col2:
        st.metric("Current Price", stock_data['Current Price'])
        st.metric("52-Week Range", stock_data['52-Week Range'])
        st.metric("Volume", stock_data['Volume'])
        st.metric("Avg. Volume", stock_data['Avg. Volume'])
    
    # Valuation Metrics
    st.markdown("### 📈 Valuation Metrics")
    val_cols = st.columns(4)
    for i, (k, v) in enumerate(stock_info['valuation'].items()):
        with val_cols[i % 4]:
            st.metric(k, v)
    
    # Financial Health
    st.markdown("### 💰 Financial Health")
    fin_cols = st.columns(3)
    for i, (k, v) in enumerate(stock_info['financials'].items()):
        with fin_cols[i % 3]:
            st.metric(k, v)
    
    # Growth Metrics
    st.markdown("### 📈 Growth & Margins")
    growth_cols = st.columns(3)
    for i, (k, v) in enumerate(stock_info['growth'].items()):
        with growth_cols[i % 3]:
            st.metric(k, v)
    
    # Multibagger Potential Analysis
    st.markdown("### 💎 Multibagger Potential Analysis")
    st.markdown("*Key indicators that may signal potential for significant long-term growth*")
    
    mb_cols = st.columns(2)
    with mb_cols[0]:
        st.metric("Earnings Growth (5Y Target)", "15-25%+", 
                 help="Look for consistent EPS growth of 15-25% or higher over 3-5 years")
        st.metric("Revenue Growth (YoY)", stock_info['growth']['Revenue Growth (YoY)'],
                 help="Target year-over-year revenue increases of at least 15-20%")
        st.metric("Debt-to-Equity", stock_info['multibagger']['Debt/Equity'],
                 help="Target ratio below 0.5, indicating minimal reliance on borrowed funds")
        st.metric("Free Cash Flow", stock_info['multibagger']['Free Cash Flow'],
                 help="Positive and growing free cash flow enables reinvestment and growth")
    
    with mb_cols[1]:
        st.metric("Return on Equity (ROE)", stock_info['multibagger']['ROE'],
                 help="Target ROE above 15-20% for efficient capital use")
        st.metric("P/E Ratio", stock_info['multibagger']['P/E Ratio'],
                 help="Target P/E below industry average (typically under 15-20)")
        st.metric("PEG Ratio", stock_info['multibagger']['PEG Ratio'],
                 help="Target below 1.0 indicates potential undervaluation relative to growth")
        st.metric("Net Margin", stock_info['multibagger']['Net Margin'],
                 help="Consistent or improving margins indicate pricing power and efficiency")
    
    st.markdown("---")
    st.markdown("""
    **Key Takeaways for Multibagger Potential:**
    - **Earnings & Revenue Growth:** Consistent growth is the primary driver of multibagger returns
    - **Strong Balance Sheet:** Low debt and healthy cash flows provide stability in downturns
    - **Efficient Operations:** High ROE and margins indicate competitive advantages
    - **Valuation:** Reasonable valuations provide margin of safety and room for multiple expansion
    - **Sector Trends:** Favorable industry tailwinds can amplify growth
    - **Management:** Strong leadership with skin in the game (high insider ownership)
    """)

def compute_latest_technical_indicators(ticker: str):
    # Fetch historical price data
    tick = yf.Ticker(ticker)
        
    data = tick.history(period="1y", interval="1d")
    data.dropna(inplace=True)

    # Compute Moving Averages
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_100'] = data['Close'].rolling(window=100).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Compute RSI
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()

    # Compute MACD
    macd = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()

    # Select only the latest row
    latest = data.iloc[-1]

    # Return as dictionary
    result = {
        'ticker': ticker,
        'Close': round(latest['Close'], 2),
        'Volume': int(latest['Volume']),
        'EMA_10': round(latest['EMA_10'], 2),
        'EMA_20': round(latest['EMA_20'], 2),
        'SMA_50': round(latest['SMA_50'], 2),
        'SMA_100': round(latest['SMA_100'], 2),
        'SMA_200': round(latest['SMA_200'], 2),
        'RSI': round(latest['RSI'], 2),
        'MACD': round(latest['MACD'], 2),
        'MACD_Signal': round(latest['MACD_Signal'], 2),
        'MACD_Diff': round(latest['MACD_Diff'], 2)
    }

    return result






def BreakoutVolume(niftylist):
    stockList = []
    
    total_items = len(niftylist)
    itr = 0
    progress_bar = st.progress(0)

    # Cache for storing stock data
    stock_cache = {}
    
    # Get current date for week and month calculations
    current_date = pd.Timestamp.now()
    current_week = current_date.strftime('%Y-%U')
    current_month = current_date.to_period('M')

    for symbol in niftylist:
        progress_bar.progress((itr + 1) / total_items)
        itr += 1

        # Check cache first
        if symbol in stock_cache:
            dt = stock_cache[symbol]
        else:
            try:
                stock = yf.Ticker(symbol)
                dt = stock.history(period="6mo", interval="1d")
                if not dt.empty:
                    dt = dt.reset_index()
                    stock_cache[symbol] = dt
                else:
                    continue
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue

        # Early exit if not enough data
        if len(dt) < 5:
            continue

        # Calculate SMAs and EMA in one go
        dt['50_SMA'] = sma_indicator(dt['Close'], window=50)
        dt['30_SMA'] = sma_indicator(dt['Close'], window=30)
        dt['100_SMA'] = sma_indicator(dt['Close'], window=100)
        dt['200_SMA'] = sma_indicator(dt['Close'], window=200)
        dt['20_SMA'] = sma_indicator(dt['Close'], window=20)
        dt['Volume_EMA20'] = dt['Volume'].ewm(span=20, adjust=False).mean()
        # Calculate 1 week ago volume (5 trading days)
        dt['Volume_1w_ago'] = dt['Volume'].shift(5)

        # Sort once
        dt.sort_values(by='Date', ascending=False, inplace=True)

        # Get daily values
        daily_values = dt.iloc[0]
        prev_day_values = dt.iloc[1]
        two_days_ago_values = dt.iloc[2]
        three_days_ago_values = dt.iloc[3]
        four_days_ago_values = dt.iloc[4]

        # Filter by volume and price first (most likely to fail)
        if (daily_values['Volume'] < daily_values['Volume_EMA20'] or 
            daily_values['Close'] < daily_values['50_SMA'] or 
            daily_values['Close'] < daily_values['20_SMA'] or
            daily_values['Close'] < daily_values['30_SMA'] or
            daily_values['Close'] < daily_values['100_SMA'] or
            daily_values['Close'] < daily_values['200_SMA'] or
            daily_values['Volume'] <= daily_values.get('Volume_1w_ago', 0)):
            continue

        # Get week and month data
        # Get week and month data
        # Sort by date in ascending order to get the most recent data
        dt_sorted = dt.sort_values(by='Date', ascending=True)
        
        # Get the most recent data point for week and month
        week_filtered = dt_sorted[dt_sorted['Date'].dt.strftime('%Y-%U') == current_week]
        month_filtered = dt_sorted[dt_sorted['Date'].dt.to_period('M') == current_month]
        
        if week_filtered.empty or month_filtered.empty:
            # If we don't have data for current week/month, try getting the most recent available data
            week_data = dt_sorted.iloc[-1]
            month_data = dt_sorted.iloc[-1]
            #print(f"Warning: Using most recent data for {symbol} instead of current week/month")
        else:
            week_data = week_filtered.iloc[-1]
            month_data = month_filtered.iloc[-1]

        # Calculate price ranges
        daily_range = abs(daily_values['High'] - daily_values['Low'])
        prev_range = abs(prev_day_values['High'] - prev_day_values['Low'])
        two_day_range = abs(two_days_ago_values['High'] - two_days_ago_values['Low'])
        three_day_range = abs(three_days_ago_values['High'] - three_days_ago_values['Low'])
        four_day_range = abs(four_days_ago_values['High'] - four_days_ago_values['Low'])

        # Check price ranges
        if not (daily_range > prev_range and 
               daily_range > two_day_range and 
               daily_range > three_day_range and 
               daily_range > four_day_range):
            continue

        # Check closing conditions
        if not (daily_values['Close'] > daily_values['Open'] and 
               daily_values['Close'] > week_data['Open'] and 
               daily_values['Close'] > month_data['Open']):
            continue

        # Check low condition
        if daily_values['Low'] <= (prev_day_values['Close'] - abs(prev_day_values['Close'] / 222)):
            continue

        # If we've made it this far, add to list
        stockList.append(symbol)

    return stockList

def results(soup):

    yearly_values = []
    quarter_values = []

    # Find the section with id "profit-loss"
    section = soup.find('section', id='profit-loss')

    if section:
        # Extract rows from this section
        rows = section.find_all('tr')

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'Net Profit' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                yearly_values = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row



          # Find the section with id "profit-loss"
    section = soup.find('section', id='quarters')

    if section:
        # Extract rows from this section
        rows = section.find_all('tr')

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'Net Profit' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                quarter_values = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row


    return  quarter_values, yearly_values


def shareholding(soup):

    Promoters = []
    DII = []
    FII = []
    Public = []

          # Find the section with id "profit-loss"
    section = soup.find('section', id='shareholding')

    if section:
        # Extract rows from this section
        rows = section.find_all('tr')

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'Promoters' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                Promoters = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'DIIs' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                DII = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'FIIs' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                FII = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'Public' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                Public = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row

    return Promoters, DII, FII, Public
  
def extract_key_insights(soup):
    company_name = soup.find('h1', class_='margin-0 show-from-tablet-landscape').text.strip()
    current_price = soup.find('div', class_='font-size-18 strong line-height-14').find('span').text.strip()
    market_cap = soup.find('li', {'data-source': 'default'}).find('span', class_='number').text.strip()
    about_section = soup.find('div', class_='company-profile').find('div', class_='sub show-more-box about').text.strip()
    pe_value = soup.find('span', class_='name', string=lambda t: t and "Stock P/E" in t).find_next('span', class_='number').string
    roe = soup.find('span', class_='name', string=lambda t: t and "ROE" in t).find_next('span', class_='number').string
    roce = soup.find('span', class_='name', string=lambda t: t and "ROCE" in t).find_next('span', class_='number').string
    
    quarter_values, yearly_values = results(soup)
    Promoters, DII, FII, Public = shareholding(soup)

    fundainfo = {
        "Company Name": company_name,
        "Current Price": current_price,
        "Market Cap": market_cap,
        "About": about_section,
        "PE" : pe_value,
        "ROE" : roe,
        "ROCE" : roce}

    shareholdnres = {"Quarter" : quarter_values,
        "Yearly" : yearly_values,
        "Promoters" : Promoters,
        "DII" : DII,
        "FII" : FII,
        "Public" : Public
    }

    return fundainfo, shareholdnres 


def scrapper(stock_ticker):
    
    stock_ticker = stock_ticker.replace('.NS', '')

    url = f"https://www.screener.in/company/{stock_ticker}/"

    response = requests.get(url, timeout=10)  # 10 second timeout

    if response.status_code == 200:
        print("Successfully fetched the webpage")
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')
    
    
    fundainfo, shareholdnres = extract_key_insights(soup)
    
    return fundainfo, shareholdnres


def get_financial_metrics(ticker, stock_info=None):
    """
    Calculate and return financial metrics for a given stock ticker
    
    Args:
        ticker: Stock ticker symbol (str, e.g., 'AAPL' or 'RELIANCE.NS') or yfinance.Ticker object
        stock_info: Optional pre-fetched stock info dictionary. If None, will be fetched.
        
    Returns:
        dict: Dictionary containing financial metrics with their values and metadata
    """
    # Handle case where ticker is a yfinance.Ticker object
    if hasattr(ticker, 'ticker'):  # If it's a Ticker object
        tick = ticker
        ticker_symbol = tick.ticker
    else:  # If it's a string
        ticker_symbol = str(ticker).upper()
        tick = yf.Ticker(ticker_symbol)
    
    # If stock_info is not provided, fetch it
    if stock_info is None:
        try:
            stock_info = extract_stock_info(tick.info)
        except Exception as e:
            print(f"Error fetching stock data for {ticker_symbol}: {e}")
            return {}
    # Calculate EPS growth if available
    eps_growth = 'N/A'
    eps_cagr = 'N/A'
    
    try:
        # Get ticker info if not already available
        ticker_info = tick.info if 'tick' in locals() else yf.Ticker(ticker_symbol).info
        
        # Calculate EPS growth
        if 'trailingEps' in ticker_info and ticker_info['trailingEps'] is not None:
            # EPS Growth (Quarterly)
            eps_growth_val = ticker_info.get('earningsQuarterlyGrowth', 0)
            if eps_growth_val is not None:
                eps_growth = f"{eps_growth_val * 100:.1f}%"
            
            # EPS CAGR (5-year)
            try:
                hist = tick.history(period="5y")
                if not hist.empty and 'Close' in hist.columns and ticker_info['trailingEps'] is not None:
                    current_eps = ticker_info['trailingEps']
                    earnings_growth = ticker_info.get('earningsGrowth', 0) or 0
                    oldest_eps = current_eps / (1 + earnings_growth) ** 4
                    if oldest_eps > 0:
                        eps_cagr_val = ((current_eps / oldest_eps) ** (1/5) - 1) * 100
                        eps_cagr = f"{eps_cagr_val:.1f}%"
            except Exception as e:
                print(f"Error calculating EPS CAGR for {ticker_symbol}: {e}")
    except Exception as e:
        print(f"Error calculating EPS CAGR: {e}")
    
    return {
        'Revenue Growth (YoY)': {
            'value': stock_info['growth']['Revenue Growth (YoY)'],
            'target': ('min', 20),
            'better': 'higher',
            'importance': 'Top-line growth is the engine of future earnings',
            'ideal': '25%+ sustained'
        },
        'Earnings Growth (YoY)': {
            'value': stock_info['growth']['Earnings Growth (YoY)'],
            'target': ('min', 50),
            'better': 'higher',
            'importance': 'Shows operating leverage and margin expansion',
            'ideal': '30%+ sustained'
        },
        'EBITDA Margin': {
            'value': stock_info['growth']['EBITDA Margin'],
            'target': ('min', 15),
            'better': 'higher',
            'importance': 'High & scalable margins = profit compounding machine',
            'ideal': '15%+ and rising'
        },
        'Net Margin': {
            'value': stock_info['growth']['Net Margin'],
            'target': ('min', 12),
            'better': 'higher',
            'importance': 'Converts revenue into shareholder profit efficiently',
            'ideal': '12%+ and rising'
        },
        'P/E (TTM)': {
            'value': stock_info['valuation'].get('P/E (TTM)', 'N/A'),
            'target': ('max', 25),
            'better': 'lower',
            'importance': 'Valuation multiple - lower is better',
            'ideal': '< 25 or PEG < 1.0'
        },
        'PEG Ratio': {
            'value': stock_info['valuation'].get('PEG Ratio', 'N/A'),
            'target': ('max', 1.0),
            'better': 'lower',
            'importance': 'Growth at reasonable price indicator',
            'ideal': '< 1.0'
        },
        'Debt/Equity': {
            'value': stock_info['financials'].get('Debt/Equity', 'N/A'),
            'target': ('max', 0.5),
            'better': 'lower',
            'importance': 'Financial leverage and risk indicator',
            'ideal': '< 0.5 (ideally < 0.3)'
        },
        'ROE': {
            'value': stock_info['financials'].get('ROE', 'N/A'),
            'target': ('min', 20),
            'better': 'higher',
            'importance': 'Return on shareholder equity',
            'ideal': '≥ 20%'
        },
        'YoY EPS Growth': {
            'value': stock_info['growth'].get('Earnings Growth (YoY)', 'N/A'),
            'target': ('min', 25),
            'better': 'higher',
            'importance': 'Year-over-year earnings growth rate',
            'ideal': '≥ 40–50% (first 2–3 yrs) → then ≥ 25–30% CAGR'
        },
        'QoQ EPS Growth': {
            'value': eps_growth,
            'target': ('min', 30),
            'better': 'higher',
            'importance': 'Quarter-over-quarter earnings growth',
            'ideal': '≥ 30–40% in early phase'
        },
        '5-Yr EPS CAGR': {
            'value': eps_cagr,
            'target': ('min', 25),
            'better': 'higher',
            'importance': '5-year compound annual growth rate of EPS',
            'ideal': '≥ 25%'
        }
    }

def get_company_styles():
    """Return the CSS styles for the company details page."""
    return """
    <style>
    /* Base styles */
    .stMarkdown h2 {
        font-size: 20px;
        margin: 1rem 0 0.5rem;
    }
    .stMarkdown h3 {
        font-size: 16px;
        margin: 0.75rem 0 0.25rem;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 14px;
        line-height: 1.4;
    }
    
    /* Compact metrics */
    .stMetric {
        padding: 0.5rem 0.75rem !important;
        margin: 0.25rem 0 !important;
        border-radius: 0.5rem !important;
        background: #f8f9fa !important;
    }
    .stMetric > div {
        gap: 0.25rem !important;
    }
    .stMetric > div > div:first-child {
        font-size: 0.8rem !important;
        color: #666 !important;
        font-weight: 500 !important;
    }
    .stMetric > div > div:last-child {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Compact tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem !important;
    }
    
    /* Compact table styles */
    .dataframe {
        font-size: 0.85rem !important;
    }
    .dataframe th, .dataframe td {
        padding: 0.4rem 0.75rem !important;
    }
    
    /* Valuation and Financial Tables */
    .valuation-table, .financial-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 14px;
    }
    .valuation-table th, .financial-table th {
        background-color: #2c3e50;
        color: white;
        text-align: left;
        padding: 10px 12px;
        font-weight: 500;
    }
    .valuation-table td, .financial-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #e0e0e0;
        vertical-align: middle;
    }
    .valuation-table tr:nth-child(even), .financial-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    .valuation-table tr:hover, .financial-table tr:hover {
        background-color: #f1f3f5;
    }
    .value-col {
        text-align: right;
        font-weight: 500;
    }
    .spacer-col {
        width: 30px;
        background: white;
    }
    .metric-col {
        color: #2c3e50;
        font-weight: 500;
    }
    
    /* Multibagger Table */
    #multibagger-table {
        width: 100%;
        min-width: 800px;
        border-collapse: collapse;
        margin: 0;
        font-size: 14px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    #multibagger-table th {
        background-color: #2c3e50;
        color: white;
        font-weight: 500;
        padding: 12px 15px;
        text-align: left;
    }
    #multibagger-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #e0e0e0;
        vertical-align: middle;
    }
    #multibagger-table tr:last-child td {
        border-bottom: none;
    }
    #multibagger-table tr:hover {
        background-color: #f8f9fa;
    }
    #multibagger-table th:first-child, 
    #multibagger-table td:first-child {
        padding-left: 20px;
    }
    #multibagger-table th:last-child, 
    #multibagger-table td:last-child {
        padding-right: 20px;
    }
    </style>
    """

def get_eps_data(tick):
    """Extract and format EPS-related data from the ticker."""
    eps_ttm = tick.info.get('trailingEps', 'N/A')
    eps_forward = tick.info.get('forwardEps', 'N/A')
    eps_growth = tick.info.get('earningsQuarterlyGrowth', 'N/A')
    
    if eps_ttm != 'N/A' and eps_ttm is not None:
        eps_ttm = f"₹{float(eps_ttm):.2f}"
    if eps_forward != 'N/A' and eps_forward is not None:
        eps_forward = f"₹{float(eps_forward):.2f}"
    if eps_growth != 'N/A' and isinstance(eps_growth, (int, float)):
        eps_growth = f"{eps_growth * 100:.1f}%"
    
    return eps_ttm, eps_forward, eps_growth

def render_company_header(company_data, ticker, stock_info):
    """Render the company header with name, ticker, and industry."""
    st.markdown(f"""
    <div style='margin-bottom: 0.5rem;'>
        <h1 style='font-size: 22px; font-weight: 600; margin: 0 0 0.25rem 0;'>{company_data.get('Company Name', '')}</h1>
        <div style='font-size: 14px; color: #666; margin-bottom: 0.5rem;'>{ticker} • {stock_info['stock_data'].get('Industry', '')}</div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics_grid(stock_info, eps_ttm, eps_growth):
    """Render the metrics grid with key company metrics."""
    metrics_grid = st.columns(4)
    metrics_data = [
        ("Current Price", stock_info['stock_data'].get('Current Price', 'N/A')),
        ("52-Week Range", stock_info['stock_data'].get('52-Week Range', 'N/A')),
        ("Market Cap", stock_info['stock_data'].get('Market Cap (Cr)', 'N/A')),
        ("Volume / Avg", f"{stock_info['stock_data'].get('Volume', 'N/A')} / {stock_info['stock_data'].get('Avg. Volume', 'N/A')}"),
        ("P/E (TTM)", stock_info['valuation'].get('P/E (TTM)', 'N/A')),
        ("Sector", stock_info['stock_data'].get('Sector', 'N/A')),
        ("EPS (TTM)", eps_ttm if eps_ttm != 'N/A' else 'N/A'),
        ("EPS Growth (QoQ)", eps_growth if eps_growth != 'N/A' else 'N/A')
    ]
    
    for i, (label, value) in enumerate(metrics_data):
        with metrics_grid[i % 4]:
            st.metric(label, value)

def render_overview_tab(tick, stock_info):
    """Render the content of the Overview tab."""
    st.subheader("Company Information")
    st.markdown(f"**Industry:** {stock_info['stock_data']['Industry']}")
    
    if 'longBusinessSummary' in tick.info:
        st.markdown("### About")
        st.markdown(f"*{tick.info['longBusinessSummary']}*")
    
    if 'companyOfficers' in tick.info and tick.info['companyOfficers']:
        st.markdown("### Key Executives")
        for officer in tick.info['companyOfficers'][:5]:
            st.markdown(f"- **{officer.get('name', 'N/A')}**: {officer.get('title', 'N/A')}")

def render_valuation_tab(stock_info, eps_ttm, eps_forward, eps_growth):
    """Render the content of the Valuation tab."""
    st.subheader("Valuation Metrics")
    
    with st.container():
        valuation_metrics = [
            ("Market Cap", stock_info['stock_data'].get('Market Cap (Cr)', 'N/A')),
            ("P/E (TTM)", stock_info['valuation'].get('P/E (TTM)', 'N/A')),
            ("Forward P/E", stock_info['valuation'].get('Forward P/E', 'N/A')),
            ("PEG Ratio", stock_info['valuation'].get('PEG Ratio', 'N/A')),
            ("P/S (TTM)", stock_info['valuation'].get('P/S (TTM)', 'N/A')),
            ("P/B", stock_info['valuation'].get('P/B', 'N/A')),
            ("P/FCF", stock_info['valuation'].get('P/FCF', 'N/A')),
            ("EV/EBITDA", stock_info['valuation'].get('EV/EBITDA', 'N/A')),
            ("EPS (TTM)", eps_ttm if eps_ttm != 'N/A' else 'N/A'),
            ("Forward EPS", eps_forward if eps_forward != 'N/A' else 'N/A'),
            ("EPS Growth (QoQ)", eps_growth if eps_growth != 'N/A' else 'N/A'),
            ("Dividend Yield", stock_info['financials'].get('Dividend Yield', 'N/A'))
        ]
        
        mid_point = (len(valuation_metrics) + 1) // 2
        left_metrics = valuation_metrics[:mid_point]
        right_metrics = valuation_metrics[mid_point:]
        
        df = pd.DataFrame({
            'Metric': [m[0] for m in left_metrics],
            'Value': [m[1] for m in left_metrics],
            '  ': [''] * len(left_metrics),
            'Metric ': [m[0] for m in right_metrics] + [''] * (len(left_metrics) - len(right_metrics)),
            'Value ': [m[1] for m in right_metrics] + [''] * (len(left_metrics) - len(right_metrics))
        })
        
        html = df.to_html(
            index=False,
            classes='valuation-table',
            escape=False,
            header=True,
            justify='left'
        )
        
        html = html.replace('<td>', '<td class="value-col">')
        html = html.replace('<td> </td>', '<td class="spacer-col"></td>')
        html = html.replace('<th>Metric</th>', '<th class="metric-col">Metric</th>')
        html = html.replace('<th>Metric </th>', '<th class="metric-col">Metric</th>')
        
        st.markdown(html, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='margin-top: 10px; font-size: 13px; color: #666;'>
            <strong>Note:</strong> All valuation metrics are based on the most recent available data.
            P/E and other ratios are calculated using TTM (Trailing Twelve Months) figures unless specified.
        </div>
        """, unsafe_allow_html=True)

def render_financials_tab(stock_info):
    """Render the content of the Financials tab."""
    st.subheader("Financial Health")
    
    with st.container():
        financial_metrics = [
            ("Current Ratio", stock_info['financials'].get('Current Ratio', 'N/A')),
            ("Quick Ratio", stock_info['financials'].get('Quick Ratio', 'N/A')),
            ("Debt/Equity", stock_info['financials'].get('Debt/Equity', 'N/A')),
            ("Interest Coverage", stock_info['financials'].get('Interest Coverage', 'N/A')),
            ("ROE", stock_info['financials'].get('ROE', 'N/A')),
            ("ROA", stock_info['financials'].get('ROA', 'N/A')),
            ("ROIC", stock_info['financials'].get('ROIC', 'N/A')),
            ("Operating Margin", stock_info['growth'].get('Operating Margin', 'N/A')),
            ("Net Margin", stock_info['growth'].get('Net Margin', 'N/A')),
            ("EBITDA Margin", stock_info['growth'].get('EBITDA Margin', 'N/A')),
            ("Revenue Growth (YoY)", stock_info['growth'].get('Revenue Growth (YoY)', 'N/A')),
            ("Earnings Growth (YoY)", stock_info['growth'].get('Earnings Growth (YoY)', 'N/A')),
            ("FCF Growth (YoY)", stock_info['growth'].get('FCF Growth (YoY)', 'N/A')),
            ("Dividend Payout Ratio", stock_info['financials'].get('Payout Ratio', 'N/A')),
            ("Dividend Yield", stock_info['financials'].get('Dividend Yield', 'N/A'))
        ]
        
        mid_point = (len(financial_metrics) + 1) // 2
        left_metrics = financial_metrics[:mid_point]
        right_metrics = financial_metrics[mid_point:]
        
        df = pd.DataFrame({
            'Metric': [m[0] for m in left_metrics],
            'Value': [m[1] for m in left_metrics],
            '  ': [''] * len(left_metrics),
            'Metric ': [m[0] for m in right_metrics] + [''] * (len(left_metrics) - len(right_metrics)),
            'Value ': [m[1] for m in right_metrics] + [''] * (len(left_metrics) - len(right_metrics))
        })
        
        html = df.to_html(
            index=False,
            classes='financial-table',
            escape=False,
            header=True,
            justify='left'
        )
        
        html = html.replace('<td>', '<td class="value-col">')
        html = html.replace('<td> </td>', '<td class="spacer-col"></td>')
        
        st.markdown(html, unsafe_allow_html=True)

def get_metric_analysis(metric_name, current_value, target_type, target_value, is_higher_better=True):
    """Analyze a single metric and return its status and styling."""
    if current_value == 'N/A' or current_value is None:
        return 'gray', '❓ Unknown', "Data not available"
    
    try:
        clean_val = float(str(current_value).replace('₹', '').replace('%', '').split(' ')[0].replace(',', ''))
        
        if 'Debt/Equity' in metric_name:
            is_met = clean_val <= target_value
            target_text = f"<= {target_value}"
            color = '#2ecc71' if is_met else '#e74c3c'
            verdict = '✅ Good' if is_met else '⚠️ High'
        elif 'P/E' in metric_name or 'PEG' in metric_name:
            is_met = clean_val <= target_value
            target_text = f"<= {target_value}"
            color = '#2ecc71' if is_met else '#e74c3c'
            verdict = '✅ Good' if is_met else '⚠️ High'
        else:
            if target_type == 'min':
                is_met = clean_val >= target_value if is_higher_better else clean_val <= target_value
                target_text = f">= {target_value}%" if is_higher_better else f"<= {target_value}"
            elif target_type == 'max':
                is_met = clean_val <= target_value if is_higher_better else clean_val >= target_value
                target_text = f"<= {target_value}" if is_higher_better else f">= {target_value}%"
            
            if is_met:
                color = '#2ecc71'
                verdict = '✅ Strong' if is_higher_better else '✅ Good'
            else:
                color = '#e74c3c'
                verdict = '⚠️ Needs Improvement' if is_higher_better else '⚠️ High'
        
        return color, verdict, target_text
    
    except (ValueError, IndexError) as e:
        print(f"Error processing {metric_name}: {e}")
        return 'gray', '❓ Unknown', "Invalid data"

def calculate_eps_cagr(tick):
    """Calculate EPS CAGR for the last 5 years."""
    eps_cagr = 'N/A'
    try:
        hist = tick.history(period="5y")
        if not hist.empty and 'Close' in hist.columns:
            if 'trailingEps' in tick.info and tick.info['trailingEps'] is not None:
                current_eps = tick.info['trailingEps']
                oldest_eps = current_eps / (1 + (tick.info.get('earningsGrowth', 0) or 0)) ** 4
                if oldest_eps > 0:
                    eps_cagr_calc = ((current_eps / oldest_eps) ** (1/5) - 1) * 100
                    eps_cagr = f"{eps_cagr_calc:.1f}%"
    except Exception as e:
        print(f"Error calculating EPS CAGR: {e}")
    
    return eps_cagr

def render_multibagger_tab(tick, stock_info, metrics):
    """Render the content of the Multibagger tab."""
    st.subheader("Multibagger Potential Analysis")
    st.markdown("*Comprehensive evaluation of key financial metrics*")
    
    # Calculate EPS CAGR
    eps_cagr = calculate_eps_cagr(tick)
    
    # Display metrics in a clean, well-formatted table
    st.markdown("### Key Financial Metrics")
    
    # Create a DataFrame for the table with proper formatting
    table_data = []
    for metric, data in metrics.items():
        is_higher_better = data['better'] == 'higher'
        color, verdict, _ = get_metric_analysis(
            metric, data['value'], data['target'][0], data['target'][1], is_higher_better
        )
        
        clean_value = str(data['value']).replace('₹', '').replace('%', '').strip()
        try:
            float_val = float(clean_value.split(' ')[0])
            formatted_value = f"{float_val:,.2f}"
            if '%' in str(data['value']):
                formatted_value += '%'
            if '₹' in str(data['value']):
                formatted_value = '₹' + formatted_value
            colored_value = f"<span style='color: {color}; font-weight: 600;'>{formatted_value}</span>"
        except (ValueError, IndexError):
            colored_value = f"<span style='color: {color}; font-weight: 600;'>{data['value']}</span>"
        
        colored_verdict = f"<span style='color: {color};'>{verdict}</span>"
        
        table_data.append({
            'Parameter': metric,
            'Your Value': colored_value,
            'Target': data.get('ideal', ''),
            'Verdict': colored_verdict,
            'Why It Matters': data['importance']
        })
    
    df_table = pd.DataFrame(table_data)
    
    st.markdown(
        "<div style='margin: 15px 0; overflow-x: auto;'>" +
        df_table.to_html(escape=False, index=False, 
                        classes='dataframe',
                        table_id='multibagger-table') +
        "</div>" +
        """
        <div style='margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 14px;'>
            <strong>Note:</strong> Projections are based on historical data and standard growth assumptions. 
            Actual performance may vary. Always conduct thorough research before making investment decisions.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Multibagger Potential Assessment
    st.markdown("### 🚀 Multibagger Potential Assessment")
    
    # Count positive and negative indicators
    positive_indicators = sum(1 for m in metrics.values() 
                            if m['value'] != 'N/A' 
                            and ((m['better'] == 'higher' and float(str(m['value']).replace('%', '')) >= m['target'][1]) or
                                (m['better'] == 'lower' and float(str(m['value']).replace('%', '')) <= m['target'][1])))
    
    total_indicators = sum(1 for m in metrics.values() if m['value'] != 'N/A')
    
    if total_indicators > 0:
        score = (positive_indicators / total_indicators) * 100
        
        if score >= 75:
            conclusion = "✅ Strong Multibagger Potential"
            reasoning = "The company shows excellent fundamentals across most key metrics. If growth continues and valuations remain reasonable, it has significant multibagger potential."
        elif score >= 50:
            conclusion = "🟡 Moderate Potential"
            reasoning = "The company shows promise but has some areas that need improvement. Monitor the key metrics closely for sustained growth."
        else:
            conclusion = "🔴 High Risk"
            reasoning = "The company has several red flags that need to be addressed. Exercise caution and conduct further due diligence."
        
        st.markdown(f"**Overall Score:** {positive_indicators}/{total_indicators} metrics met ({score:.0f}%)")
        st.markdown(f"**Conclusion:** {conclusion}")
        st.markdown(f"**Analysis:** {reasoning}")
    
    # Key Conditions for Multibagger Potential
    st.markdown("### 📈 When Could This Still Become a Multibagger?")
    
    # Create a list of conditions and their descriptions with icons
    conditions = [
        ("🏦 Debt Reduction", "D/E must fall below 1.0 in 2–3 years via FCF or equity raise"),
        ("📈 Sustained Growth", "25–30% Earnings CAGR for 3–5 years to justify high P/E"),
        ("📊 PEG Ratio", "Must drop below 1.0 (growth at reasonable price)"),
        ("📉 ROE Improvement", "Target >25% through margin expansion and lower debt"),
        ("💰 Positive FCF", "Cash flow must turn positive to fund growth without dilution")
    ]
    
    # Create a container with better spacing
    with st.container():
        # Create a grid layout for conditions
        for i in range(0, len(conditions), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(conditions):
                    with cols[j]:
                        condition, description = conditions[i + j]
                        st.markdown(
                            f"""
                            <div style='
                                background: white;
                                border-radius: 10px;
                                padding: 15px;
                                margin-bottom: 15px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                border-left: 4px solid #2c3e50;
                            '>
                                <div style='font-weight: 600; margin-bottom: 8px; font-size: 15px;'>{condition}</div>
                                <div style='color: #555; font-size: 14px; line-height: 1.5;'>{description}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    
    # Add note with improved styling
    st.markdown(
        """
        <div style='
            margin: 20px 0;
            padding: 15px;
            background-color: #f0f7ff;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            font-size: 14px;
            line-height: 1.6;
        '>
            <strong>📊 Monitoring Tip:</strong> Track these conditions quarterly to assess if the company is on track to become a multibagger. 
            Focus on management's execution against these key metrics in earnings calls and reports.
        </div>
        """,
        unsafe_allow_html=True
    )

def companyDetails(company_data, ticker):
    """Display comprehensive company information and analysis."""
    # Apply custom styles
    st.markdown(get_company_styles(), unsafe_allow_html=True)
    
    try:
        # Create Ticker object and get data
        tick = yf.Ticker(ticker)
        stock_info = extract_stock_info(tick.info)
        metrics = get_financial_metrics(tick, stock_info)
        eps_ttm, eps_forward, eps_growth = get_eps_data(tick)
        
        # Render company header and metrics
        render_company_header(company_data, ticker, stock_info)
        render_metrics_grid(stock_info, eps_ttm, eps_growth)
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", "📈 Valuation", 
            "💰 Financials", "💎 Multibagger"
        ])
        
        with tab1:
            render_overview_tab(tick, stock_info)
        with tab2:
            render_valuation_tab(stock_info, eps_ttm, eps_forward, eps_growth)
        with tab3:
            render_financials_tab(stock_info)
        with tab4:
            render_multibagger_tab(tick, stock_info, metrics)
            
        # Display technical indicators in an expander
        with st.expander("📈 View Technical Indicators"):
            compute_latest_technical_indicators(ticker)
            
    except Exception as e:
        # Error handling and fallback UI
        st.error(f"Error fetching stock data: {str(e)}")
        st.title(company_data.get('Company Name', ''))
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", company_data.get('Current Price', 'N/A'))
            st.metric("P/E Ratio", company_data.get('PE', 'N/A'))
        with col2:
            st.metric("Market Cap (₹ Cr)", company_data.get('Market Cap', 'N/A'))
            st.metric("ROE (%)", company_data.get('ROE', 'N/A'))
        
        if 'About' in company_data:
            st.subheader("About")
            st.markdown(f"*{company_data['About']}*")


def CompanyNews(name):
    # Fetch news articles
    news = google_news.get_news(name)

    # Display news titles with URLs in Streamlit
    st.title(f"Latest News on {name}")

    if news:
        for article in news:
            
            title = article.get('title', 'No title available')
            url = article.get('url', '#')
            
            # Display the title as a clickable link
            st.markdown(f"[{title}]({url})")
    else:
        st.write("No news found for this topic.")

    return news
    


def plotChart(symbol):
    stock = yf.Ticker(symbol)

    df = stock.history(period="1y", interval="1d")
    df = df.reset_index()

    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'])])
    
    st.plotly_chart(fig, width='stretch',config={'displayModeBar': False})
    


# Helper function to convert percentage strings to floats
def convert_to_float(data):
    if not data or not isinstance(data, list):
        return []
    
    if all(isinstance(item, str) for item in data):
        # Remove commas and convert percentages
        if all('%' in item for item in data):
            return [float(value.replace(',', '').strip('%')) for value in data]
        # Remove commas and convert numeric strings
        else:
            return [float(value.replace(',', '')) for value in data]
    return []

def check_status(values):
        if not values or len(values) < 2:
            st.subheader(f"⚠️ :orange[Data not available]")
            return
        
        last_val = values[-1]
        second_last_val = values[-2]

        if last_val < 0:
            st.subheader(f"📉 :red[LOSS]")
        elif last_val == second_last_val:
            st.subheader(f"📈 :rainbow[Unchnaged ⚪]")
        elif last_val > second_last_val:
            st.subheader(f"📈 :green[Increased ✅]")
        else:
            st.subheader(f"📉 :red[Decreased ❌]")

def check_status_public(values):
        last_val = values[-1]
        second_last_val = values[-2]

        if last_val > second_last_val:
            st.subheader(f"📈 :red[Increased ❌]")
        else:
            st.subheader(f"📉 :green[Decreased ✅]")


def analyze_financial_data(data):
    # Function to check and return the status with appropriate coloring
    data = {key: convert_to_float(value) for key, value in data.items()}
    
    col1, col2 = st.columns(2)
    
    
    with col1:
        # Check Quarterly and Yearly data
        st.subheader("1. Quarterly Profit Status:")
        check_status(data.get('Quarter', []))

        st.subheader("3. FII Shareholding Status:")
        check_status(data.get('FII', []))

        # Check Shareholding data (Promoters, DII, FII, Public)
        st.subheader("5. Promoters Shareholding Status:")
        check_status(data.get('Promoters', []))
        
    with col2:
        st.subheader("2. Yearly Profit Status:")
        check_status(data.get('Yearly', []))

        st.subheader("4. DII Shareholding Status:")
        check_status(data.get('DII', []))

        st.subheader("6. Public Shareholding Status:")
        check_status_public(data['Public'])

def plotShareholding(shareholdnres):

    # Convert percentages in each list where necessary
    converted_data = {key: convert_to_float(value) for key, value in shareholdnres.items()}

    # Filter out empty lists
    filtered_data = {key: value for key, value in converted_data.items() if value}

    # Determine the number of subplots
    num_plots = len(filtered_data)
    rows = (num_plots + 1) // 2

    # Create subplots based on the number of non-empty data lists
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=list(filtered_data.keys())
    )

    # Plot each non-empty dataset
    for i, (key, values) in enumerate(filtered_data.items(), start=1):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        fig.add_trace(
            go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode='lines+markers',
                    name=key, line=dict(width=2)),
            row=row, col=col
        )

        fig.update_xaxes(title_text=key, row=row, col=col)
        fig.update_yaxes(title_text="Net Profit" if key in ["Quarter", "Yearly"] else "Holding (%)", row=row, col=col)

    # Update layout
    fig.update_layout(
        height=rows * 400,
        width=1000,
        title_text="Financial Data Analysis",
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, width='stretch',config={'displayModeBar': False})



def reportGenerator(option):
    """
    Main function to generate report
    """
    fundainfo, shareholdnres = scrapper(option)
    technical_indicators = compute_latest_technical_indicators(option)
    companyDetails(fundainfo,option) 
    multibagger_metrics = get_financial_metrics(option)
    chart(ticker=option)
    plotChart(option)
    analyze_financial_data(shareholdnres)
    plotShareholding(shareholdnres)
    news = CompanyNews(fundainfo['Company Name'])

    top_news = news[:8] if len(news) >= 8 else news




    if st.button("AI Research Report"):
        with st.spinner("Generating Report..."):
            st.markdown(heading, unsafe_allow_html=True)
            report = stock_node(fundainfo,shareholdnres,technical_indicators,multibagger_metrics,top_news)

        # Extract <think> section
        think_match = re.search(r"<think>(.*?)</think>", report, re.DOTALL)
        thinking_part = think_match.group(1).strip() if think_match else "No reasoning available."
        report_without_think = re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL).strip()

        with st.chat_message("StockAgent"):

                with st.expander("🧠 Agent Reasoning (Click to Expand)"):
                    st.markdown(thinking_part)

                st.markdown(report_without_think)
    


def StockScan():
    """
    Main function to scan and analyze stocks
    """
    
    # Initialize selected option from session state
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None
    if 'stockList' not in st.session_state:
        st.session_state.stockList = []
    if 'nifty500_stockList' not in st.session_state:
        st.session_state.nifty500_stockList = []
    if 'microcap250_stockList' not in st.session_state:
        st.session_state.microcap250_stockList = []
    if 'analysis_stockList' not in st.session_state:
        st.session_state.analysis_stockList = []

    # Two side-by-side buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Volume Breakout NIFTY500", key='nifty500_btn'):
            st.session_state.selected_option = "Volume Breakout NIFTY500"

    with col2:
        if st.button("📈 Volume Breakout MICROCAP250", key='microcap250_btn'):
            st.session_state.selected_option = "Volume Breakout MICROCAP250"

    with col3:
        if st.button("📊 Stocks Analysis", key='analysis_btn'):
            st.session_state.selected_option = "Stocks Analysis"

    # Get the current selection from session state
    selected_option = st.session_state.selected_option

    # Show results based on selected option
    if selected_option == "Volume Breakout NIFTY500":
        st.title("Running Scan on NIFTY500")
        if st.button("Run Scan", key='nifty500_run_btn'):
            stockList = BreakoutVolume(df500)
            st.session_state.nifty500_stockList = stockList

        if st.session_state.nifty500_stockList:
            st.success(f'Scan Complete : {len(st.session_state.nifty500_stockList)} Stocks Found', icon="✅")
            st.subheader("Stocks")
            cols = st.columns(2)
            for i, stock in enumerate(st.session_state.nifty500_stockList):
                cols[i % 2].write(stock)
            
            option = st.selectbox(
                "List of Stocks",
                st.session_state.nifty500_stockList,
                index=None,
                placeholder="Select the Stock",
            )

            if option:
                reportGenerator(option)

    elif selected_option == "Volume Breakout MICROCAP250":
        st.title("Running Scan on MICROCAP250")
        if st.button("Run Scan", key='microcap250_run_btn'):
            stockList = BreakoutVolume(microcap250)
            st.session_state.microcap250_stockList = stockList

        if st.session_state.microcap250_stockList:
            st.success(f'Scan Complete : {len(st.session_state.microcap250_stockList)} Stocks Found', icon="✅")
            st.subheader("Stocks")
            cols = st.columns(2)
            for i, stock in enumerate(st.session_state.microcap250_stockList):
                cols[i % 2].write(stock)
            
            option = st.selectbox(
                "List of Stocks",
                st.session_state.microcap250_stockList,
                index=None,
                placeholder="Select the Stock",
            )

            if option:
                reportGenerator(option)
                
    elif selected_option == "Stocks Analysis":
        option = st.selectbox(
            "List of Stocks",
            complist,
            index=None,
            placeholder="Select the Stock",
        )

        st.write("You selected:", option)

        option = get_yf_symbol(option)
        
        if option:
            reportGenerator(option)
    
   