import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import sma_indicator
from niftystocks import ns
import streamlit as st
import requests
from bs4 import BeautifulSoup
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import mplfinance as mpf
from gnews import GNews

# Initialize the GNews object
google_news = GNews(language='en', period='30d',max_results=20)

from StockScreener.mlpchart.mlpchart import chart

import matplotlib.pyplot as plt


def BreakoutVolume():
  stockList = []
  df500 = ns.get_nifty500_with_ns()
  
  total_items = len(df500)
  
  itr = 0
  
  progress_bar = st.progress(0)

  for symbol in df500:
      
    progress_bar.progress((itr + 1) / total_items)
    itr += 1

    stock = yf.Ticker(symbol)
    try:
      dt = stock.history(period="6mo", interval="1d")
      dt = dt.reset_index()


      # rsi_14 = RSIIndicator(close=dt['Close'], window=14)
      # # This returns a Pandas series.
      # dt['RSI'] = rsi_14.rsi()

      # dt['200_SMA'] = sma_indicator(dt['Close'],window=200)
      dt['50_SMA'] = sma_indicator(dt['Close'],window=50)
      dt['20_SMA'] = sma_indicator(dt['Close'],window=20)
      # dt['10_SMA'] = sma_indicator(dt['Close'],window=10)
      # dt['5_SMA'] = sma_indicator(dt['Close'],window=5)

      dt['Volume_EMA20'] = dt['Volume'].ewm(span=20, adjust=False).mean()

      dt.sort_values(by='Date', ascending=False, inplace=True)

      # Extract relevant values from the DataFrame
      daily_values = dt.iloc[0]
      prev_day_values = dt.iloc[1] if len(dt) > 1 else None
      two_days_ago_values = dt.iloc[2] if len(dt) > 2 else None
      three_days_ago_values = dt.iloc[3] if len(dt) > 3 else None
      four_days_ago_values = dt.iloc[4] if len(dt) > 4 else None

      # Extract the week number and year of the dates in the DataFrame
      dt['YearWeek'] = dt['Date'].dt.strftime('%Y-%U')
      # Get the earliest date for the latest week-year in the DataFrame
      first_date_of_week = dt[dt['YearWeek'] == dt['YearWeek'].iloc[0]]['Date'].min()

      dt['YearMonth'] = dt['Date'].dt.to_period('M')
      # Get the earliest date for the latest month-year in the DataFrame
      first_date_of_month = dt[dt['YearMonth'] == dt['YearMonth'].iloc[0]]['Date'].min()


      MonthData = dt.loc[dt['Date'] == first_date_of_month].squeeze()
      WeekData = dt.loc[dt['Date'] == first_date_of_week].squeeze()

      cond1 = abs(daily_values['High'] - daily_values['Low']) > abs(prev_day_values['High'] - prev_day_values['Low']) if prev_day_values is not None else False
      cond2 = abs(daily_values['High'] - daily_values['Low']) > abs(two_days_ago_values['High'] - two_days_ago_values['Low']) if two_days_ago_values is not None else False
      cond3 = abs(daily_values['High'] - daily_values['Low']) > abs(three_days_ago_values['High'] - three_days_ago_values['Low']) if three_days_ago_values is not None else False
      cond4 = abs(daily_values['High'] - daily_values['Low']) > abs(four_days_ago_values['High'] - four_days_ago_values['Low']) if four_days_ago_values is not None else False
      cond5 = daily_values['Close'] > daily_values['Open']
      cond6 = daily_values['Close'] > WeekData['Open']  # Assuming Weekly Open is the same as Daily Close for this example
      cond7 = daily_values['Close'] > MonthData['Open']  # Assuming Monthly Open is the same as Daily Close for this example
      cond8 = daily_values['Low'] > (prev_day_values['Close'] - abs(prev_day_values['Close'] / 222)) if prev_day_values is not None else False
      cond9 = daily_values['Volume'] >= daily_values['Volume_EMA20']
      cond10 = daily_values['Close'] >= daily_values['50_SMA']
      cond11 = daily_values['Close'] >= daily_values['20_SMA']

      result = cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8 and cond9 and cond10 and cond11

      if result:
        stockList.append(symbol)
      
    except Exception as e:
      print("An error occurred while fetching the data:", str(e))

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
        "ROCE" : roce,}

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

    response = requests.get(url)

    if response.status_code == 200:
        print("Successfully fetched the webpage")
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')
    
    
    fundainfo, shareholdnres = extract_key_insights(soup)
    
    return fundainfo, shareholdnres


def companyDetails(company_data):
    # Streamlit layout
    st.title(company_data['Company Name'])

    # Display current price and market cap in columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Price", company_data['Current Price'])
    with col2:
        st.metric("Market Cap (₹ Cr)", company_data['Market Cap'])

    # Display PE, ROE, and ROCE in columns
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("P/E Ratio", company_data['PE'])
    with col4:
        st.metric("ROE (%)", company_data['ROE'])
    with col5:
        st.metric("ROCE (%)", company_data['ROCE'])

    # Display the "About" section with some emphasis
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
    


def plotChart(symbol):
    stock = yf.Ticker(symbol)

    df = stock.history(period="1y", interval="1d")
    df = df.reset_index()

    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'])])
    
    st.plotly_chart(fig, use_container_width=True)
    


# Helper function to convert percentage strings to floats
def convert_to_float(data):
    if all(isinstance(item, str) for item in data):
        # Remove commas and convert percentages
        if all('%' in item for item in data):
            return [float(value.replace(',', '').strip('%')) for value in data]
        # Remove commas and convert numeric strings
        else:
            return [float(value.replace(',', '')) for value in data]
    return data

def check_status(values):
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
        check_status(data['Quarter'])

        st.subheader("2. Yearly Profit Status:")
        check_status(data['Yearly'])

        # Check Shareholding data (Promoters, DII, FII, Public)
        st.subheader("3. Promoters Shareholding Status:")
        check_status(data['Promoters'])
        
    with col2:
        st.subheader("4. DII Shareholding Status:")
        check_status(data['DII'])

        st.subheader("5. FII Shareholding Status:")
        check_status(data['FII'])

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
    
    st.plotly_chart(fig, use_container_width=True)



def StockScan():

    stockList = []
    if "stockList" not in st.session_state:
        st.session_state.stockList = []
        
    if st.button("Run Scan"):
        st.title("Running Scan on NIFTY500")
        stockList = BreakoutVolume()
        st.session_state.stockList = stockList  # Store in session state
        
    
    if st.session_state.stockList:
        st.success(f'Scan Complete : {len(st.session_state.stockList)} Stocks Found', icon="✅")
        
        st.subheader("Stocks")
        cols = st.columns(2)
        for i, stock in enumerate(st.session_state.stockList):
            cols[i % 2].write(stock)
        
        option = st.selectbox(
            "List of Stocks",
            st.session_state.stockList,
            index=None,
            placeholder="Select the Stock",
        )

        st.write("You selected:", option)
        
        if option != None:
           fundainfo, shareholdnres = scrapper(option)
           companyDetails(fundainfo) 
           chart(ticker=option)
           plotChart(option)
           analyze_financial_data(shareholdnres)
           plotShareholding(shareholdnres)
           CompanyNews(fundainfo['Company Name'])
           
           
           
        
        

    
    