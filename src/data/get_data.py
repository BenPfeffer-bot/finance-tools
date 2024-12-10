import pandas as pd
import numpy as np
import requests
import json
import logging
import os

import yfinance as yf
import alpha_vantage as av

from src.utils.load_config import load_config
from src.utils.load_logging import load_logger 
from src.data.load_data import save_data, save_json

config = load_config()
logger = load_logger()

def get_historical_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    """
    Retrieves historical stock price data for a given ticker using yfinance.
    """
    logger.info(f"Retrieving historical data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.to_csv(f"/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/Cloud_Code/Projects-Porfolio/Tech-Sector-Deep-Dive/data/raw/{ticker}_historical_data.csv")
    logger.info(f"Historical data for {ticker} saved to /data/raw/{ticker}_historical_data.csv")
    return df

def get_financial_statements(symbol: str, api_key: str = config['av']['api_key']) -> pd.DataFrame:
    logger.info(f"Retrieving financial statements for {symbol}...")
    
    # Get income statement data
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}"
    income_data = requests.get(url).json()
    
    # Get balance sheet data
    url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={api_key}"
    balance_data = requests.get(url).json()
    
    # Get cash flow data
    url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={api_key}"
    cash_flow_data = requests.get(url).json()

    # Combine all financial data
    financial_data = {
        'income_statement': income_data,
        'balance_sheet': balance_data, 
        'cash_flow': cash_flow_data
    }

    # Save combined data
    save_json(financial_data, f"/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/Cloud_Code/Projects-Porfolio/Tech-Sector-Deep-Dive/data/raw/{symbol}_financial_statements.json")
    logger.info(f"All financial statements for {symbol} saved to /data/raw/{symbol}_financial_statements.json")
    
    return financial_data

def news_sentiment(ticker: str, api_key: str = config['av']['api_key']) -> pd.DataFrame:
    """
    Retrieve the latest news sentiment data for a given ticker using the Alpha Vantage News Sentiment API.

    This function queries the Alpha Vantage API for news sentiment related to the given ticker,
    and returns the data as a pandas DataFrame. It also saves the raw JSON data to the 
    'data/news' directory. If the directory does not exist, it will be created.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g., "AAPL") for which to retrieve news sentiment.
    api_key : str, optional
        Your Alpha Vantage API key. Defaults to the 'api_key' entry in the configuration.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the news sentiment data. The DataFrame may include columns such as
        headlines, summaries, sources, and sentiment scores, depending on the API response.
        If no data is available or an error occurs, an empty DataFrame is returned.

    Raises
    ------
    Exception
        If an unexpected error occurs during data retrieval or parsing, it will be logged and
        an empty DataFrame will be returned.
    """
    logger.info(f"Retrieving news sentiment data for {ticker}...")
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
    
    # Ensure the data/news directory exists
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    news_dir = os.path.join(project_root, "data", "news")
    os.makedirs(news_dir, exist_ok=True)
    logger.info(f"News sentiment data directory: {news_dir}")
    
    try:
        response = requests.get(url, timeout=10)  # Set a timeout for the request
        response.raise_for_status()  # Raise HTTPError if request returned an unsuccessful status code

        data = response.json()
        json_path = os.path.join(news_dir, f"{ticker}_news_sentiment.json")
        save_json(data, json_path)
        logger.info(f"News sentiment data for {ticker} saved to {json_path}.")

        # Parse the JSON data into a DataFrame
        if "feed" in data:
            df = pd.DataFrame(data["feed"])
            if df.empty:
                logger.warning(f"No news articles found in 'feed' for {ticker}. Returning empty DataFrame.")
                return pd.DataFrame()
            
            # Save DataFrame to CSV
            csv_path = os.path.join(news_dir, f"{ticker}_news_sentiment.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"News sentiment DataFrame for {ticker} saved to {csv_path}.")
        else:
            logger.warning(f"No 'feed' key found in news sentiment data for {ticker}. Returning empty DataFrame.")
            return pd.DataFrame()

        logger.info(f"News sentiment data for {ticker} successfully retrieved and processed.")
        return df
    

    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred while retrieving news sentiment for {ticker}: {http_err}")
        return pd.DataFrame()
    except requests.RequestException as req_err:
        logger.error(f"Network error occurred while retrieving news sentiment for {ticker}: {req_err}")
        return pd.DataFrame()
    except ValueError as val_err:
        logger.error(f"Error parsing JSON data for {ticker}: {val_err}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred while retrieving news sentiment for {ticker}: {e}")
        return pd.DataFrame()

def get_market_indicators(ticker: str, api_key: str = config['av']['api_key']) -> pd.DataFrame:
    """
    Retrieves market indicators for a given ticker using Alpha Vantage.
    # """
    # url = f"https://www.alphavantage.co/query?function=MARKET_STATUS&tickers={ticker}&apikey={api_key}"
    # r = requests.get(url)
    # data = r.json()
    # return data
    pass

