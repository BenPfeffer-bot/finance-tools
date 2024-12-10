import os
import pandas as pd
import logging
from datetime import datetime
import sys
sys.path.append('/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/Cloud_Code/Projects-Porfolio/Tech-Sector-Deep-Dive/')

from src.helper.metrics import calculate_metrics, calculate_market_metrics


def load_merged_data(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        # Load financial metrics data
        financial_metrics_path = f"/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/Cloud_Code/Projects-Porfolio/Tech-Sector-Deep-Dive/data/processed/fundamentals-analysis/metrics_annual_{ticker}.csv"
        if not os.path.exists(financial_metrics_path):
            raise FileNotFoundError(f"Financial metrics file not found for {ticker}")
        financial_metrics = pd.read_csv(financial_metrics_path)
        
        # Load market data
        market_data_path = f'/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/Cloud_Code/Projects-Porfolio/Tech-Sector-Deep-Dive/data/raw/{ticker}_historical_data.csv'
        if not os.path.exists(market_data_path):
            raise FileNotFoundError(f"Market data file not found for {ticker}")
        market_data = pd.read_csv(market_data_path)

        logging.info(f"Loading market data for {ticker}")
        # Convert and standardize dates
        market_data['Date'] = pd.to_datetime(market_data['Date'], utc=True).dt.strftime('%Y-%m-%d')
        
        # Calculate market metrics
        try:
            market_metrics = calculate_market_metrics(market_data)
        except Exception as e:
            logging.error(f"Error calculating market metrics: {str(e)}")
            raise
            
        logging.info(f"Calculating financial metrics for {ticker}")
        
        # Calculate financial metrics
        try:
            financial_metrics = calculate_metrics(financial_metrics)
        except Exception as e:
            logging.error(f"Error calculating financial metrics: {str(e)}")
            raise
            
        financial_metrics = financial_metrics.rename(columns={'Year': 'Date'})
        logging.info(f"Merging financial and market metrics for {ticker}")

        # Convert dates to datetime for merging
        financial_metrics['Date'] = pd.to_datetime(financial_metrics['Date'], errors='coerce')
        market_metrics['Date'] = pd.to_datetime(market_metrics['Date'], errors='coerce')
        
        # Check for null dates after conversion
        if financial_metrics['Date'].isnull().any():
            logging.warning("Some dates in financial metrics could not be parsed")
        if market_metrics['Date'].isnull().any():
            logging.warning("Some dates in market metrics could not be parsed")

        # Return the processed dataframes
        return financial_metrics, market_metrics

    except Exception as e:
        logging.error(f"Error in load_merged_data for {ticker}: {str(e)}")
        raise