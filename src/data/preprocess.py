import pandas as pd
import numpy as np
import logging
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def preprocess_financial_data(df: pd.DataFrame, date_col: str = "fiscalDateEnding") -> pd.DataFrame:
    """
    Clean and preprocess financial statements DataFrame.
    
    Steps:
    - Convert date_col to datetime
    - Sort by symbol and date
    - Handle missing values by either dropping or filling them
    - Ensure correct data types for numeric columns
    
    Args:
        df (pd.DataFrame): Financial statements DataFrame.
        date_col (str): Name of the date column in the dataset.
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed financial statements DataFrame.
    """
    # Convert date column to datetime
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        except Exception as e:
            logging.error(f"Error converting {date_col} to datetime: {e}")

    # Sort by symbol and date
    if "symbol" in df.columns and date_col in df.columns:
        df = df.sort_values(by=["symbol", date_col])
    
    # Drop duplicates
    df = df.drop_duplicates()

    # Convert numeric columns to float
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values (example: fill with 0 or forward fill)
    df = df.fillna(method='ffill').fillna(0)

    logging.info("Financial data preprocessing complete.")
    return df


def preprocess_stock_data(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Clean and preprocess stock DataFrame.
    
    Steps:
    - Convert date_col to datetime
    - Sort by date
    - Set date as index if desired
    - Compute additional metrics (e.g., daily returns)
    
    Args:
        df (pd.DataFrame): Stock DataFrame.
        date_col (str): Name of the date column in the dataset.
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed stock DataFrame.
    """
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(by=date_col)
        df.set_index(date_col, inplace=True)

    # Handle missing numerical values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')

    # Compute daily returns as an example
    if 'Close' in df.columns:
        df['daily_return'] = df['Close'].pct_change().fillna(0)

    # Drop duplicates
    df = df.drop_duplicates()

    logging.info("Stock data preprocessing complete.")
    return df


def merge_financial_and_stocks(financial_df: pd.DataFrame, stock_df: pd.DataFrame, 
                               financial_date_col: str = "fiscalDateEnding", 
                               stock_date_col: str = "Date") -> pd.DataFrame:
    """
    Merge the financial dataset and stocks dataset on symbol and date.
    
    This may require aligning dates. Since the financial statements are often quarterly or annually,
    consider merging on symbol and the closest available date or exactly matching dates if possible.

    Args:
        financial_df (pd.DataFrame): Preprocessed financial DataFrame.
        stock_df (pd.DataFrame): Preprocessed stock DataFrame.
        financial_date_col (str): The date column in the financial_df.
        stock_date_col (str): The date column in the stock_df index or columns.
    
    Returns:
        pd.DataFrame: Merged DataFrame with financial and stock data.
    """
    # If financial_date_col isn't the index, set a multi-index by symbol and fiscal date
    if financial_date_col in financial_df.columns:
        financial_df.set_index(["symbol", financial_date_col], inplace=True)
    
    if stock_df.index.name != stock_date_col:
        stock_df.reset_index(inplace=True)
    
    # If the stocks also have a symbol column, we can do a multi-index merge
    if "symbol" in stock_df.columns:
        stock_df.set_index(["symbol", stock_date_col], inplace=True)

    # Perform an outer or left join as needed
    merged_df = financial_df.join(stock_df, how='left', rsuffix='_stock')
    
    # Fill missing stock data
    merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
    
    # Reset index if desired
    merged_df.reset_index(inplace=True)

    logging.info("Merging of financial and stock data complete.")
    return merged_df



if __name__ == "__main__":
    main()