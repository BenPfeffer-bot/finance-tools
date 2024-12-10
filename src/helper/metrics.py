import numpy as np
import pandas as pd
import sys

sys.path.append("/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/Cloud_Code/Projects-Porfolio/Tech-Sector-Deep-Dive/")
from src.utils.load_logging import *

import logging

logging = load_logger()

def calculate_financial_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate various financial metrics from financial statement data, including:
    - Liquidity ratios
    - Profitability ratios
    - Solvency ratios
    - Growth metrics

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing financial statement information. The DataFrame 
        is expected to have at least the following columns for core metrics:
            Profitability (required):
                - 'grossProfit'
                - 'totalRevenue'
                - 'operatingIncome'
                - 'netIncome'
                - 'ebitda'
                - 'operatingExpenses'
                - 'researchAndDevelopment'
                - 'sellingGeneralAndAdministrative'
                - 'incomeTaxExpense'
                - 'incomeBeforeTax'
                - 'ebit'
                - 'interestExpense'
            
            For growth metrics:
                - 'Date' column must exist.

            For Liquidity & Solvency ratios (optional):
                - 'totalCurrentAssets', 'totalCurrentLiabilities', 'inventory', 'cashAndShortTermInvestments'
                - 'totalAssets', 'totalLiabilities', 'totalShareholderEquity'

            If optional columns are missing, those particular ratios will be skipped.

    Returns
    -------
    dict
        A dictionary mapping metric names to Series or arrays, including:
        
        Liquidity Ratios:
            - current_ratio (optional)
            - quick_ratio (optional)
            - cash_ratio (optional)
        
        Profitability Ratios:
            - gross_margin
            - operating_margin
            - net_margin
            - ebitda_margin
            - return_on_assets (optional)
            - return_on_equity (optional)
        
        Solvency Ratios:
            - debt_to_equity (optional)
            - debt_to_assets (optional)
            - interest_coverage
        
        Growth Metrics (if 'Date' provided):
            - revenue_growth
            - net_income_growth
            - ebitda_growth

        Missing ratios due to missing columns will not appear in the dictionary.
    """
    logging.debug("Starting calculation of financial metrics.")

    metrics = {}
    try:
        # Profitability Metrics
        metrics['gross_margin'] = (df['grossProfit'] / df['totalRevenue']) * 100
        metrics['operating_margin'] = (df['operatingIncome'] / df['totalRevenue']) * 100
        metrics['net_margin'] = (df['netIncome'] / df['totalRevenue']) * 100
        metrics['ebitda_margin'] = (df['ebitda'] / df['totalRevenue']) * 100
        logging.debug("Profitability metrics calculated successfully.")

        # Efficiency and operating ratios (as done before)
        metrics['opex_ratio'] = (df['operatingExpenses'] / df['totalRevenue']) * 100
        metrics['rd_to_revenue'] = (df['researchAndDevelopment'] / df['totalRevenue']) * 100
        metrics['sga_to_revenue'] = (df['sellingGeneralAndAdministrative'] / df['totalRevenue']) * 100
        logging.debug("Efficiency metrics calculated successfully.")

        # Tax and Interest Metrics
        metrics['effective_tax_rate'] = (df['incomeTaxExpense'] / df['incomeBeforeTax']) * 100
        metrics['interest_coverage'] = df['ebit'] / df['interestExpense'].replace(0, np.nan)
        logging.debug("Tax and interest metrics calculated successfully.")

        # Liquidity Ratios (check optional columns)
        if 'totalCurrentAssets' in df.columns and 'totalCurrentLiabilities' in df.columns:
            metrics['current_ratio'] = df['totalCurrentAssets'] / df['totalCurrentLiabilities']

            if 'inventory' in df.columns:
                metrics['quick_ratio'] = (df['totalCurrentAssets'] - df['inventory']) / df['totalCurrentLiabilities']

            if 'cashAndShortTermInvestments' in df.columns:
                metrics['cash_ratio'] = df['cashAndShortTermInvestments'] / df['totalCurrentLiabilities']

            logging.debug("Liquidity ratios calculated (where possible).")

        # Solvency Ratios (check optional columns)
        # Debt-to-Equity and Debt-to-Assets
        if 'totalLiabilities' in df.columns and 'totalShareholderEquity' in df.columns:
            # Avoid division by zero
            metrics['debt_to_equity'] = df['totalLiabilities'] / df['totalShareholderEquity'].replace(0, np.nan)

        if 'totalLiabilities' in df.columns and 'totalAssets' in df.columns:
            metrics['debt_to_assets'] = df['totalLiabilities'] / df['totalAssets'].replace(0, np.nan)

        logging.debug("Solvency ratios calculated (where possible).")

        # Additional Profitability Ratios (if columns present)
        if 'totalAssets' in df.columns:
            metrics['return_on_assets'] = (df['netIncome'] / df['totalAssets']) * 100

        if 'totalShareholderEquity' in df.columns:
            metrics['return_on_equity'] = (df['netIncome'] / df['totalShareholderEquity'].replace(0, np.nan)) * 100

        # Growth Metrics
        if 'Date' in df.columns:
            sorted_df = df.sort_values('Date')
            metrics['revenue_growth'] = sorted_df['totalRevenue'].pct_change() * 100
            metrics['net_income_growth'] = sorted_df['netIncome'].pct_change() * 100
            metrics['ebitda_growth'] = sorted_df['ebitda'].pct_change() * 100
            logging.debug("Growth metrics calculated successfully.")
        else:
            logging.warning("No 'Date' column found in the DataFrame. Growth metrics could not be calculated.")

        # Filter out None values
        metrics = {k: v for k, v in metrics.items() if v is not None}
        logging.debug("Finished calculation of all requested financial metrics.")

    except KeyError as e:
        logging.error(f"Missing expected column in DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while calculating financial metrics: {e}")
        raise

    return metrics


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a financial DataFrame to compute standardized financial metrics 
    from raw financial statement data, including liquidity, profitability, 
    solvency, and growth metrics.

    Steps performed:
    1. Set 'Date' column as the index.
    2. Calculate financial metrics using `calculate_financial_metrics`.
    3. Convert percentages to ratios (i.e., divide by 100).
    4. Fill missing values with 0.
    5. Ensure no missing values remain.
    6. Reset index and rename 'Date' column to 'Year'.
    7. Sort the DataFrame by 'Year' in descending order.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw financial statement data. Must include:
        - 'Date' column for indexing and growth calculations.
        - Other columns as required by `calculate_financial_metrics`.

    Returns
    -------
    pd.DataFrame
        A DataFrame of calculated financial metrics indexed by 'Year', sorted in descending order.
        All profitability, return-based, and growth metrics are expressed as ratios (i.e., 0.35 for 35%).
        Ratios that are inherently not percentages (like current_ratio, quick_ratio) remain as is.

    Raises
    ------
    AssertionError
        If there are still missing values in the DataFrame after fill.
    KeyError
        If the 'Date' column or required financial columns are missing.
    Exception
        For any other unexpected errors.
    """
    logging.debug("Starting the metrics calculation and cleaning process.")

    try:
        # Set 'Date' as the index
        df = df.set_index('Date')
        logging.debug("'Date' column set as index successfully.")

        # Calculate the financial metrics dictionary
        metrics = calculate_financial_metrics(df)
        metrics_df = pd.DataFrame(metrics)
        logging.info("Dataframe has been set to index and metrics have been calculated.")

        # Convert percentage-based metrics to ratios
        # Identify which metrics are percentage-based. 
        # In this example, all margins, return_on_assets, return_on_equity, and growth are in percentages.
        percentage_metrics = [m for m in metrics_df.columns if 'margin' in m or 'growth' in m or 'return_on' in m or 'tax_rate' in m]
        for col in percentage_metrics:
            metrics_df[col] = metrics_df[col] / 100

        logging.debug("Percentage-based metrics converted from percentages to ratios.")

        # Fill NA values with 0
        metrics_df = metrics_df.fillna(0)
        logging.debug("NA values in the metrics dataframe have been filled with 0.")

        # Verify that no missing values remain
        assert metrics_df.isnull().sum().sum() == 0, "There are still missing values in the dataframe"
        logging.debug("No missing values remain in the metrics dataframe.")

        # Reset index and rename 'Date' column to 'Year'
        metrics_df = metrics_df.reset_index()
        metrics_df = metrics_df.rename(columns={'Date': 'Year'})
        logging.debug("'Date' column renamed to 'Year' and dataframe index reset.")

        # Sort by Year in descending order
        metrics_df = metrics_df.sort_values('Year', ascending=False)
        logging.debug("Dataframe sorted by 'Year' in descending order.")

        logging.info("Dataframe has been cleaned and prepared successfully.")
        return metrics_df

    except AssertionError as e:
        logging.error(e)
        raise
    except KeyError as e:
        logging.error(f"Required column missing in DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during metrics calculation: {e}")
        raise

def calculate_market_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various technical market metrics and indicators from historical stock price data.
    
    This function computes a set of metrics that help in understanding price trends, 
    volatility, momentum, and overall trading conditions. The returned DataFrame 
    retains the original data plus the new computed metrics.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing historical stock price data with at least the following columns:
            - 'Date' (datetime-like or string convertible to datetime)
            - 'Open'
            - 'High'
            - 'Low'
            - 'Close'
            - 'Volume'

        Optional but not directly used in these calculations:
            - 'Dividends'
            - 'Stock Splits'

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the original columns and the following calculated metrics:
            - daily_returns: Daily percentage change in closing price.
            - trading_range: (High - Low) / Open, indicating intraday volatility.
            - volume_change: Daily percentage change in volume.
            - volatility_20d: 20-day rolling standard deviation of daily returns.
            - MA20: 20-day moving average of the Close price.
            - MA50: 50-day moving average of the Close price.
            - RSI: 14-day Relative Strength Index.

        Missing values in the computed metrics are filled with 0.
    
    Raises
    ------
    KeyError
        If one or more of the required columns ('Date', 'Open', 'High', 'Low', 'Close', 'Volume') 
        are missing from the DataFrame.
    Exception
        If any unexpected error occurs during metric calculation, it will be logged and raised.
    """
    logging.debug("Starting calculation of market metrics.")

    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    try:
        # Validate required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # Make a copy to avoid modifying the original DataFrame
        metrics_df = df.copy()
        logging.debug("Dataframe copy created successfully.")

        # Ensure 'Date' column is in datetime format for consistent time-series operations
        if not pd.api.types.is_datetime64_any_dtype(metrics_df['Date']):
            metrics_df['Date'] = pd.to_datetime(metrics_df['Date'], errors='coerce')
            # If after coercion any NaT found, raise an error
            if metrics_df['Date'].isna().any():
                raise ValueError("Unable to convert some 'Date' values to datetime.")

        # Sort by Date to ensure chronological calculations
        metrics_df = metrics_df.sort_values('Date')
        logging.debug("Dataframe sorted by 'Date' successfully.")

        # Calculate daily returns
        metrics_df['daily_returns'] = metrics_df['Close'].pct_change()
        logging.debug("Daily returns calculated.")

        # Calculate trading range
        metrics_df['trading_range'] = (metrics_df['High'] - metrics_df['Low']) / metrics_df['Open']
        logging.debug("Trading range calculated.")

        # Calculate volume change
        metrics_df['volume_change'] = metrics_df['Volume'].pct_change()
        logging.debug("Volume change calculated.")

        # Calculate 20-day price volatility
        metrics_df['volatility_20d'] = metrics_df['daily_returns'].rolling(window=20).std()
        logging.debug("20-day volatility calculated.")

        # Calculate moving averages
        metrics_df['MA20'] = metrics_df['Close'].rolling(window=20).mean()
        metrics_df['MA50'] = metrics_df['Close'].rolling(window=50).mean()
        logging.debug("Moving averages (MA20, MA50) calculated.")

        # Calculate RSI
        delta = metrics_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        # If all losses are 0, this can lead to division by zero. Replace 0 with np.nan to avoid errors
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        metrics_df['RSI'] = 100 - (100 / (1 + rs))
        logging.debug("RSI calculated.")

        # Fill NaN values
        metrics_df = metrics_df.fillna(0)
        logging.debug("NaN values filled with 0.")

        logging.info("Market metrics calculated successfully.")
        return metrics_df

    except KeyError as e:
        logging.error(f"Missing expected column in DataFrame: {e}")
        raise
    except ValueError as e:
        logging.error(f"Value error in data processing: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while calculating market metrics: {e}")
        raise