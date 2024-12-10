import pandas as pd
import logging

import pandas as pd
import json
import os
import logging
from typing import Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def transforms_financial_data(json_paths: str, 
                              output_path: str = "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/Cloud_Code/Projects-Porfolio/Tech-Sector-Deep-Dive/data/processed/financial_statements_merged.csv") -> pd.DataFrame:
    """
    Transforms multiple JSON financial statement files into a single DataFrame and saves as a CSV file.
    
    This function takes a string of comma-separated file paths to JSON financial statement files.
    It will:
    1. Parse each JSON file
    2. Check for required keys: 'income_statement', 'annualReports', 'quarterlyReports', 'symbol'
    3. Convert available reports into DataFrames and concatenate them
    4. Save the merged DataFrame as a CSV file in the specified output directory
    5. Return the resulting DataFrame
    
    Args:
        json_paths (str): A comma-separated string of file paths to JSON financial statement files.
        output_path (str): The file path where the merged CSV will be saved.
                           Defaults to "/data/processed/financial_statements_merged.csv"
    
    Returns:
        pd.DataFrame: The combined DataFrame containing data from all provided JSON files.
    
    Raises:
        ValueError: If no valid JSON files are provided, or if no valid DataFrames could be created.
        FileNotFoundError: If any of the provided file paths do not exist.
    """
    if not json_paths:
        logging.error("No JSON paths provided.")
        raise ValueError("The 'json_paths' argument cannot be empty.")
    
    file_list = [path.strip() for path in json_paths.split(',') if path.strip()]
    if not file_list:
        logging.error("No valid file paths extracted from the provided string.")
        raise ValueError("No valid file paths found in the 'json_paths' string.")
    
    all_dfs = []
    
    for fpath in file_list:
        if not os.path.exists(fpath):
            logging.error(f"File not found: {fpath}")
            raise FileNotFoundError(f"File not found: {fpath}")
        
        try:
            with open(fpath, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError as jde:
            logging.error(f"JSON decoding error in file {fpath}: {str(jde)}")
            continue
        except Exception as e:
            logging.error(f"Unexpected error reading file {fpath}: {str(e)}")
            continue
        
        # Initialize variables
        symbol = None
        annual_data = []
        quarterly_data = []
        
        # Check for 'income_statement' key
        if "income_statement" in data:
            income_data = data["income_statement"]
            symbol = income_data.get("symbol")
            annual_data = income_data.get("annualReports", [])
            quarterly_data = income_data.get("quarterlyReports", [])
        else:
            # Try reading 'symbol', 'annualReports', 'quarterlyReports' at top level
            symbol = data.get("symbol")
            annual_data = data.get("annualReports", [])
            quarterly_data = data.get("quarterlyReports", [])
            if symbol is None:
                logging.warning(f"'symbol' key not found in {fpath}. Skipping this file.")
                continue
        
        if not annual_data and not quarterly_data:
            logging.warning(f"No 'annualReports' or 'quarterlyReports' data found in {fpath}.")
            continue

        # Process annual reports if available
        if annual_data:
            try:
                df_annual = pd.json_normalize(annual_data)
                # If df_annual is empty or malformed, skip
                if df_annual.empty:
                    logging.warning(f"'annualReports' is empty or invalid in {fpath}. Skipping annual data.")
                else:
                    df_annual["report_type"] = "annual"
                    df_annual["symbol"] = symbol
                    all_dfs.append(df_annual)
            except ValueError as ve:
                logging.error(f"ValueError normalizing 'annualReports' in {fpath}: {ve}")
            except Exception as e:
                logging.error(f"Unexpected error normalizing 'annualReports' in {fpath}: {e}")

        # Process quarterly reports if available
        if quarterly_data:
            try:
                df_quarterly = pd.json_normalize(quarterly_data)
                if df_quarterly.empty:
                    logging.warning(f"'quarterlyReports' is empty or invalid in {fpath}. Skipping quarterly data.")
                else:
                    df_quarterly["report_type"] = "quarterly"
                    df_quarterly["symbol"] = symbol
                    all_dfs.append(df_quarterly)
            except ValueError as ve:
                logging.error(f"ValueError normalizing 'quarterlyReports' in {fpath}: {ve}")
            except Exception as e:
                logging.error(f"Unexpected error normalizing 'quarterlyReports' in {fpath}: {e}")

        logging.info(f"Finished processing file: {fpath}")

    if not all_dfs:
        logging.error("No valid DataFrames were loaded from the provided JSON files.")
        raise ValueError("No valid DataFrames were loaded.")
    
    # Concatenate all DataFrames
    try:
        final_df = pd.concat(all_dfs, ignore_index=True)
    except ValueError as ve:
        logging.error(f"Error concatenating DataFrames: {ve}")
        raise ValueError("Failed to concatenate DataFrames due to inconsistent data structures.")
    
    # Attempt to sort; if 'fiscalDateEnding' is missing, just skip sorting by that column
    if "fiscalDateEnding" in final_df.columns:
        final_df = final_df.sort_values(by=["symbol", "fiscalDateEnding"], ascending=[True, False])
    else:
        final_df = final_df.sort_values(by=["symbol"], ascending=[True])
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created directory: {output_dir}")
    
    # Save to CSV
    try:
        final_df.to_csv(output_path, index=False)
        logging.info(f"Data successfully saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV: {str(e)}")
        raise e
    
    return final_df

def merge_stock_data(stock_dfs: list) -> pd.DataFrame:
    """
    Merge multiple stock DataFrames into a single DataFrame.
    
    Args:
        stock_dfs (list): List of stock DataFrames to merge. Each DataFrame should have
                         a 'Date' column and consistent column names across DataFrames.
        
    Returns:
        pd.DataFrame: Merged DataFrame containing all stock data
        
    Raises:
        ValueError: If stock_dfs is empty or if DataFrames are missing required columns
        TypeError: If input is not a list of DataFrames
    """
    # Input validation
    if not isinstance(stock_dfs, list):
        raise TypeError("Input must be a list of DataFrames")
    if not stock_dfs:
        raise ValueError("Input list cannot be empty")
    if not all(isinstance(df, pd.DataFrame) for df in stock_dfs):
        raise TypeError("All elements must be pandas DataFrames")
    if not all('Date' in df.columns for df in stock_dfs):
        raise ValueError("All DataFrames must contain a 'Date' column")
        
    try:
        # Create empty DataFrame to store merged data
        merged_df = pd.DataFrame()
        
        # Merge each stock DataFrame
        for i, df in enumerate(stock_dfs):
            logging.info(f"Merging DataFrame {i+1} of {len(stock_dfs)}")
            if merged_df.empty:
                merged_df = df.copy()  # Use copy to avoid modifying original
            else:
                # Perform outer merge to keep all dates
                merged_df = pd.merge(merged_df, df, on='Date', how='outer')
                
        # Sort by date and handle any missing values
        merged_df = merged_df.sort_values('Date')
        merged_df = merged_df.fillna(method='ffill')  # Forward fill missing values
        
        logging.info(f"Successfully merged {len(stock_dfs)} stock DataFrames with {len(merged_df)} total rows")
        return merged_df
        
    except Exception as e:
        logging.error(f"Error merging stock data: {str(e)}")
        raise



