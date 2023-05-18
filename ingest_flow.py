from pathlib import Path
import pandas as pd
from prefect import flow, task
import os
import kaggle
import zipfile
import multiprocessing

# @task(retries=3, log_prints=True)
def fetch_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('jacksoncrow/stock-market-dataset', 'stock-market-dataset/', quiet=False, unzip=False)

    with zipfile.ZipFile("./stock-market-dataset/stock-market-dataset.zip", "r") as zip_ref:
        for name in zip_ref.namelist():
            try:
                zip_ref.extract(name, "stock-market-dataset/")
            except Exception as e:
                print(e)

# @task(log_prints=True)
def combine_data() -> pd.DataFrame:
    path = os.getcwd()
    symbols_df = pd.read_csv('./stock-market-dataset/symbols_valid_meta.csv')

    etf_files = Path(path+'\stock-market-dataset\etfs').glob('*.csv')
    etf_dfs = []
    for f in etf_files:
        ticker = os.path.basename(f).split(".")[0]
        etf_df = pd.read_csv(f)
        etf_df['Symbol'] = ticker
        etf_dfs.append(etf_df)
        
    combined_etfs = pd.concat(etf_dfs)
    # merge to pick up security name
    combined_etfs = pd.merge(combined_etfs, symbols_df[['Symbol', 'Security Name']], how='left', on='Symbol')

    stock_files = Path(path+'\stock-market-dataset\stocks').glob('*.csv')
    stock_dfs = []
    for f in stock_files:
        ticker = os.path.basename(f).split(".")[0]
        stock_df = pd.read_csv(f)
        stock_df['Symbol'] = ticker
        stock_dfs.append(stock_df)
        
    combined_stocks = pd.concat(stock_dfs)
    # merge to pick up security name
    combined_stocks = pd.merge(combined_stocks, symbols_df[['Symbol', 'Security Name']], how='left', on='Symbol')
    all_securities_df = pd.concat([combined_etfs, combined_stocks])

    # rearrange columns
    all_securities_df = all_securities_df[['Symbol', 'Security Name', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    return all_securities_df

# @task(log_prints=True)
def clean(df) -> pd.DataFrame:
    '''
    null values exist in the following columns.
    We will use drop N/A to get ride of the columns with no open, high, low etc. amount and ignore security name for now
    Symbol               0
    Security Name    21524
    Date                 0
    Open               683
    High               683
    Low                683
    Close              683
    Adj Close          683
    Volume             683
    '''
    df = df.dropna(subset=['Open'])

    """Fix dtype issues"""
    df.Date = pd.to_datetime(df.Date)
    df['Symbol'] = df['Symbol'].astype(str)
    df['Security Name'] = df['Security Name'].astype(str)

    """Sort dataframe by symmbol and date"""
    df = df.sort_values(['Symbol', 'Date'])

    return df

# @task(log_prints=True)
def calculate_rolling_averages(data):
    window = 30
    symbol, group = data
    group['vol_moving_avg'] = group['Volume'].rolling(window=window, min_periods=window).mean()
    group['adj_close_rolling_med'] = group['Adj Close'].rolling(window=window, min_periods=window).mean()
    return group

# @task(log_prints=True)
def apply_rolling_averages(df):
    with multiprocessing.Pool() as pool:
        results = pool.map(calculate_rolling_averages, df.groupby('Symbol'))
    new_df = pd.concat(results)
    return new_df

if __name__ == "__main__":
    fetch_dataset()
    combined_df = combine_data()
    combined_df.to_parquet("solution-1.parquet", index=False)
    cleaned_df = clean(combined_df)
    rolling_df = apply_rolling_averages(cleaned_df)
    rolling_df.to_parquet("solution-2.parquet", index=False)