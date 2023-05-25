'''
This is a prefect implementation of the ETL flow.
Prefect is used to track process workflow of the data pipeline by tracking and timing tasks, and
provinding a simple platform to manage your infrastructure with blocks (such a docker infra)

To use prefect, you must first have it installed on your computer, start the prefect orion server, then run your flows.
To use with docker, you need to build a flow using docker_deploy.py, run an agent, and pass that flow run to the agent.
'''

from pathlib import Path
import pandas as pd
from prefect import flow, task
import os
import zipfile
from ml_sklearn import train_model_sklearn
from kaggle_build import build_kaggle_json
from multiprocessing import Pool
from tqdm import tqdm
import pickle

@task(retries=3, log_prints=True)
def fetch_dataset():
    # builds kaggle file based on env variables passed on docker run
    # build_kaggle_json()
    
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('jacksoncrow/stock-market-dataset', 'stock-market-dataset/', quiet=False, unzip=False)

    with zipfile.ZipFile("./stock-market-dataset/stock-market-dataset.zip", "r") as zip_ref:
        for name in zip_ref.namelist():
            try:
                zip_ref.extract(name, "stock-market-dataset/")
            except Exception as e:
                print(e)

@task(log_prints=True)
def combine_data() -> pd.DataFrame:
    path = os.getcwd()
    symbols_df = pd.read_csv('./stock-market-dataset/symbols_valid_meta.csv')

    etf_files = Path(path+'/stock-market-dataset/etfs').glob('*.csv')
    etf_dfs = []
    for f in etf_files:
        ticker = os.path.basename(f).split(".")[0]
        etf_df = pd.read_csv(f)
        etf_df['Symbol'] = ticker
        etf_dfs.append(etf_df)
        
    combined_etfs = pd.concat(etf_dfs)
    # merge to pick up security name
    combined_etfs = pd.merge(combined_etfs, symbols_df[['Symbol', 'Security Name']], how='left', on='Symbol')

    stock_files = Path(path+'/stock-market-dataset/stocks').glob('*.csv')
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

@task(log_prints=True)
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
    df = df.copy()

    """Fix dtype issues"""
    df.Date = pd.to_datetime(df.Date)
    df['Symbol'] = df['Symbol'].astype(str)
    df['Security Name'] = df['Security Name'].astype(str)

    """Sort dataframe by symmbol and date"""
    df = df.sort_values(['Symbol', 'Date'])

    return df

@task(log_prints=True)
def create_job_files(df):
    job_dir = 'data/job_files'
    job_files = []
    for i, (symbol, group) in enumerate(df.groupby('Symbol')):
        job_file = os.path.join(job_dir, f'job_{i}.pickle')
        with open(job_file, 'wb') as f:
            pickle.dump((symbol, group), f)
        job_files.append(job_file)

    return job_files

def calculate_rolling_averages(job_file):
    with open(job_file, 'rb') as f:
        symbol, group = pickle.load(f)
        window = 30
        group['vol_moving_avg'] = group['Volume'].rolling(window=window, min_periods=window).mean()
        group['adj_close_rolling_med'] = group['Adj Close'].rolling(window=window, min_periods=window).mean()
        return group

@task(log_prints=True)
def apply_rolling_averages(job_files):
    pool = Pool()
    # Perform multiprocessing with job filenames
    results = []
    with tqdm(total=len(job_files), desc="Processing jobs") as pbar:
        for result in pool.imap(calculate_rolling_averages, job_files):
            results.append(result)
            pbar.update(1)

    # Concatenate the resulting group DataFrames
    new_df = pd.concat(results)
    pool.close()
    pool.join()

    return new_df

@task(log_prints=True)
def train_ml_model(ml_data):
    train_model_sklearn(ml_data)

@flow()
def etl_parent_flow():
    fetch_dataset()
    combined_df = combine_data()
    combined_df.to_parquet("data/solution-1.parquet", index=False)
    cleaned_df = clean(combined_df)
    job_files = create_job_files(cleaned_df)
    rolling_df = apply_rolling_averages(job_files)
    rolling_df.to_parquet("data/solution-2.parquet", index=False)
    train_ml_model(rolling_df)

if __name__ == "__main__":
    etl_parent_flow()