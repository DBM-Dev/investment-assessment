'''This module contains functions and scripts used to collect stock market data
and parse it into standardized DataFrames.

by: David Martin
on: July 16, 2024'''

import requests
import datetime
import os
import logging
import time

import pandas as pd

logger = logging.getLogger('main_logger')

DEFAULT_TICKERS = ['VTI', 'VNQ', 'VNQI', 'VOO', 'VYM', 'CRSP']


def get_api_key():
    '''Retrieve the Alpha Vantage API key from environment variable.

    OUTPUTS:
    str - the API key

    RAISES:
    ValueError if the API key is not configured'''
    key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not key:
        raise ValueError(
            'ALPHA_VANTAGE_API_KEY environment variable is not set. '
            'Set it with: export ALPHA_VANTAGE_API_KEY=your_key_here'
        )
    return key


def fetch_weekly_adjusted(ticker, api_key, output_size='full'):
    '''Fetch weekly adjusted time series data from Alpha Vantage for a
    single ticker.

    INPUTS:
    ticker - string stock ticker symbol
    api_key - Alpha Vantage API key
    output_size - 'full' for 20+ years, 'compact' for last 100 data points

    OUTPUTS:
    bytes - raw CSV content from the API

    RAISES:
    requests.HTTPError on HTTP failure
    ValueError on rate limit or malformed response'''
    url = (
        f'https://www.alphavantage.co/query'
        f'?function=TIME_SERIES_WEEKLY_ADJUSTED'
        f'&symbol={ticker}'
        f'&outputsize={output_size}'
        f'&datatype=csv'
        f'&apikey={api_key}'
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    content = response.content.decode('utf-8', errors='replace')

    # Alpha Vantage returns JSON error messages even when requesting CSV
    if content.strip().startswith('{'):
        if 'rate limit' in content.lower() or 'api call frequency' in content.lower():
            raise ValueError(f'API rate limit reached for {ticker}. '
                             f'Response: {content[:200]}')
        if 'error' in content.lower() or 'invalid' in content.lower():
            raise ValueError(f'API error for {ticker}: {content[:200]}')

    # Minimal sanity check: CSV should have a header line with 'timestamp'
    if 'timestamp' not in content.lower()[:200]:
        raise ValueError(f'Unexpected response format for {ticker}: '
                         f'{content[:200]}')

    return response.content


def download_ticker(ticker, api_key, write_loc='./', output_size='full'):
    '''Download weekly adjusted data for a single ticker and save to CSV.

    INPUTS:
    ticker - stock ticker symbol
    api_key - Alpha Vantage API key
    write_loc - directory to write the CSV file
    output_size - 'full' or 'compact'

    OUTPUTS:
    str - path to the written file'''
    content = fetch_weekly_adjusted(ticker, api_key, output_size)
    date = str(datetime.datetime.today().date())
    fpath = os.path.join(write_loc, f'Weekly{ticker}{date}.csv')

    with open(fpath, 'wb') as file:
        file.write(content)
    logger.info(f'Wrote Weekly adjusted {ticker} to {fpath}')
    return fpath


def main(write_loc='./', tickers=None, wait_between=15):
    '''Download weekly adjusted data for all configured tickers.

    INPUTS:
    write_loc - directory to save CSV files
    tickers - list of ticker symbols (defaults to DEFAULT_TICKERS)
    wait_between - seconds to wait between API calls to avoid rate limits

    OUTPUTS:
    list of file paths written'''
    if tickers is None:
        tickers = DEFAULT_TICKERS

    api_key = get_api_key()
    paths = []

    for i, ticker in enumerate(tickers):
        try:
            fpath = download_ticker(ticker, api_key, write_loc)
            paths.append(fpath)
            logger.info(f'Downloaded {ticker} ({i + 1}/{len(tickers)})')
        except requests.HTTPError as e:
            logger.error(f'HTTP error downloading {ticker}: {e}')
        except ValueError as e:
            logger.error(f'Data error for {ticker}: {e}')
        except requests.ConnectionError as e:
            logger.error(f'Connection error for {ticker}: {e}')

        # Wait between calls to respect rate limits (except after last ticker)
        if i < len(tickers) - 1 and wait_between > 0:
            time.sleep(wait_between)

    logger.info(f'Finished downloading {len(paths)}/{len(tickers)} tickers')
    return paths


def parse_csv(csv_path):
    '''Parse a downloaded Alpha Vantage weekly adjusted CSV into a
    standardized DataFrame.

    INPUTS:
    csv_path - path to the CSV file

    OUTPUTS:
    DataFrame with columns: date, open, high, low, close, adjusted_close,
    volume, dividend_amount, avg_price'''
    raw = pd.read_csv(csv_path)
    raw.columns = [c.strip().lower().replace(' ', '_') for c in raw.columns]

    df = pd.DataFrame()
    df['date'] = pd.to_datetime(raw['timestamp'])
    df['open'] = raw['open'].astype(float)
    df['high'] = raw['high'].astype(float)
    df['low'] = raw['low'].astype(float)
    df['close'] = raw['close'].astype(float)
    df['adjusted_close'] = raw['adjusted_close'].astype(float)
    df['volume'] = raw['volume'].astype(int)

    if 'dividend_amount' in raw.columns:
        df['dividend_amount'] = raw['dividend_amount'].astype(float)
    else:
        df['dividend_amount'] = 0.0

    # Average of weekly high and low as representative price (per design.MD)
    df['avg_price'] = (df['high'] + df['low']) / 2.0

    df = df.sort_values('date').reset_index(drop=True)

    logger.info(f'Parsed {len(df)} records from {csv_path}')
    return df


def compute_avg_price(df):
    '''Compute the average of weekly high and low as the representative price.

    INPUTS:
    df - DataFrame with 'high' and 'low' columns

    OUTPUTS:
    Series of average prices'''
    return (df['high'] + df['low']) / 2.0


def update_csv(existing_path, ticker, api_key, write_loc='./'):
    '''Incrementally update an existing CSV with new data.

    Fetches compact data (last 100 data points) and appends only the
    rows with dates beyond the latest date in the existing file.

    INPUTS:
    existing_path - path to the existing CSV file
    ticker - stock ticker symbol
    api_key - Alpha Vantage API key
    write_loc - directory for the updated file

    OUTPUTS:
    str - path to the updated file'''
    existing = parse_csv(existing_path)
    latest_date = existing['date'].max()

    # Fetch compact (recent) data
    content = fetch_weekly_adjusted(ticker, api_key, output_size='compact')

    # Write to a temp file for parsing
    temp_path = os.path.join(write_loc, f'_temp_{ticker}.csv')
    with open(temp_path, 'wb') as f:
        f.write(content)

    new_data = parse_csv(temp_path)
    os.remove(temp_path)

    # Keep only rows newer than existing data
    new_rows = new_data.loc[new_data['date'] > latest_date]

    if new_rows.empty:
        logger.info(f'No new data for {ticker} since {latest_date}')
        return existing_path

    updated = pd.concat([existing, new_rows], ignore_index=True)
    updated = updated.sort_values('date').reset_index(drop=True)

    # Write updated file
    date = str(datetime.datetime.today().date())
    output_path = os.path.join(write_loc, f'Weekly{ticker}{date}.csv')
    updated.to_csv(output_path, index=False)

    logger.info(f'Updated {ticker}: added {len(new_rows)} new rows, '
                 f'total {len(updated)} rows')
    return output_path
