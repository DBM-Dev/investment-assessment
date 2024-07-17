'''This module contains functions and scripts used to collect stock market data

by: David Martin
on: July 16, 2024'''

import requests
from _apikey import apikey
import datetime
import os

def main(write_loc='./'):
    '''Runs the main loop'''
    stock_tickers = ['VTI','VNQ','VNQI','VOO','VYM','CRSP']
    date = str(datetime.datetime.today().date())
    cnt = 0
    for stock_ticker in stock_tickers:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={stock_ticker}&outputsize=full&datatype=csv&apikey={apikey}'
        r = requests.get(url)
        fpath = os.path.join(write_loc,f'Weekly{stock_ticker}{date}.csv')
        with open(fpath,'wb') as file:
            file.write(r.content)
            print(f'Wrote Weekly adjusted {stock_ticker} to {fpath}')
        cnt+=1
    print(f'Finished, wrote {cnt} stock tickers')
            
