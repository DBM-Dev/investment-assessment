'''Shared test fixtures for the investment assessment test suite.'''

import pytest
import pandas as pd
import numpy as np
import os
import tempfile


@pytest.fixture
def sample_price_history():
    '''Create a sample price history DataFrame for testing.'''
    dates = pd.date_range(start='2023-01-02', periods=12, freq='W-MON')
    prices = [100.0, 102.0, 99.0, 105.0, 103.0, 107.0,
              110.0, 108.0, 112.0, 115.0, 113.0, 118.0]
    dividends = [0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
                 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]

    return pd.DataFrame({
        'date': dates,
        'ticker_price': prices,
        'per_stock_return': dividends,
        'purchase_amt': 0.0,
        'sell_amt': 0.0,
        'total_dividend': 0.0,
        'total_owned': 0.0,
        'total_value': 0.0,
    })


@pytest.fixture
def money_market_history():
    '''Create a price history for a money market fund (price always $1).'''
    dates = pd.date_range(start='2023-01-02', periods=12, freq='W-MON')
    return pd.DataFrame({
        'date': dates,
        'ticker_price': 1.0,
        'per_stock_return': 0.0,
        'purchase_amt': 0.0,
        'sell_amt': 0.0,
        'total_dividend': 0.0,
        'total_owned': 0.0,
        'total_value': 0.0,
    })


@pytest.fixture
def sample_csv_file(tmp_path):
    '''Create a sample Alpha Vantage weekly adjusted CSV file.'''
    csv_content = (
        'timestamp,open,high,low,close,adjusted close,volume,dividend amount\n'
        '2023-03-17,100.50,105.00,98.00,103.00,103.00,1000000,0.50\n'
        '2023-03-10,99.00,102.00,97.00,100.50,100.50,950000,0.00\n'
        '2023-03-03,101.00,104.00,96.00,99.00,99.00,1100000,0.00\n'
        '2023-02-24,98.00,103.00,95.00,101.00,101.00,900000,0.25\n'
        '2023-02-17,97.00,100.00,94.00,98.00,98.00,880000,0.00\n'
    )
    csv_path = tmp_path / 'WeeklyTEST2023-03-17.csv'
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def sample_schedule_hist():
    '''Create a sample schedule history with deposits.'''
    dates = pd.date_range(start='2023-01-02', periods=4, freq='MS')
    return pd.DataFrame({
        'date': dates,
        'activity_type': 'deposit',
        'dollars': 1000.0,
        'source': 'External',
        'to': 'VMFXX',
    })
