'''Unit tests for ticker_collection.py.

These tests use mock data and do not require live API calls.'''

import pytest
import pandas as pd
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ticker_collection import parse_csv, compute_avg_price, get_api_key


class TestParseCSV:
    def test_parse_csv_basic(self, sample_csv_file):
        df = parse_csv(sample_csv_file)
        assert len(df) == 5
        assert 'date' in df.columns
        assert 'avg_price' in df.columns
        assert 'dividend_amount' in df.columns

    def test_parse_csv_sorted_by_date(self, sample_csv_file):
        df = parse_csv(sample_csv_file)
        dates = df['date'].tolist()
        assert dates == sorted(dates)

    def test_parse_csv_avg_price(self, sample_csv_file):
        df = parse_csv(sample_csv_file)
        # First row after sorting (2023-02-17): high=100, low=94, avg=97
        assert df.iloc[0]['avg_price'] == pytest.approx(97.0)

    def test_parse_csv_dividend(self, sample_csv_file):
        df = parse_csv(sample_csv_file)
        # Row for 2023-03-17 has dividend=0.50
        row = df.loc[df['date'] == pd.Timestamp('2023-03-17')]
        assert row.iloc[0]['dividend_amount'] == pytest.approx(0.50)


class TestComputeAvgPrice:
    def test_compute_avg_price(self):
        df = pd.DataFrame({
            'high': [100.0, 110.0, 105.0],
            'low': [90.0, 100.0, 95.0],
        })
        result = compute_avg_price(df)
        assert result.tolist() == [95.0, 105.0, 100.0]


class TestGetApiKey:
    def test_get_api_key_missing(self, monkeypatch):
        monkeypatch.delenv('ALPHA_VANTAGE_API_KEY', raising=False)
        with pytest.raises(ValueError, match='ALPHA_VANTAGE_API_KEY'):
            get_api_key()

    def test_get_api_key_present(self, monkeypatch):
        monkeypatch.setenv('ALPHA_VANTAGE_API_KEY', 'test_key_123')
        assert get_api_key() == 'test_key_123'
