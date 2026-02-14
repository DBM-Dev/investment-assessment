'''Tests for the strategy_templates module.'''

import pytest
import os
import json
import pandas as pd
import numpy as np

from src.strategy_templates import (
    STRATEGY_TEMPLATES,
    get_compatible_templates,
    save_strategy_config,
    load_strategy_config,
    list_saved_strategies,
    delete_saved_strategy,
    build_and_run_strategy,
)
from src.investment import Investment
from src.money_manager import MoneyManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_investments():
    '''Create a dict of demo investments matching the Quick Demo pattern.'''
    np.random.seed(42)
    weeks = 52
    dates = pd.date_range(start='2022-01-03', periods=weeks, freq='W-MON')

    # Money market
    mm_ph = pd.DataFrame({
        'date': dates, 'ticker_price': 1.0, 'per_stock_return': 0.0,
    })
    vmfxx = Investment('VMFXX', price_history=mm_ph, divisible=True)

    # VTI-like stock
    vti_prices = 200.0 * np.cumprod(1 + np.random.normal(0.002, 0.025, weeks))
    vti_divs = np.zeros(weeks)
    vti_divs[12::13] = 0.75
    vti_ph = pd.DataFrame({
        'date': dates, 'ticker_price': vti_prices, 'per_stock_return': vti_divs,
    })
    vti = Investment('VTI', price_history=vti_ph, divisible=True)

    # VNQ-like REIT
    vnq_prices = 90.0 * np.cumprod(1 + np.random.normal(0.001, 0.03, weeks))
    vnq_divs = np.zeros(weeks)
    vnq_divs[12::13] = 0.55
    vnq_ph = pd.DataFrame({
        'date': dates, 'ticker_price': vnq_prices, 'per_stock_return': vnq_divs,
    })
    vnq = Investment('VNQ', price_history=vnq_ph, divisible=True)

    return {'VMFXX': vmfxx, 'VTI': vti, 'VNQ': vnq}


@pytest.fixture
def sample_config():
    return {
        'name': 'Test Strategy',
        'allocations': {'VTI': 0.7, 'VNQ': 0.3},
        'buy_strategy': 'basic',
        'deposit_freq': 'MS',
        'deposit_amount': 1000.0,
        'money_market': 'VMFXX',
    }


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_templates_exist(self):
        assert len(STRATEGY_TEMPLATES) > 0

    def test_templates_have_required_keys(self):
        for name, tmpl in STRATEGY_TEMPLATES.items():
            assert 'description' in tmpl, f'{name} missing description'
            assert 'allocations' in tmpl, f'{name} missing allocations'
            assert 'money_market' in tmpl, f'{name} missing money_market'
            assert 'deposit_freq' in tmpl, f'{name} missing deposit_freq'
            assert 'deposit_amount' in tmpl, f'{name} missing deposit_amount'

    def test_template_allocations_are_valid(self):
        for name, tmpl in STRATEGY_TEMPLATES.items():
            total = sum(tmpl['allocations'].values())
            assert 0 < total <= 1.0 + 1e-9, f'{name}: allocations sum to {total}'

    def test_get_compatible_templates_all_available(self):
        available = ['VTI', 'VNQ', 'VMFXX']
        compatible = get_compatible_templates(available)
        assert '100% Stock Market (VTI)' in compatible
        assert '70/30 Stocks/Real Estate (VTI/VNQ)' in compatible

    def test_get_compatible_templates_missing_ticker(self):
        available = ['VMFXX']  # VTI and VNQ missing
        compatible = get_compatible_templates(available)
        assert '100% Stock Market (VTI)' not in compatible
        assert '70/30 Stocks/Real Estate (VTI/VNQ)' not in compatible

    def test_get_compatible_templates_empty(self):
        assert get_compatible_templates([]) == []


# ---------------------------------------------------------------------------
# Save / Load tests
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load(self, tmp_path, sample_config):
        filepath = save_strategy_config(sample_config, 'test', directory=str(tmp_path))
        assert os.path.exists(filepath)
        loaded = load_strategy_config(filepath)
        assert loaded['allocations'] == sample_config['allocations']
        assert loaded['deposit_amount'] == sample_config['deposit_amount']
        assert loaded['name'] == 'test'

    def test_list_saved_strategies(self, tmp_path, sample_config):
        save_strategy_config(sample_config, 'alpha', directory=str(tmp_path))
        save_strategy_config(sample_config, 'beta', directory=str(tmp_path))
        result = list_saved_strategies(directory=str(tmp_path))
        assert len(result) == 2
        assert 'alpha' in result
        assert 'beta' in result

    def test_list_saved_strategies_empty_dir(self, tmp_path):
        result = list_saved_strategies(directory=str(tmp_path))
        assert result == {}

    def test_list_saved_strategies_nonexistent_dir(self, tmp_path):
        result = list_saved_strategies(directory=str(tmp_path / 'nonexistent'))
        assert result == {}

    def test_delete_saved_strategy(self, tmp_path, sample_config):
        save_strategy_config(sample_config, 'to_delete', directory=str(tmp_path))
        assert len(list_saved_strategies(directory=str(tmp_path))) == 1
        delete_saved_strategy('to_delete', directory=str(tmp_path))
        assert len(list_saved_strategies(directory=str(tmp_path))) == 0

    def test_save_creates_directory(self, tmp_path, sample_config):
        nested = str(tmp_path / 'a' / 'b' / 'c')
        filepath = save_strategy_config(sample_config, 'nested', directory=nested)
        assert os.path.exists(filepath)


# ---------------------------------------------------------------------------
# Simulation runner tests
# ---------------------------------------------------------------------------

class TestBuildAndRunStrategy:
    def test_basic_run(self, demo_investments, sample_config):
        mm = build_and_run_strategy(
            sample_config, demo_investments,
            start_date='2022-01-03', end_date='2022-12-26',
        )
        assert isinstance(mm, MoneyManager)
        assert len(mm.activity_lst) > 0
        state = mm.return_current_financial_state()
        assert state['total_value'] > 0

    def test_100pct_single_ticker(self, demo_investments):
        config = {
            'allocations': {'VTI': 1.0},
            'deposit_freq': 'MS',
            'deposit_amount': 500.0,
            'money_market': 'VMFXX',
        }
        mm = build_and_run_strategy(
            config, demo_investments,
            start_date='2022-01-03', end_date='2022-06-30',
        )
        state = mm.return_current_financial_state()
        assert state['total_value'] > 0

    def test_missing_ticker_raises(self, demo_investments):
        config = {
            'allocations': {'AAPL': 1.0},
            'deposit_freq': 'MS',
            'deposit_amount': 1000.0,
            'money_market': 'VMFXX',
        }
        with pytest.raises(ValueError, match='Missing tickers'):
            build_and_run_strategy(
                config, demo_investments,
                start_date='2022-01-03', end_date='2022-12-26',
            )

    def test_original_investments_not_modified(self, demo_investments, sample_config):
        # Record original state
        original_vti_purchases = demo_investments['VTI'].history['purchase_amt'].sum()

        build_and_run_strategy(
            sample_config, demo_investments,
            start_date='2022-01-03', end_date='2022-12-26',
        )

        # Original should be unchanged
        assert demo_investments['VTI'].history['purchase_amt'].sum() == original_vti_purchases

    def test_different_configs_produce_different_results(self, demo_investments):
        config_a = {
            'allocations': {'VTI': 1.0},
            'deposit_freq': 'MS',
            'deposit_amount': 1000.0,
            'money_market': 'VMFXX',
        }
        config_b = {
            'allocations': {'VNQ': 1.0},
            'deposit_freq': 'MS',
            'deposit_amount': 1000.0,
            'money_market': 'VMFXX',
        }

        mm_a = build_and_run_strategy(
            config_a, demo_investments,
            start_date='2022-01-03', end_date='2022-12-26',
        )
        mm_b = build_and_run_strategy(
            config_b, demo_investments,
            start_date='2022-01-03', end_date='2022-12-26',
        )

        state_a = mm_a.return_current_financial_state()
        state_b = mm_b.return_current_financial_state()

        # Both should have value, but they should differ
        assert state_a['total_value'] > 0
        assert state_b['total_value'] > 0
        assert abs(state_a['total_value'] - state_b['total_value']) > 1.0
