'''Integration tests for end-to-end investment scenarios.'''

import pytest
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from investment import Investment
from money_manager import MoneyManager
from schedule import Schedule


@pytest.fixture
def full_price_history():
    '''Create a longer price history for integration testing.'''
    dates = pd.date_range(start='2023-01-02', periods=52, freq='W-MON')
    # Simulate a stock that generally trends upward with some volatility
    import numpy as np
    np.random.seed(42)
    base_prices = 100 + np.cumsum(np.random.randn(52) * 2)
    base_prices = np.maximum(base_prices, 50)  # Floor at 50

    dividends = [0.0] * 52
    # Quarterly dividends
    for i in [12, 25, 38, 51]:
        if i < 52:
            dividends[i] = 0.50

    return pd.DataFrame({
        'date': dates,
        'ticker_price': base_prices,
        'per_stock_return': dividends,
        'purchase_amt': 0.0,
        'sell_amt': 0.0,
        'total_dividend': 0.0,
        'total_owned': 0.0,
        'total_value': 0.0,
    })


@pytest.fixture
def full_mm_history():
    '''Money market price history for a full year.'''
    dates = pd.date_range(start='2023-01-02', periods=52, freq='W-MON')
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


class TestEndToEnd:
    def test_schedule_deposit_and_buy(self, full_price_history, full_mm_history):
        '''Create a schedule, load prices, run auto_buy_basic, verify state.'''
        # Setup investments
        mm = MoneyManager()
        vmfxx = Investment('VMFXX', price_history=full_mm_history)
        vti = Investment('VTI', price_history=full_price_history)
        mm.add_investment(vmfxx)
        mm.add_investment(vti)

        # Create schedule: monthly deposits
        sched = Schedule()
        sched.automate_investment_schedule(
            start='2023-01-02',
            stop='2023-12-31',
            freq='MS',
            amount=1000.0,
        )

        # Execute deposit schedule
        mm.schedule = sched
        mm.run_schedule()

        # Verify deposits were made
        state = mm.return_current_financial_state()
        assert state['cash'] > 0
        assert len(mm.activity_lst) > 0

    def test_transfer_buy_sell_cycle(self, full_price_history, full_mm_history):
        '''Test a complete transfer -> buy -> sell -> state check cycle.'''
        mm = MoneyManager()
        vmfxx = Investment('VMFXX', price_history=full_mm_history)
        vti = Investment('VTI', price_history=full_price_history)
        mm.add_investment(vmfxx)
        mm.add_investment(vti)

        date = full_mm_history['date'].iloc[10]

        # Transfer cash in
        mm.transfer_cash(time=date, amount=10000.0)
        state = mm.return_current_financial_state(date=date)
        assert state['cash'] > 0

        # Buy stock
        shares, cost = mm.buy_ticket(dollars=5000.0, date=date, ticket='VTI')
        assert shares > 0

        state = mm.return_current_financial_state(date=date)
        assert 'VTI' in state['holdings']
        assert state['holdings']['VTI']['shares'] > 0

        # Sell stock
        shares_sold, proceeds = mm.sell_ticket(quantity=1, date=date, ticket='VTI')
        assert shares_sold > 0
        assert proceeds > 0

    def test_multiple_investments_comparison(self, full_mm_history):
        '''Test managing multiple different stock investments.'''
        import numpy as np
        np.random.seed(123)

        # Create two different stock price histories
        dates = pd.date_range(start='2023-01-02', periods=52, freq='W-MON')

        stock_a_prices = 50 + np.cumsum(np.random.randn(52) * 1.5)
        stock_a_prices = np.maximum(stock_a_prices, 20)

        stock_b_prices = 200 + np.cumsum(np.random.randn(52) * 3)
        stock_b_prices = np.maximum(stock_b_prices, 100)

        hist_a = pd.DataFrame({
            'date': dates,
            'ticker_price': stock_a_prices,
            'per_stock_return': 0.0,
            'purchase_amt': 0.0, 'sell_amt': 0.0,
            'total_dividend': 0.0, 'total_owned': 0.0, 'total_value': 0.0,
        })
        hist_b = pd.DataFrame({
            'date': dates,
            'ticker_price': stock_b_prices,
            'per_stock_return': 0.0,
            'purchase_amt': 0.0, 'sell_amt': 0.0,
            'total_dividend': 0.0, 'total_owned': 0.0, 'total_value': 0.0,
        })

        mm = MoneyManager()
        mm.add_investment(Investment('VMFXX', price_history=full_mm_history))
        mm.add_investment(Investment('STOCK_A', price_history=hist_a))
        mm.add_investment(Investment('STOCK_B', price_history=hist_b))

        date = dates[5]
        mm.transfer_cash(time=date, amount=10000.0)
        mm.buy_ticket(dollars=3000.0, date=date, ticket='STOCK_A')
        mm.buy_ticket(dollars=3000.0, date=date, ticket='STOCK_B')

        state = mm.return_current_financial_state(date=date)
        assert 'STOCK_A' in state['holdings']
        assert 'STOCK_B' in state['holdings']
        assert state['total_value'] > 0
