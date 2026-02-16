'''Unit tests for the MoneyManager class.'''

import pytest
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from money_manager import MoneyManager
from investment import Investment
from schedule import Schedule


@pytest.fixture
def portfolio(sample_price_history, money_market_history):
    '''Create a MoneyManager with a money market and one stock investment.'''
    mm = MoneyManager()
    vmfxx = Investment('VMFXX', price_history=money_market_history)
    vti = Investment('VTI', price_history=sample_price_history)
    mm.add_investment(vmfxx)
    mm.add_investment(vti)
    return mm


class TestMoneyManagerInit:
    def test_init_defaults(self):
        mm = MoneyManager()
        assert mm.investments == {}
        assert mm.schedules == []
        assert mm.activity_lst == []
        assert mm.money_market_ticker == 'VMFXX'

    def test_add_investment(self, sample_price_history):
        mm = MoneyManager()
        inv = Investment('VTI', price_history=sample_price_history)
        mm.add_investment(inv)
        assert 'VTI' in mm.investments
        assert mm.investments['VTI'] is inv


class TestTransferCash:
    def test_transfer_cash(self, portfolio):
        date = portfolio.investments['VMFXX'].history['date'].iloc[0]
        shares, dollars = portfolio.transfer_cash(time=date, amount=5000.0)
        assert shares > 0
        assert dollars > 0
        assert len(portfolio.activity_lst) == 1
        assert portfolio.activity_lst[0]['type'] == 'transfer'

    def test_transfer_cash_missing_investment(self):
        mm = MoneyManager()
        with pytest.raises(ValueError, match='not found'):
            mm.transfer_cash(time='2023-01-01', amount=1000.0)


class TestBuyTicket:
    def test_buy_ticket_by_dollars(self, portfolio):
        date = portfolio.investments['VMFXX'].history['date'].iloc[0]
        # First fund the money market
        portfolio.transfer_cash(time=date, amount=5000.0)
        # Then buy stock
        shares, dollars = portfolio.buy_ticket(
            dollars=1000.0, date=date, ticket='VTI'
        )
        assert shares > 0
        assert dollars > 0
        assert len(portfolio.activity_lst) == 2

    def test_buy_ticket_missing_ticket(self, portfolio):
        with pytest.raises(ValueError, match='not found'):
            portfolio.buy_ticket(dollars=100.0, date='2023-01-02',
                                 ticket='FAKE')

    def test_buy_ticket_no_date(self, portfolio):
        with pytest.raises(ValueError, match='date parameter'):
            portfolio.buy_ticket(dollars=100.0, ticket='VTI')


class TestSellTicket:
    def test_sell_ticket(self, portfolio):
        date = portfolio.investments['VMFXX'].history['date'].iloc[0]
        portfolio.transfer_cash(time=date, amount=5000.0)
        portfolio.buy_ticket(dollars=1000.0, date=date, ticket='VTI')
        shares_sold, dollars_received = portfolio.sell_ticket(
            quantity=2, date=date, ticket='VTI'
        )
        assert shares_sold > 0
        assert dollars_received > 0

    def test_sell_ticket_missing_ticket(self, portfolio):
        with pytest.raises(ValueError, match='not found'):
            portfolio.sell_ticket(quantity=1, date='2023-01-02', ticket='FAKE')


class TestFinancialState:
    def test_empty_state(self):
        mm = MoneyManager()
        state = mm.return_current_financial_state()
        assert state['total_value'] == 0.0
        assert state['cash'] == 0.0
        assert state['holdings'] == {}

    def test_state_with_holdings(self, portfolio):
        date = portfolio.investments['VMFXX'].history['date'].iloc[0]
        portfolio.transfer_cash(time=date, amount=5000.0)
        portfolio.buy_ticket(dollars=1000.0, date=date, ticket='VTI')
        state = portfolio.return_current_financial_state(date=date)
        assert 'VMFXX' in state['holdings']
        assert 'VTI' in state['holdings']
        assert state['total_value'] > 0


class TestRunSchedule:
    def test_run_schedule(self, portfolio):
        sched = Schedule()
        sched.automate_investment_schedule(
            start='2023-01-02',
            stop='2023-03-01',
            freq='MS',
            amount=1000.0,
        )
        portfolio.schedules = [sched]
        portfolio.run_schedule()
        assert len(portfolio.activity_lst) > 0

    def test_run_multiple_schedules(self, portfolio):
        sched1 = Schedule()
        sched1.automate_investment_schedule(
            start='2023-01-02',
            stop='2023-02-01',
            freq='MS',
            amount=500.0,
        )
        sched2 = Schedule()
        sched2.automate_investment_schedule(
            start='2023-02-06',
            stop='2023-03-01',
            freq='MS',
            amount=750.0,
        )
        portfolio.schedules = [sched1, sched2]
        portfolio.run_schedule()
        assert len(portfolio.activity_lst) > 0

    def test_run_schedule_none(self, portfolio):
        with pytest.raises(ValueError, match='No schedule'):
            portfolio.run_schedule()
