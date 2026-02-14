'''Unit tests for the Schedule class.'''

import pytest
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from schedule import Schedule


class TestScheduleInit:
    def test_init_defaults(self):
        s = Schedule()
        assert s.hist.empty
        assert s.money_market == 'VMFXX'
        assert s.minimum_amt == 0

    def test_init_with_history(self, sample_schedule_hist):
        s = Schedule(hist=sample_schedule_hist)
        assert len(s.hist) == 4

    def test_init_custom_money_market(self):
        s = Schedule(money_market='SPAXX')
        assert s.money_market == 'SPAXX'


class TestAutomateInvestmentSchedule:
    def test_basic_schedule(self):
        s = Schedule()
        calendar = s.automate_investment_schedule(
            start='2023-01-01',
            stop='2023-06-01',
            freq='MS',
            amount=1000.0,
        )
        assert len(calendar) > 0
        assert all(calendar['activity_type'] == 'deposit')
        assert all(calendar['dollars'] == 1000.0)
        assert all(calendar['to'] == 'VMFXX')

    def test_replace_hist_true(self):
        s = Schedule()
        s.automate_investment_schedule(
            start='2023-01-01',
            stop='2023-06-01',
            freq='MS',
            amount=1000.0,
            replace_hist=True,
        )
        assert not s.hist.empty
        assert all(s.hist['activity_type'] == 'deposit')

    def test_replace_hist_false_appends(self, sample_schedule_hist):
        s = Schedule(hist=sample_schedule_hist)
        initial_len = len(s.hist)
        s.automate_investment_schedule(
            start='2023-06-01',
            stop='2023-09-01',
            freq='MS',
            amount=500.0,
            replace_hist=False,
        )
        assert len(s.hist) > initial_len

    def test_stop_defaults_to_today(self):
        s = Schedule()
        calendar = s.automate_investment_schedule(
            start='2023-01-01',
            freq='MS',
            amount=1000.0,
        )
        assert len(calendar) > 0


class TestAddTransaction:
    def test_add_to_self_hist(self):
        s = Schedule()
        s.automate_investment_schedule(
            start='2023-01-01',
            stop='2023-03-01',
            freq='MS',
            amount=1000.0,
        )
        initial_len = len(s.hist)
        s.add_transaction(
            source='VMFXX',
            to='VTI',
            activity_type='buy/sell',
            date='2023-01-15',
            amount=500.0,
        )
        assert len(s.hist) == initial_len + 1

    def test_add_to_external_hist(self):
        s = Schedule()
        external_hist = pd.DataFrame({
            'date': ['2023-01-01'],
            'activity_type': ['deposit'],
            'dollars': [1000.0],
            'source': ['External'],
            'to': ['VMFXX'],
        })
        result = s.add_transaction(
            source='VMFXX',
            to='VTI',
            activity_type='buy/sell',
            date='2023-01-15',
            amount=500.0,
            hist=external_hist,
        )
        assert len(result) == 2
        # self.hist should remain empty
        assert s.hist.empty

    def test_activity_type_is_passed_through(self):
        s = Schedule()
        s.add_transaction(
            source='VMFXX',
            to='VTI',
            activity_type='transfer',
            date='2023-01-15',
            amount=500.0,
        )
        assert s.hist.iloc[-1]['activity_type'] == 'transfer'

    def test_source_column_used(self):
        s = Schedule()
        s.add_transaction(
            source='VMFXX',
            to='VTI',
            activity_type='buy/sell',
            date='2023-01-15',
            amount=500.0,
        )
        assert 'source' in s.hist.columns
        assert s.hist.iloc[-1]['source'] == 'VMFXX'
