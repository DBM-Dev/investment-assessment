'''Unit tests for the Investment class.'''

import pytest
import pandas as pd
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from investment import Investment, CDInvestment, SavingsAccount


class TestInvestmentInit:
    def test_init_empty(self):
        inv = Investment('VTI')
        assert inv.ticker == 'VTI'
        assert inv.divisible is True
        assert inv.history.empty

    def test_init_with_price_history(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        assert len(inv.history) == 12
        assert 'purchase_amt' in inv.history.columns
        assert 'total_value' in inv.history.columns

    def test_init_divisible_false(self):
        inv = Investment('VTI', divisible=False)
        assert inv.divisible is False


class TestPriceLookup:
    def test_get_price_exact_date(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        date = sample_price_history['date'].iloc[0]
        price = inv._get_price(date)
        assert price == 100.0

    def test_get_price_between_dates(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        # Use a date between first and second data points
        date = sample_price_history['date'].iloc[0] + pd.Timedelta(days=3)
        price = inv._get_price(date)
        assert price == 100.0  # Should return the most recent price on or before

    def test_get_price_no_data(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        with pytest.raises(ValueError, match='No price data'):
            inv._get_price('2020-01-01')


class TestCalcConversions:
    def test_calc_amt_from_ticket(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        # Latest price is 118.0
        assert inv.calc_amt_from_ticket(10) == 1180.0

    def test_calc_amt_from_ticket_with_date(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        date = sample_price_history['date'].iloc[0]
        assert inv.calc_amt_from_ticket(5, date=date) == 500.0

    def test_calc_ticket_from_amt_divisible(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        # Latest price is 118.0
        shares = inv.calc_ticket_from_amt(500.0)
        assert abs(shares - 500.0 / 118.0) < 0.001

    def test_calc_ticket_from_amt_not_divisible(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history,
                         divisible=False)
        shares = inv.calc_ticket_from_amt(500.0)
        assert shares == math.floor(500.0 / 118.0)
        assert isinstance(shares, int)


class TestBuyInvestment:
    def test_buy_investment_basic(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        date = sample_price_history['date'].iloc[5]  # price=107.0
        shares, dollars = inv.buy_investment(dollars=500.0, date=date)
        assert shares > 0
        assert dollars > 0
        assert dollars <= 500.0

    def test_buy_investment_integer_shares(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        date = sample_price_history['date'].iloc[5]
        shares, dollars = inv.buy_investment(dollars=500.0, date=date,
                                              req_integer=True)
        assert shares == math.floor(500.0 / 107.0)
        assert dollars == shares * 107.0

    def test_buy_investment_insufficient_funds(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history,
                         divisible=False)
        date = sample_price_history['date'].iloc[5]
        shares, dollars = inv.buy_investment(dollars=5.0, date=date,
                                              req_integer=True)
        assert shares == 0
        assert dollars == 0.0

    def test_buy_updates_history(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        date = sample_price_history['date'].iloc[5]
        inv.buy_investment(dollars=1000.0, date=date)
        idx = inv._get_row_index(date)
        assert inv.history.at[idx, 'purchase_amt'] > 0
        assert inv.history.at[idx, 'total_owned'] > 0


class TestSellInvestment:
    def test_sell_investment(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        date = sample_price_history['date'].iloc[5]
        # Buy first
        inv.buy_investment(dollars=1000.0, date=date)
        # Then sell
        shares_sold, dollars_received = inv.sell_investment(qty=2, date=date)
        assert shares_sold > 0
        assert dollars_received > 0

    def test_sell_by_dollar_amount(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        date = sample_price_history['date'].iloc[5]
        inv.buy_investment(dollars=1000.0, date=date)
        shares_sold, dollars_received = inv.sell_investment(
            date=date, dollar_amount=200.0
        )
        assert shares_sold > 0
        assert dollars_received > 0


class TestRecalcInvestment:
    def test_recalc_empty(self):
        inv = Investment('VTI')
        inv.recalc_investment()  # Should not raise

    def test_recalc_after_purchase(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        date = sample_price_history['date'].iloc[3]
        inv.buy_investment(dollars=1050.0, date=date)
        inv.recalc_investment()
        idx = inv._get_row_index(date)
        assert inv.history.at[idx, 'total_owned'] > 0
        assert inv.history.at[idx, 'total_value'] > 0


class TestGetFundsAvailable:
    def test_funds_available_no_holdings(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        date = sample_price_history['date'].iloc[5]
        available = inv.get_funds_available(date=date)
        assert available == 0.0

    def test_funds_available_with_holdings(self, sample_price_history):
        inv = Investment('VTI', price_history=sample_price_history)
        date = sample_price_history['date'].iloc[5]
        inv.buy_investment(dollars=1000.0, date=date)
        available = inv.get_funds_available(date=date)
        assert available > 0


class TestCDInvestment:
    def test_cd_init(self):
        cd = CDInvestment(
            ticker='CD-5pct-2023',
            start_date='2023-01-01',
            end_date='2024-01-01',
            rate=0.05,
            principal=10000.0,
        )
        assert cd.ticker == 'CD-5pct-2023'
        assert cd.rate == 0.05
        assert cd.divisible is False
        assert not cd.history.empty

    def test_cd_early_withdrawal_penalty(self):
        cd = CDInvestment(
            ticker='CD-5pct-2023',
            start_date='2023-01-01',
            end_date='2024-01-01',
            rate=0.05,
            principal=10000.0,
            early_withdrawal_penalty_months=3,
        )
        penalty = cd.get_early_withdrawal_penalty('2023-06-01')
        expected = 10000.0 * 0.05 / 12.0 * 3
        assert abs(penalty - expected) < 0.01

    def test_cd_no_penalty_after_maturity(self):
        cd = CDInvestment(
            ticker='CD-5pct-2023',
            start_date='2023-01-01',
            end_date='2024-01-01',
            rate=0.05,
            principal=10000.0,
        )
        penalty = cd.get_early_withdrawal_penalty('2024-06-01')
        assert penalty == 0.0


class TestFromCSV:
    def test_from_csv(self, sample_csv_file):
        inv = Investment.from_csv('TEST', sample_csv_file)
        assert inv.ticker == 'TEST'
        assert len(inv.history) == 5
        # Avg price of first row (sorted by date): (100+94)/2 = 97.0
        assert inv.history.iloc[0]['ticker_price'] == pytest.approx(97.0)
        assert inv.divisible is True


class TestSavingsAccount:
    def test_init_basic(self):
        sa = SavingsAccount(
            ticker='HYSA-4.5%',
            apy=0.045,
            start_date='2023-01-01',
            end_date='2024-01-01',
        )
        assert sa.ticker == 'HYSA-4.5%'
        assert sa.apy == 0.045
        assert sa.divisible is True
        assert not sa.history.empty
        # All prices should be 1.0
        assert (sa.history['ticker_price'] == 1.0).all()

    def test_init_with_initial_deposit(self):
        sa = SavingsAccount(
            ticker='HYSA',
            apy=0.05,
            start_date='2023-01-01',
            end_date='2024-01-01',
            initial_deposit=10000.0,
        )
        # The first row should have a purchase recorded
        assert sa.history.iloc[0]['purchase_amt'] == 10000.0
        # After recalc, we should own shares
        assert sa.history.iloc[0]['total_owned'] > 0

    def test_compound_interest_over_one_year(self):
        """Verify that a 5% APY compounded monthly yields ~5% after one year."""
        sa = SavingsAccount(
            ticker='HYSA',
            apy=0.05,
            start_date='2023-01-01',
            end_date='2024-01-01',
            payout_freq='MS',
            initial_deposit=10000.0,
        )
        final_value = sa.history.iloc[-1]['total_value']
        # Should be close to $10,500 (5% APY on $10,000)
        assert final_value == pytest.approx(10500.0, rel=0.01)

    def test_interest_accrues_each_period(self):
        sa = SavingsAccount(
            ticker='HYSA',
            apy=0.05,
            start_date='2023-01-01',
            end_date='2024-01-01',
            payout_freq='MS',
            initial_deposit=10000.0,
        )
        # total_dividend (interest) should increase over time
        dividends = sa.history['total_dividend'].tolist()
        for i in range(1, len(dividends)):
            assert dividends[i] >= dividends[i - 1]

    def test_deposit_mid_period(self):
        sa = SavingsAccount(
            ticker='HYSA',
            apy=0.05,
            start_date='2023-01-01',
            end_date='2024-01-01',
            payout_freq='MS',
            initial_deposit=5000.0,
        )
        # Add another deposit mid-year
        sa.buy_investment(dollars=5000.0, date='2023-07-01')
        # Final value should reflect both deposits plus interest
        final_value = sa.history.iloc[-1]['total_value']
        # Must be more than $10,000 (principal) due to interest
        assert final_value > 10000.0

    def test_withdrawal(self):
        sa = SavingsAccount(
            ticker='HYSA',
            apy=0.05,
            start_date='2023-01-01',
            end_date='2024-01-01',
            payout_freq='MS',
            initial_deposit=10000.0,
        )
        # Withdraw $3000 mid-year
        shares_sold, dollars_received = sa.sell_investment(
            date='2023-07-01', dollar_amount=3000.0
        )
        assert shares_sold > 0
        assert dollars_received == pytest.approx(3000.0, rel=0.01)
        # Final balance should be less than the no-withdrawal case
        final_value = sa.history.iloc[-1]['total_value']
        assert final_value < 10500.0
        assert final_value > 7000.0

    def test_multiple_deposits_and_withdrawals(self):
        sa = SavingsAccount(
            ticker='HYSA',
            apy=0.05,
            start_date='2023-01-01',
            end_date='2024-01-01',
            payout_freq='MS',
            initial_deposit=1000.0,
        )
        # Monthly deposits of $500
        for month in range(2, 7):
            date = f'2023-{month:02d}-01'
            sa.buy_investment(dollars=500.0, date=date)
        # One withdrawal
        sa.sell_investment(date='2023-08-01', dollar_amount=1000.0)

        final_value = sa.history.iloc[-1]['total_value']
        total_deposited = 1000.0 + 500.0 * 5  # $3,500
        total_withdrawn = 1000.0
        net_deposited = total_deposited - total_withdrawn  # $2,500
        # Should be more than net deposited due to interest
        assert final_value > net_deposited

    def test_zero_apy(self):
        sa = SavingsAccount(
            ticker='HYSA-0%',
            apy=0.0,
            start_date='2023-01-01',
            end_date='2024-01-01',
            initial_deposit=10000.0,
        )
        final_value = sa.history.iloc[-1]['total_value']
        # No interest, should equal deposit
        assert final_value == pytest.approx(10000.0)

    def test_divisible_fractional_deposits(self):
        """Savings accounts should allow any dollar amount."""
        sa = SavingsAccount(
            ticker='HYSA',
            apy=0.04,
            start_date='2023-01-01',
            end_date='2024-01-01',
            initial_deposit=0.0,
        )
        shares, dollars = sa.buy_investment(dollars=123.45, date='2023-01-01')
        assert dollars == pytest.approx(123.45)
        assert shares == pytest.approx(123.45)  # price is 1.0

    def test_get_funds_available(self):
        sa = SavingsAccount(
            ticker='HYSA',
            apy=0.05,
            start_date='2023-01-01',
            end_date='2024-01-01',
            initial_deposit=10000.0,
        )
        available = sa.get_funds_available('2023-06-01')
        assert available > 0

    def test_get_funds(self):
        sa = SavingsAccount(
            ticker='HYSA',
            apy=0.05,
            start_date='2023-01-01',
            end_date='2024-01-01',
            initial_deposit=10000.0,
        )
        value = sa.get_funds('2023-06-01')
        # Should have grown from initial deposit
        assert value > 10000.0
