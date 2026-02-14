'''This module contains the class that represents an individual investment
holding (stock ticker or CD).

By: David Martin
On: July 17, 2024'''

import pandas as pd
import numpy as np
import logging
import math

logger = logging.getLogger('main_logger')


class Investment(object):
    '''Represents a single investment holding and its transaction history.

    ATTRIBUTES:

    ticker - string identifier for the investment (e.g. "VTI")
    history - DataFrame tracking: date, ticker_price, per_stock_return,
              purchase_amt, sell_amt, total_dividend, total_owned, total_value
    divisible - boolean, whether fractional shares are allowed (default True)

    METHODS:

    calc_amt_from_ticket - convert share count to dollar value
    calc_ticket_from_amt - convert dollar amount to share count
    recalc_investment - recalculate derived columns from transaction history
    buy_investment - record a purchase
    sell_investment - record a sale
    get_funds_available - return withdrawable balance at a point in time
    load_price_history - load price data from a CSV file'''

    def __init__(self, ticker, price_history=None, divisible=True):
        '''Initialize an Investment.

        INPUTS:
        ticker - string ticker symbol (e.g. "VTI")
        price_history - optional DataFrame with columns: date, ticker_price,
                        per_stock_return. If None, an empty history is created.
        divisible - boolean, whether fractional shares are allowed'''
        self.ticker = ticker
        self.divisible = divisible

        if price_history is not None:
            self.history = price_history.copy()
            # Ensure required columns exist with defaults
            for col in ['purchase_amt', 'sell_amt', 'total_dividend',
                        'total_owned', 'total_value']:
                if col not in self.history.columns:
                    self.history[col] = 0.0
        else:
            self.history = pd.DataFrame(columns=[
                'date', 'ticker_price', 'per_stock_return',
                'purchase_amt', 'sell_amt', 'total_dividend',
                'total_owned', 'total_value'
            ])

        logger.info(f'Investment initialized for {self.ticker} with '
                     f'{len(self.history)} price records, '
                     f'divisible={self.divisible}')

    @classmethod
    def from_csv(cls, ticker, csv_path, divisible=True):
        '''Create an Investment by loading price history from a CSV file
        produced by ticker_collection.py (Alpha Vantage weekly adjusted).

        The CSV is expected to have columns: timestamp, open, high, low, close,
        adjusted close, volume, dividend amount.

        The representative price is the average of weekly high and low per the
        design document.

        INPUTS:
        ticker - string ticker symbol
        csv_path - path to the CSV file
        divisible - whether fractional shares are allowed

        OUTPUTS:
        Investment instance with loaded price history'''
        raw = pd.read_csv(csv_path)
        raw.columns = [c.strip().lower().replace(' ', '_') for c in raw.columns]

        price_history = pd.DataFrame()
        price_history['date'] = pd.to_datetime(raw['timestamp'])
        price_history['ticker_price'] = (raw['high'] + raw['low']) / 2.0
        if 'dividend_amount' in raw.columns:
            price_history['per_stock_return'] = raw['dividend_amount']
        else:
            price_history['per_stock_return'] = 0.0

        price_history = price_history.sort_values('date').reset_index(drop=True)

        logger.info(f'Loaded {len(price_history)} price records for {ticker} '
                     f'from {csv_path}')
        return cls(ticker, price_history=price_history, divisible=divisible)

    def _get_price(self, date):
        '''Get the ticker price on or before a given date.

        INPUTS:
        date - the target date

        OUTPUTS:
        float - the ticker price'''
        date = pd.Timestamp(date)
        available = self.history.loc[self.history['date'] <= date]
        if available.empty:
            raise ValueError(f'No price data available for {self.ticker} '
                             f'on or before {date}')
        return available.iloc[-1]['ticker_price']

    def _get_row_index(self, date):
        '''Get the index of the history row on or before a given date.

        INPUTS:
        date - the target date

        OUTPUTS:
        int - index into self.history'''
        date = pd.Timestamp(date)
        available = self.history.loc[self.history['date'] <= date]
        if available.empty:
            raise ValueError(f'No price data available for {self.ticker} '
                             f'on or before {date}')
        return available.index[-1]

    def calc_amt_from_ticket(self, number, date=None):
        '''Convert a share count to a dollar value at current price.

        INPUTS:
        number - number of shares
        date - optional date for price lookup (defaults to latest)

        OUTPUTS:
        float - dollar value'''
        if date is None:
            price = self.history.iloc[-1]['ticker_price']
        else:
            price = self._get_price(date)
        return number * price

    def calc_ticket_from_amt(self, dollars, date=None):
        '''Convert a dollar amount to a share count at current price.

        INPUTS:
        dollars - dollar amount
        date - optional date for price lookup (defaults to latest)

        OUTPUTS:
        float - number of shares (may be fractional if divisible=True)'''
        if date is None:
            price = self.history.iloc[-1]['ticker_price']
        else:
            price = self._get_price(date)

        if price == 0:
            return 0.0

        shares = dollars / price
        if not self.divisible:
            shares = math.floor(shares)
        return shares

    def recalc_investment(self):
        '''Recalculate total_dividend, total_owned, and total_value columns
        from the purchase/sell history and per-stock returns.

        This walks through the history in date order, accumulating shares
        bought/sold and dividends earned.'''
        if self.history.empty:
            return

        self.history = self.history.sort_values('date').reset_index(drop=True)

        cumulative_shares = 0.0
        cumulative_dividends = 0.0

        for idx in self.history.index:
            row = self.history.loc[idx]
            price = row['ticker_price']

            # Shares bought this period
            if price > 0 and row['purchase_amt'] > 0:
                bought = row['purchase_amt'] / price
                if not self.divisible:
                    bought = math.floor(bought)
            else:
                bought = 0.0

            # Shares sold this period
            if price > 0 and row['sell_amt'] > 0:
                sold = row['sell_amt'] / price
                if not self.divisible:
                    sold = math.floor(sold)
            else:
                sold = 0.0

            cumulative_shares += bought - sold

            # Dividends earned this period
            dividend = row['per_stock_return'] * cumulative_shares
            cumulative_dividends += dividend

            self.history.at[idx, 'total_dividend'] = cumulative_dividends
            self.history.at[idx, 'total_owned'] = cumulative_shares
            self.history.at[idx, 'total_value'] = cumulative_shares * price

        logger.debug(f'Recalculated {self.ticker}: {cumulative_shares} shares, '
                      f'${cumulative_dividends:.2f} total dividends')

    def buy_investment(self, dollars, date, req_integer=False, from_acct=None):
        '''Record a purchase of this investment.

        INPUTS:
        dollars - dollar amount to invest
        date - date of purchase
        req_integer - if True, only buy whole shares
        from_acct - optional Investment object representing the funding source

        OUTPUTS:
        tuple of (shares_purchased, dollars_spent)'''
        price = self._get_price(date)
        if price <= 0:
            logger.warning(f'Cannot buy {self.ticker} at price {price}')
            return (0, 0.0)

        shares = dollars / price
        if req_integer or not self.divisible:
            shares = math.floor(shares)

        if shares <= 0:
            logger.debug(f'Insufficient funds to buy any shares of '
                         f'{self.ticker} at ${price:.2f}')
            return (0, 0.0)

        dollars_spent = shares * price

        # Record the purchase in history
        idx = self._get_row_index(date)
        self.history.at[idx, 'purchase_amt'] = (
            self.history.at[idx, 'purchase_amt'] + dollars_spent
        )

        # Deduct from funding source if provided
        if from_acct is not None:
            from_acct.sell_investment(qty=shares, date=date,
                                     req_integer=False,
                                     dollar_amount=dollars_spent)

        self.recalc_investment()

        logger.info(f'Bought {shares} shares of {self.ticker} for '
                     f'${dollars_spent:.2f} on {date}')
        return (shares, dollars_spent)

    def sell_investment(self, qty=None, date=None, req_integer=False,
                        dollar_amount=None):
        '''Record a sale of this investment.

        INPUTS:
        qty - number of shares to sell (mutually exclusive with dollar_amount)
        date - date of sale
        req_integer - if True, only sell whole shares
        dollar_amount - dollar amount to sell (alternative to qty)

        OUTPUTS:
        tuple of (shares_sold, dollars_received)'''
        price = self._get_price(date)
        if price <= 0:
            logger.warning(f'Cannot sell {self.ticker} at price {price}')
            return (0, 0.0)

        idx = self._get_row_index(date)
        current_owned = self.history.at[idx, 'total_owned']

        if dollar_amount is not None:
            shares = dollar_amount / price
        elif qty is not None:
            shares = qty
        else:
            shares = current_owned

        if req_integer or not self.divisible:
            shares = math.floor(shares)

        shares = min(shares, current_owned)

        if shares <= 0:
            return (0, 0.0)

        dollars_received = shares * price

        self.history.at[idx, 'sell_amt'] = (
            self.history.at[idx, 'sell_amt'] + dollars_received
        )

        self.recalc_investment()

        logger.info(f'Sold {shares} shares of {self.ticker} for '
                     f'${dollars_received:.2f} on {date}')
        return (shares, dollars_received)

    def get_funds_available(self, date):
        '''Return the dollar value available for withdrawal at a given date,
        accounting for future committed withdrawals.

        INPUTS:
        date - the date to check availability

        OUTPUTS:
        float - available dollar amount'''
        date = pd.Timestamp(date)
        idx = self._get_row_index(date)
        current_value = self.history.at[idx, 'total_value']

        # Account for future committed sell transactions
        future = self.history.loc[self.history['date'] > date]
        future_sells = future['sell_amt'].sum()

        available = current_value - future_sells
        return max(0.0, available)

    def get_funds(self, date):
        '''Return the total dollar value of this investment at a given date.

        INPUTS:
        date - the date to check

        OUTPUTS:
        float - total dollar value'''
        idx = self._get_row_index(date)
        return self.history.at[idx, 'total_value']


class CDInvestment(Investment):
    '''Represents a Certificate of Deposit investment.

    CDs have a fixed term, interest rate, and payout schedule.
    They cannot be partially sold (divisible=False) and early
    withdrawal incurs a penalty.'''

    def __init__(self, ticker, start_date, end_date, rate, payout_freq='MS',
                 principal=0.0, early_withdrawal_penalty_months=3):
        '''Initialize a CD investment.

        INPUTS:
        ticker - identifier string (e.g. "CD-5.3%-2022")
        start_date - date the CD begins
        end_date - date the CD matures
        rate - annual interest rate as a decimal (e.g. 0.053 for 5.3%)
        payout_freq - pandas frequency string for interest payouts (default monthly)
        principal - initial deposit amount
        early_withdrawal_penalty_months - months of interest forfeited on early withdrawal'''
        self.rate = rate
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.payout_freq = payout_freq
        self.principal = principal
        self.early_withdrawal_penalty_months = early_withdrawal_penalty_months

        # Build the price/interest history for the CD
        dates = pd.date_range(start=self.start_date, end=self.end_date,
                              freq=payout_freq)
        if dates.empty:
            dates = pd.DatetimeIndex([self.start_date, self.end_date])

        # For a CD, ticker_price is always 1.0 (dollar-denominated)
        # per_stock_return represents the interest earned per period
        num_periods = len(dates)
        if num_periods > 0:
            days_total = (self.end_date - self.start_date).days
            period_rate = rate * (days_total / 365.0) / num_periods
        else:
            period_rate = 0.0

        price_history = pd.DataFrame({
            'date': dates,
            'ticker_price': 1.0,
            'per_stock_return': period_rate,
            'purchase_amt': 0.0,
            'sell_amt': 0.0,
            'total_dividend': 0.0,
            'total_owned': 0.0,
            'total_value': 0.0,
        })

        super().__init__(ticker, price_history=price_history, divisible=False)

        # Record the initial principal if provided
        if principal > 0:
            self.buy_investment(dollars=principal, date=start_date)

        logger.info(f'CD initialized: {ticker}, rate={rate}, '
                     f'{start_date} to {end_date}, '
                     f'principal=${principal:.2f}')

    def get_early_withdrawal_penalty(self, date):
        '''Calculate the penalty for early withdrawal at a given date.

        INPUTS:
        date - the proposed withdrawal date

        OUTPUTS:
        float - dollar amount of penalty'''
        date = pd.Timestamp(date)
        if date >= self.end_date:
            return 0.0

        # Penalty is N months of interest
        monthly_interest = self.principal * self.rate / 12.0
        penalty = monthly_interest * self.early_withdrawal_penalty_months
        return penalty
