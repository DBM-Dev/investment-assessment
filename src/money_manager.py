'''This module contains the MoneyManager class that manages the overall
investment portfolio, coordinating multiple Investment objects and
executing transactions.

By: David Martin
On: July 17, 2024'''

import pandas as pd
import logging

logger = logging.getLogger('main_logger')


class MoneyManager(object):
    '''Manages the overall investment portfolio.

    ATTRIBUTES:

    investments - dictionary of Investment objects keyed by ticker
    schedule - a Schedule instance defining the investment plan
    activity_lst - ordered list of all executed transactions

    METHODS:

    return_current_financial_state - return a summary of all holdings
    transfer_cash - move money from external source into money market
    buy_ticket - move money from money market into a stock position
    sell_ticket - liquidate a stock position back into money market'''

    def __init__(self, schedule=None, money_market_ticker='VMFXX'):
        '''Initialize the MoneyManager.

        INPUTS:
        schedule - optional Schedule instance
        money_market_ticker - ticker symbol for the default money market fund'''
        self.investments = {}
        self.schedule = schedule
        self.activity_lst = []
        self.money_market_ticker = money_market_ticker
        logger.info(f'MoneyManager initialized with money market '
                     f'{self.money_market_ticker}')

    def add_investment(self, investment):
        '''Add an Investment object to the portfolio.

        INPUTS:
        investment - an Investment instance'''
        self.investments[investment.ticker] = investment
        logger.info(f'Added investment {investment.ticker} to portfolio')

    def return_current_financial_state(self, date=None):
        '''Return a summary of all holdings, cash, and total portfolio value.

        INPUTS:
        date - optional date for the snapshot (defaults to latest available)

        OUTPUTS:
        dict with keys: holdings (dict of ticker -> value), cash, total_value,
        total_invested, total_dividends'''
        state = {
            'holdings': {},
            'cash': 0.0,
            'total_value': 0.0,
            'total_invested': 0.0,
            'total_dividends': 0.0,
        }

        for ticker, inv in self.investments.items():
            if inv.history.empty:
                continue

            try:
                if date is not None:
                    idx = inv._get_row_index(date)
                else:
                    idx = inv.history.index[-1]

                row = inv.history.loc[idx]
                value = row['total_value']
                invested = inv.history.loc[:idx, 'purchase_amt'].sum()
                sold = inv.history.loc[:idx, 'sell_amt'].sum()
                dividends = row['total_dividend']

                state['holdings'][ticker] = {
                    'shares': row['total_owned'],
                    'price': row['ticker_price'],
                    'value': value,
                    'invested': invested,
                    'sold': sold,
                    'dividends': dividends,
                }

                if ticker == self.money_market_ticker:
                    state['cash'] += value
                else:
                    state['total_value'] += value

                state['total_invested'] += invested
                state['total_dividends'] += dividends

            except (ValueError, KeyError):
                logger.debug(f'No data for {ticker} at date {date}')
                continue

        # Total value includes cash
        state['total_value'] += state['cash']

        logger.info(f'Portfolio state: ${state["total_value"]:.2f} total, '
                     f'${state["cash"]:.2f} cash, '
                     f'{len(state["holdings"])} holdings')
        return state

    def transfer_cash(self, time, amount, into=None):
        '''Move money from an external source into the money market fund
        (or a specified account).

        INPUTS:
        time - date of the transfer
        amount - dollar amount to transfer
        into - optional ticker to transfer into (defaults to money market)

        OUTPUTS:
        tuple of (shares, dollars) from the buy'''
        if into is None:
            into = self.money_market_ticker

        if into not in self.investments:
            raise ValueError(f'Investment {into} not found in portfolio. '
                             f'Add it first with add_investment().')

        shares, dollars = self.investments[into].buy_investment(
            dollars=amount, date=time
        )

        self.activity_lst.append({
            'date': time,
            'type': 'transfer',
            'source': 'External',
            'destination': into,
            'dollars': dollars,
            'shares': shares,
        })

        logger.info(f'Transferred ${dollars:.2f} into {into} on {time}')
        return (shares, dollars)

    def buy_ticket(self, quantity=None, date=None, ticket=None, dollars=None):
        '''Move money from the money market fund into a stock position.

        Either quantity (shares) or dollars must be specified.

        INPUTS:
        quantity - number of shares to buy (optional)
        date - date of the purchase
        ticket - ticker symbol to buy
        dollars - dollar amount to invest (optional, alternative to quantity)

        OUTPUTS:
        tuple of (shares_purchased, dollars_spent)'''
        if ticket is None:
            raise ValueError('ticket parameter is required')
        if ticket not in self.investments:
            raise ValueError(f'Investment {ticket} not found in portfolio')
        if date is None:
            raise ValueError('date parameter is required')

        inv = self.investments[ticket]
        mm = self.investments.get(self.money_market_ticker)

        # Determine dollar amount
        if dollars is None and quantity is not None:
            dollars = inv.calc_amt_from_ticket(quantity, date=date)
        elif dollars is None:
            raise ValueError('Either quantity or dollars must be specified')

        # Check money market funds
        if mm is not None:
            available = mm.get_funds_available(date=date)
            if dollars > available:
                logger.warning(f'Requested ${dollars:.2f} but only '
                               f'${available:.2f} available in '
                               f'{self.money_market_ticker}')
                dollars = available

        shares, dollars_spent = inv.buy_investment(
            dollars=dollars, date=date, from_acct=mm
        )

        self.activity_lst.append({
            'date': date,
            'type': 'buy',
            'source': self.money_market_ticker,
            'destination': ticket,
            'dollars': dollars_spent,
            'shares': shares,
        })

        logger.info(f'Bought {shares} of {ticket} for ${dollars_spent:.2f} '
                     f'on {date}')
        return (shares, dollars_spent)

    def sell_ticket(self, quantity=None, date=None, ticket=None, dollars=None):
        '''Liquidate a stock position back into the money market fund.

        Either quantity (shares) or dollars must be specified.

        INPUTS:
        quantity - number of shares to sell (optional)
        date - date of the sale
        ticket - ticker symbol to sell
        dollars - dollar amount to sell (optional, alternative to quantity)

        OUTPUTS:
        tuple of (shares_sold, dollars_received)'''
        if ticket is None:
            raise ValueError('ticket parameter is required')
        if ticket not in self.investments:
            raise ValueError(f'Investment {ticket} not found in portfolio')
        if date is None:
            raise ValueError('date parameter is required')

        inv = self.investments[ticket]
        mm = self.investments.get(self.money_market_ticker)

        shares_sold, dollars_received = inv.sell_investment(
            qty=quantity, date=date, dollar_amount=dollars
        )

        # Transfer proceeds into money market
        if mm is not None and dollars_received > 0:
            mm.buy_investment(dollars=dollars_received, date=date)

        self.activity_lst.append({
            'date': date,
            'type': 'sell',
            'source': ticket,
            'destination': self.money_market_ticker,
            'dollars': dollars_received,
            'shares': shares_sold,
        })

        logger.info(f'Sold {shares_sold} of {ticket} for '
                     f'${dollars_received:.2f} on {date}')
        return (shares_sold, dollars_received)

    def run_schedule(self):
        '''Execute all transactions defined in self.schedule against
        the current portfolio.

        Processes deposits (transfer_cash) and buy/sell transactions
        in chronological order.'''
        if self.schedule is None:
            raise ValueError('No schedule has been set')

        if self.schedule.hist.empty:
            logger.warning('Schedule history is empty, nothing to execute')
            return

        hist = self.schedule.hist.sort_values('date').reset_index(drop=True)

        for _, row in hist.iterrows():
            date = row['date']
            activity = row['activity_type']
            dollars = row['dollars']
            source = row.get('source', 'External')
            to = row.get('to', self.money_market_ticker)

            if activity == 'deposit':
                self.transfer_cash(time=date, amount=dollars, into=to)
            elif activity == 'buy/sell':
                if source == self.money_market_ticker or source == 'External':
                    self.buy_ticket(dollars=dollars, date=date, ticket=to)
                else:
                    self.sell_ticket(dollars=dollars, date=date, ticket=source)

        logger.info(f'Finished executing schedule: '
                     f'{len(hist)} transactions processed')
