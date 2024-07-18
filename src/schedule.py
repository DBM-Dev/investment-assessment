'''This module contains the class, methods, functions, and attributes of the
schedule. This handles storing a schedule of investments

By: David Martin
On: July 17, 2024'''

import pandas as pd
import logging
from datetime import datetime

logger = logging.get_logger('main_logger')

class Schedule(object):
    '''The Schedule class contains information and methods related to the 
    schedule of investments made.
    
    ATTRIBUTES:

    hist - pandas dataframe with fields of date, activity_type, from, to, and dollars

    METHODS:

    automate_investment_schedule - This method automates adding investment values on a regular schedule
    auto_buy_basic - This method automates purchasing investments, with each purchase going splitting to follow the investment strategy defined
    auto_buy_balanced - this method automates purchasing investments, aiming to balance the portfolio with each purchase
    load_history - this method loads a history.csv file in the format of the history dataframe
    add_transaction - this method adds a single transaction with the values specified'''
    def __init__(self, hist=pd.Dataframe(), money_market='VMFXX', minimum_amt=0):
        '''Initialize the class'''
        self.hist = hist
        self.money_market = money_market
        self.minimum_amt = minimum_amt
        logger.info(f'Schedule class initialized with history of length {len(self.hist)} with money_market {self.money_market}')

    def automate_investment_schedule(self, start, freq, amount, replace_hist=True, stop=None):
        '''This function automates adding investment values, but not purchasing stock or CDs, on a regular schedule

        INPUTS:
        start - date - the start date to have investment deposits
        stop - date. Defaults to the current date
        freq - int - the frequency to have investment deposits in days
        amount - float - the amount to invest in dollars
        replace_hist - a boolean indicating if the self.hist attribute should be overwritten or not

        OUTPUTS:
        hist - the updated history

        ACTIONS:
        updates self.hist if replace_hist = True'''
        calendar = pd.DataFrame({'date':pd.date_range(start=start, end=end, freq=freq), 'activity_type': 'deposit', 'dollars':amount, 'from':'External', 'to':self.money_market})
        logger.debug(f'Created calendar of shape {calendar.shape}, with a total deposit of {calendar["dollars"].sum()}')
        if replace_hist == True:
            self.hist = calendar
            logger.debug(f'Replaced self.hist with new calendar')
        return calendar

    def auto_buy_basic(self, money_manager, ticket_lookup, update_hist=True):
        '''This method automates purchasing investments. Effectively, it will ensure that there are purchases immediately following each investment that will use the dollars in the money_market greater than the minimum_amount to purchase tickets according to the ticket lookup

        INPUTS:
        money_manager - MoneyManager object, used to identify how much money is avaialble in the money market, and what the prices of the tickets are
        ticket_lookup - dataframe with fiels of tickets to purchase, and values are percentages (or relative numbers) of how much of each to buy in dollars
        update_hist - boolean, used to indicate if the self.hist should be updated or if the new history should be returned

        OUTPUTS:
        hist - the updated history, now with purchases

        ACTIONS:
        updates self.hist if update_hist = True'''
        logger.info(f'Attempting to auto buy {len(ticket_lookup.keys())} tickets after each deposit')
        #Clear out old purchases
        hst = self.hist.copy()
        # Loop through each deposit_date
        for deposit_date in hst.loc[hst['activity_type']=='deposit']['date'].unique():
            logger.debug(f'Attempting to purchase after deposits on {deposit_date}')
            avaialble_funds = money_manager.investments[self.money_market].get_funds_available(date=deposit_date)
            logger.debug(f'{available_funds} are available to purchase')
            denominator = ticket_lookup['value'].sum()
            ticket_lookup['purchase_dollars'] = ticket_lookup['value'] / denominator * available_funds
            tot_dollars = 0
            for ticket in ticket_lookup.keys():
                purchase_dollars = ticket_lookup.loc[ticket,'purchase_dollars']
                logger.debug(f'Attempting to purchase {purchase_dollars} dollars of {ticket}')
                stock, dollars = money_manager.investments[ticket].buy_investment(dollars=purchase_dollars, date=deposit_date, req_integer=True, from=money_manager.investments[self.money_market])
                if stock > 0:
                    self.add_transaction(from=self.money_manager, to=ticket, activity_type='buy/sell', date=deposit_date, amount=dollars)
                    logger.debug(f'Purchased {stock} stocks of {ticket} for a total of ${dollars}')
                    tot_dollars += dollars
                else:
                    logger.debug(f'Did not purchase any {stock}, insufficient funds available')
            logger.debug(f'Finished purchased ${tot_dollars} of stock. ${money_manager.investments[self.money_market].get_funds(date=deposit_date)} remain in {self.money_market} on {date_deposit}, ${money_manager.investments[self.money_market].get_funds_available(date=deposit_date)} available to purchase')
        logger.info(f'Finished auto buying from deposits')
        return hst

    def auto_buy_balanced(self, money_manager, ticket_lookup, update_hist=True):
        '''This method automates purchasing investments. Effectively, it will ensure that there are purchases immediately following each investment that will use the dollars in the money_market greater than the minimum_amount to purchase tickets according to the ticket lookup, purchasing stocks in a way to best balance the desired ticket portfolio

        INPUTS:
        money_manager - MoneyManager object, used to identify how much money is avaialble in the money market, and what the prices of the tickets are
        ticket_lookup - dataframe with fiels of tickets to purchase, and values are percentages (or relative numbers) of how much of each to buy in dollars
        update_hist - boolean, used to indicate if the self.hist should be updated or if the new history should be returned

        OUTPUTS:
        hist - the updated history, now with purchases

        ACTIONS:
        updates self.hist if update_hist = True'''
        pass

    def add_transaction(self, from, to, activity_type, date, amount, hist=None):
        '''This module adds transactions to the history. If no history is provided, then it adds them to the self.hist

        INPUTS:
        from - stock money is coming from
        to - stock money is going to
        activity_type - activity type
        date - date of transaction
        amount - dollars of transaction
        hist - history to add transaction to. Defaults to self.hist

        OUTPUTS:
        hist - history of transactions

        ACTIONS:
        updates self.hist if hist=None'''
        logger.info(f'Adding a transfer of ${amount} from {from} to {to} on {date}')
        new_transaction = pd.DataFrame({'date':[date], 'activity_type':'buy/sell', 'dollars':amount,'from':from,'to':to})
        if hist is None:
            self.hist = pd.concat([self.hist,new_transaction])
            hist = self.hist
        else:
            hist = pd.concat([hist,new_transaction])
        logger.info('Returning updated history')
        return hist

