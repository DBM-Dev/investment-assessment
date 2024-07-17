# High Level

This application is intended to review past market performance and provide guidance on what would have happened had different investment decisions been made. The goal is to provide a simple way to trade off different strategies of investments. Towards that extent, the main goal of the app is to track the amount of money that would likely have come from various investments. The three primary types of investments that are tracked here are:

1. Specific stock tickers (e.g. VTI, CRSP, VMFXX)
2. CDs of specified duration, payout, and schedule (e.g. March 1, 2022 - March 2023, 5.3%, monthly)
3. General interest bearing accounts (e.g. High Yield Savings) with specified interest rate schedules - THIS IS FUTURE STATE, NOT PLANNED FOR INITIAL APP

There are two main ways the app will track money.

First, via a manually entered schedule of payments. All money going into the accounts and past transactions must be mapped for what actually happened. The application will estimate how much money is in the stocks and CDs as a result of that money growing. This should be fairly close to the actual dollars in the accounts. Then, the application will treat each money investment as a separate entry, and allow different pre-defined strategies to be implemented in terms of how much money would go to which stocks and CDs. This allows what if-ing to examine how different investment strategies with the same amount of money put in at the same time would have played out.

Second, via a regularly scheduled payment scheme. I.e. say "Invest 1k each month starting in January 2020" and see what different investment strategies would have led to. This is to assume that similar investment opportunities will play out in the future, and examine what the potential values could be.

Finally, after different schedules have been assessed, a summary of the financial implications will be written out, including the amount of moeny made via interest and dividends (income taxable), the amount of money in stocks, and the total amount of money grown.

# Modules

## ticker_collection.py
This module collects the data required from Alpha Vantage to get weekly stock market information. We will use the average between high and low for the week to determine the stock price at any given time

## money_manager.py
This module contains the class that manages the money. Attributes are:
investments - a dictionary of Investment objects
schedule - a list of investment activities to make
activity_lst - a list of all transactions (various function calls) performed
and methods are:
return_current_financial_state - a function that returns a summary of the financial state
transfer_cash - a function that moves money from outside the investment to VMFXX. Parameters are time, amount, and into
buy_ticket - a function that moves money from VMFXX to a specific ticket. Parameters are quantity, date, and ticket
sell_ticket - a function that moves money from a specific ticket to VMFXX. Parameters are quantity, date and ticket

## investment.py
This module contains the class that contains each individual investment. Attributes are:
ticker - a string of the ticker symbol
history - a dataframe with fields of date, ticker price, per_stock_return, purchase_amt, sell_amt, total_divident, total_owned, total_value
divisible - a boolean indicating if the ticket can only be purchased for in integer amounts or not. Defaults to True
and methods are:
calc_amt_from_ticket - a function that calculates the amount of dollars for a number of tickets. Parameters are number and the output is dollars
calc_ticket_from_amt - a function that calculates the amount of stock from a dollar amount put in. Parameters are dollars and output is number
recalc_investment - a function that recalculates the total_dividend, total_owned, and total_value columns of the history dataframe given the investments made
buy_investment - a function that adds to the purchase_amt field of the history dataframe and runs recalc_investment. Parameters are dollars, date, and req_integer
sell_investment - a function that adds to the sell_amt field of the history dataframe and runs recalc_investment. Parameters are qty, date, and req_integer

## schedule.py
This module contains the Schedule class, which provides the information on what the schedule of purchases, sales, and investments will be for a particular investment strategy. Attributes are:
history - a dataframe of date, activity_type, from, to, amount. Where activity_type can be one of '