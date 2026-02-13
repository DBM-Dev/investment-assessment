# Next Steps

This document describes the remaining work needed to complete the Investment Assessment application. Items are organized by priority and grouped by category.

---

## 1. Fix Bugs in Existing Code

The current `src/schedule.py` has several issues that must be resolved before any new development can proceed.

### 1.1 Syntax and Runtime Errors

- **`logging.get_logger` does not exist** (line 11): Replace with `logging.getLogger`.
- **`pd.Dataframe` is miscapitalized** (line 28): Replace with `pd.DataFrame`.
- **`from` used as a parameter name** (lines 84, 86, 110, 126, 127): `from` is a Python reserved keyword. Rename to `source` or `from_acct` throughout the `add_transaction` method signature and all call sites.
- **Undefined variable `end`** in `automate_investment_schedule` (line 50): The `stop` parameter is accepted but never used; `end` is referenced instead. Replace `end` with `stop`.
- **Typo `avaialble_funds` vs `available_funds`** (lines 76-77): Variable is assigned as `avaialble_funds` but read as `available_funds`. Use a single consistent name.
- **Undefined variable `date_deposit`** (line 91): Should be `deposit_date` to match the loop variable.
- **Wrong attribute `self.money_manager`** (line 86): Should be `self.money_market` (a string), not `self.money_manager` (which does not exist on Schedule).

### 1.2 Logic Issues

- `auto_buy_basic` copies `self.hist` into `hst` but never writes back to `self.hist` when `update_hist=True`. Add the write-back logic.
- `add_transaction` hard-codes `activity_type` to `'buy/sell'` in the new DataFrame instead of using the `activity_type` parameter.
- `automate_investment_schedule` does not append to `self.hist` when `replace_hist=False`; it only returns the new calendar. It should concatenate with the existing history.

---

## 2. Implement Core Modules

### 2.1 `src/investment.py` — Individual Investment Tracking

This module is fully designed in `design.MD` but has no implementation. It must contain an `Investment` class with:

**Attributes:**
- `ticker` — string identifier (e.g. `"VTI"`)
- `history` — DataFrame with columns: `date`, `ticker_price`, `per_stock_return`, `purchase_amt`, `sell_amt`, `total_dividend`, `total_owned`, `total_value`
- `divisible` — boolean, whether fractional shares are allowed (default `True`)

**Methods:**
- `calc_amt_from_ticket(number)` — convert a share count to a dollar value at current price
- `calc_ticket_from_amt(dollars)` — convert a dollar amount to a share count at current price
- `recalc_investment()` — recompute `total_dividend`, `total_owned`, and `total_value` columns from purchase/sell history
- `buy_investment(dollars, date, req_integer, from_acct)` — record a purchase, return `(shares, dollars_spent)`
- `sell_investment(qty, date, req_integer)` — record a sale
- `get_funds_available(date)` — return withdrawable balance at a point in time, accounting for future committed withdrawals

**Additional considerations:**
- Load price history from the CSV files produced by `ticker_collection.py`
- Use the average of weekly high and low as the representative price (per `design.MD`)
- Handle dividend data from the Alpha Vantage adjusted weekly series
- Support CD investments with fixed term, rate, and payout schedule

### 2.2 `src/money_manager.py` — Portfolio Management

This module is fully designed in `design.MD` but has no implementation. It must contain a `MoneyManager` class with:

**Attributes:**
- `investments` — dictionary of `Investment` objects keyed by ticker
- `schedule` — a `Schedule` instance defining the investment plan
- `activity_lst` — ordered list of all executed transactions

**Methods:**
- `return_current_financial_state()` — return a summary of all holdings, cash, and total portfolio value
- `transfer_cash(time, amount, into)` — move money from external source into VMFXX (or specified account)
- `buy_ticket(quantity, date, ticket)` — move money from VMFXX into a stock position
- `sell_ticket(quantity, date, ticket)` — liquidate a stock position back into VMFXX

### 2.3 Complete `Schedule.auto_buy_balanced`

This method (line 95-108 in `schedule.py`) is currently a stub. It should:

- After each deposit, assess the current portfolio allocation
- Compare actual allocation to the target allocation defined in `ticket_lookup`
- Purchase stocks in the order that brings the portfolio closest to the target balance
- Respect integer-share constraints and minimum purchase amounts

---

## 3. Data Pipeline

### 3.1 Improve `ticker_collection.py`

- Add error handling for API failures (HTTP errors, rate limits, malformed responses)
- Externalize the API key configuration (environment variable or config file instead of `_apikey.py` import)
- Add a function to parse the downloaded CSVs into pandas DataFrames with standardized column names
- Add a function to compute the average of weekly high/low as the representative price
- Add support for incremental updates (only fetch new data, append to existing files)

### 3.2 CD and Fixed-Income Data

- Implement a data source or manual entry mechanism for CD rates and terms
- Model CD maturity, early withdrawal penalties, and interest accrual schedules

---

## 4. Reporting and Output

### 4.1 Financial Summary

Per the design document, after schedules are assessed the application should produce a summary report including:

- Total money invested (principal)
- Total money earned via dividends (income-taxable)
- Total money earned via interest from CDs (income-taxable)
- Total unrealized capital gains/losses per holding
- Total portfolio value
- Comparison across multiple strategies side by side

### 4.2 Strategy Comparison

- Accept multiple `Schedule` configurations and run each against the same price history
- Output a comparison table or chart showing how each strategy performed over the same time period

---

## 5. Testing

No tests currently exist. A test suite should cover:

### 5.1 Unit Tests

- `Investment` — price lookups, buy/sell math, dividend calculations, `get_funds_available` correctness
- `Schedule` — deposit generation, transaction recording, auto-buy logic
- `MoneyManager` — transfers, portfolio state queries, multi-investment coordination
- `ticker_collection` — CSV parsing, price averaging (mock the API)

### 5.2 Integration Tests

- End-to-end scenario: create a schedule, load historical prices, run `auto_buy_basic`, verify final portfolio state
- Balanced vs. basic strategy comparison on identical inputs

### 5.3 Infrastructure

- Add `pytest` as the test runner
- Add a `tests/` directory with fixtures for sample price data and schedules
- Add mock data files so tests do not require live API calls

---

## 6. Project Infrastructure

### 6.1 Packaging and Dependencies

- Create a `requirements.txt` listing: `pandas`, `requests`, `pytest`
- Consider adding a `pyproject.toml` or `setup.py` for installable packaging
- Specify the minimum Python version (3.8+ recommended for f-string and DataFrame features used)

### 6.2 Configuration

- Add a `.env.example` or `config.example.yaml` documenting the required API key
- Remove the hard-coded `_apikey` import in favor of `os.environ.get("ALPHA_VANTAGE_API_KEY")`

### 6.3 Documentation

- Expand `README.md` with setup instructions, usage examples, and module descriptions
- Add docstrings with type hints to all public methods

---

## 7. Future Enhancements (Post-MVP)

These items are noted in the design but explicitly deferred:

- **Interest-bearing accounts** (e.g. high-yield savings) with variable rate schedules
- **CLI or web interface** for interactive strategy configuration and report viewing
- **Visualization** — charts of portfolio growth over time, allocation pie charts
- **Tax modeling** — estimate tax liability from dividends, interest, and realized gains
- **Rebalancing simulation** — periodic rebalancing (quarterly/annually) instead of only at deposit time

---

## Suggested Implementation Order

| Phase | Work Items | Depends On |
|-------|-----------|------------|
| **Phase 1** | Fix all bugs in `schedule.py` | — |
| **Phase 2** | Implement `investment.py` | Phase 1 |
| **Phase 3** | Implement `money_manager.py` | Phase 2 |
| **Phase 4** | Complete `auto_buy_balanced` | Phase 3 |
| **Phase 5** | Improve `ticker_collection.py`, add config | — (parallel) |
| **Phase 6** | Add test suite | Phases 1-4 |
| **Phase 7** | Implement reporting and strategy comparison | Phases 3-4 |
| **Phase 8** | Project infrastructure (`requirements.txt`, docs) | — (parallel) |
