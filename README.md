# Investment Assessment

## Purpose

This app is intended to demonstrate what would have happened with a variety of investment strategies applied to historical stock market data. It lets you simulate regular deposits, allocate across stocks, CDs, and money market accounts, then compare strategy outcomes side-by-side.

## Prerequisites

- Python 3.8+
- A free [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key) for fetching stock market data

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd investment-assessment

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and add your Alpha Vantage API key:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_api_key_here` with your actual key:

```
ALPHA_VANTAGE_API_KEY=your_actual_key
```

## Usage

### 1. Fetch Stock Data

Download weekly adjusted price data from Alpha Vantage:

```python
from src.ticker_collection import main, parse_csv, update_csv

# Download data for default tickers (VTI, VMFXX, etc.)
paths = main(write_loc='./', wait_between=15)

# Parse a downloaded CSV into a DataFrame
df = parse_csv('WeeklyVTI2024-01-15.csv')

# Incrementally update an existing CSV with the latest data
updated_path = update_csv('WeeklyVTI2024-01-15.csv', 'VTI', api_key)
```

### 2. Create Investments

```python
from src.investment import Investment, CDInvestment

# Load a stock investment from a CSV file
vti = Investment.from_csv('VTI', 'WeeklyVTI2024-01-15.csv')

# Create a CD investment
cd = CDInvestment(
    'CD-5.3%-2022',
    start_date='2022-03-01',
    end_date='2023-03-01',
    rate=0.053,
    principal=10000.0
)
```

### 3. Set Up an Investment Schedule

Define a recurring deposit schedule and an automatic buying strategy:

```python
from src.schedule import Schedule

schedule = Schedule(money_market='VMFXX')

# Deposit $1,000 monthly from Jan 2020 through Dec 2023
schedule.automate_investment_schedule(
    start='2020-01-01',
    stop='2023-12-31',
    freq='MS',
    amount=1000.0
)

# Auto-buy proportionally: 70% VTI, 30% VNQ
schedule.auto_buy_basic(
    money_manager=mm,
    ticket_lookup={'VTI': 0.7, 'VNQ': 0.3}
)

# Or use balanced buying to maintain target allocation
schedule.auto_buy_balanced(
    money_manager=mm,
    ticket_lookup={'VTI': 0.6, 'VOO': 0.4}
)
```

### 4. Manage a Portfolio

```python
from src.money_manager import MoneyManager

mm = MoneyManager(schedule=schedule, money_market_ticker='VMFXX')

# Add investments to the portfolio
mm.add_investment(vmfxx)
mm.add_investment(vti)

# Execute all scheduled transactions
mm.run_schedule()

# Get the portfolio state at a specific date
state = mm.return_current_financial_state(date='2023-12-31')
```

### 5. Generate Reports

```python
from src.reporting import financial_summary, format_summary, compare_strategies, format_comparison

# Single-strategy summary
summary = financial_summary(mm, date='2023-12-31')
print(format_summary(summary))

# Compare multiple strategies
comparison = compare_strategies([
    {'name': '70/30 VTI/VNQ', 'money_manager': mm1},
    {'name': '60/40 VTI/VOO', 'money_manager': mm2},
], date='2023-12-31')
print(format_comparison(comparison))
```

## Running Tests

```bash
# Run the full test suite
pytest tests/

# Run a specific test file with verbose output
pytest tests/test_investment.py -v

# Run with coverage reporting
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
src/
  ticker_collection.py  # Fetch and parse Alpha Vantage market data
  investment.py         # Individual investment tracking (stocks, CDs)
  money_manager.py      # Portfolio management and transaction execution
  schedule.py           # Deposit schedules and auto-buy strategies
  reporting.py          # Financial summaries and strategy comparisons
tests/
  conftest.py           # Shared pytest fixtures
  test_investment.py
  test_money_manager.py
  test_schedule.py
  test_ticker_collection.py
  test_integration.py
```

## License

See [LICENSE.md](LICENSE.md) for details.
