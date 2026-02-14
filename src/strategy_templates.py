'''Strategy template definitions, save/load persistence, and simulation runner.

Provides built-in strategy presets, JSON-based save/load for user-defined
strategies, and a one-call function to build and execute a full simulation
from a configuration dict.

By: Claude Code
On: February 2026'''

import json
import os
import logging

import pandas as pd

from src.investment import Investment
from src.money_manager import MoneyManager
from src.schedule import Schedule

logger = logging.getLogger('main_logger')

STRATEGIES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'saved_strategies',
)

# ---------------------------------------------------------------------------
# Built-in strategy templates
# ---------------------------------------------------------------------------

STRATEGY_TEMPLATES = {
    "100% Stock Market (VTI)": {
        "description": "All-in on total US stock market index",
        "allocations": {"VTI": 1.0},
        "buy_strategy": "basic",
        "deposit_freq": "MS",
        "deposit_amount": 1000.0,
        "money_market": "VMFXX",
    },
    "70/30 Stocks/Real Estate (VTI/VNQ)": {
        "description": "70% US stocks, 30% real estate investment trust",
        "allocations": {"VTI": 0.7, "VNQ": 0.3},
        "buy_strategy": "basic",
        "deposit_freq": "MS",
        "deposit_amount": 1000.0,
        "money_market": "VMFXX",
    },
    "50/50 Stocks/Real Estate (VTI/VNQ)": {
        "description": "Equal split between US stocks and real estate",
        "allocations": {"VTI": 0.5, "VNQ": 0.5},
        "buy_strategy": "basic",
        "deposit_freq": "MS",
        "deposit_amount": 1000.0,
        "money_market": "VMFXX",
    },
    "80/20 Stocks/Real Estate (VTI/VNQ)": {
        "description": "Heavy stock tilt with modest real estate exposure",
        "allocations": {"VTI": 0.8, "VNQ": 0.2},
        "buy_strategy": "basic",
        "deposit_freq": "MS",
        "deposit_amount": 1000.0,
        "money_market": "VMFXX",
    },
    "100% Real Estate (VNQ)": {
        "description": "All-in on real estate investment trust",
        "allocations": {"VNQ": 1.0},
        "buy_strategy": "basic",
        "deposit_freq": "MS",
        "deposit_amount": 1000.0,
        "money_market": "VMFXX",
    },
    "S&P 500 Focus (VOO)": {
        "description": "100% S&P 500 index fund",
        "allocations": {"VOO": 1.0},
        "buy_strategy": "basic",
        "deposit_freq": "MS",
        "deposit_amount": 1000.0,
        "money_market": "VMFXX",
    },
    "Three-Fund Portfolio (VTI/VXUS/BND)": {
        "description": "50% US stocks, 30% international, 20% bonds",
        "allocations": {"VTI": 0.5, "VXUS": 0.3, "BND": 0.2},
        "buy_strategy": "basic",
        "deposit_freq": "MS",
        "deposit_amount": 1000.0,
        "money_market": "VMFXX",
    },
    "Dividend Focus (VYM/SCHD)": {
        "description": "Dividend-oriented portfolio split between two ETFs",
        "allocations": {"VYM": 0.5, "SCHD": 0.5},
        "buy_strategy": "basic",
        "deposit_freq": "MS",
        "deposit_amount": 1000.0,
        "money_market": "VMFXX",
    },
}


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def get_compatible_templates(available_tickers):
    '''Return template names whose required tickers are all loaded.

    INPUTS:
    available_tickers - list/set of loaded ticker symbols

    OUTPUTS:
    list of template name strings'''
    available = set(available_tickers)
    compatible = []
    for name, tmpl in STRATEGY_TEMPLATES.items():
        needed = set(tmpl['allocations'].keys()) | {tmpl.get('money_market', 'VMFXX')}
        if needed.issubset(available):
            compatible.append(name)
    return compatible


# ---------------------------------------------------------------------------
# Save / Load persistence
# ---------------------------------------------------------------------------

def save_strategy_config(config, name, directory=None):
    '''Save a strategy configuration dict as a JSON file.

    INPUTS:
    config - strategy configuration dict
    name - human-readable name (used to derive filename)
    directory - save directory (defaults to STRATEGIES_DIR)

    OUTPUTS:
    str - path to the saved file'''
    if directory is None:
        directory = STRATEGIES_DIR
    os.makedirs(directory, exist_ok=True)

    safe_name = name.replace(' ', '_').replace('/', '_').lower()
    filepath = os.path.join(directory, f'{safe_name}.json')

    payload = dict(config)
    payload['name'] = name

    with open(filepath, 'w') as f:
        json.dump(payload, f, indent=2)

    logger.info(f'Saved strategy "{name}" to {filepath}')
    return filepath


def load_strategy_config(filepath):
    '''Load a strategy configuration dict from a JSON file.

    INPUTS:
    filepath - path to the JSON file

    OUTPUTS:
    dict - strategy configuration'''
    with open(filepath, 'r') as f:
        config = json.load(f)
    logger.info(f'Loaded strategy from {filepath}')
    return config


def list_saved_strategies(directory=None):
    '''List all saved strategy configs from a directory.

    INPUTS:
    directory - directory to scan (defaults to STRATEGIES_DIR)

    OUTPUTS:
    dict of {display_name: config_dict}'''
    if directory is None:
        directory = STRATEGIES_DIR
    if not os.path.exists(directory):
        return {}

    result = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(directory, filename)
        try:
            config = load_strategy_config(filepath)
            display = config.get('name', filename[:-5])
            result[display] = config
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f'Skipping {filepath}: {e}')
    return result


def delete_saved_strategy(name, directory=None):
    '''Delete a saved strategy JSON file.

    INPUTS:
    name - the strategy name (used to derive filename)
    directory - directory to look in (defaults to STRATEGIES_DIR)'''
    if directory is None:
        directory = STRATEGIES_DIR
    safe_name = name.replace(' ', '_').replace('/', '_').lower()
    filepath = os.path.join(directory, f'{safe_name}.json')
    if os.path.exists(filepath):
        os.remove(filepath)
        logger.info(f'Deleted strategy "{name}" at {filepath}')


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def build_and_run_strategy(config, investments, start_date, end_date):
    '''Build and execute a complete investment strategy simulation.

    Creates fresh copies of all needed investments, sets up the deposit
    schedule, runs deposits, and executes proportional buys.

    INPUTS:
    config - dict with keys: allocations, buy_strategy, deposit_freq,
             deposit_amount, money_market
    investments - dict of {ticker: Investment} (originals, will be copied)
    start_date - str or date, simulation start
    end_date - str or date, simulation end

    OUTPUTS:
    MoneyManager instance with the fully executed strategy'''
    mm_ticker = config.get('money_market', 'VMFXX')
    allocations = config['allocations']
    deposit_freq = config.get('deposit_freq', 'MS')
    deposit_amount = config.get('deposit_amount', 1000.0)

    needed = set(allocations.keys()) | {mm_ticker}
    missing = needed - set(investments.keys())
    if missing:
        raise ValueError(
            f'Missing tickers: {", ".join(sorted(missing))}. '
            f'Available: {", ".join(sorted(investments.keys()))}'
        )

    # Create fresh Investment copies from original price histories
    fresh = {}
    for ticker in needed:
        orig = investments[ticker]
        ph = orig.history[['date', 'ticker_price', 'per_stock_return']].copy()
        fresh[ticker] = Investment(ticker, price_history=ph, divisible=orig.divisible)

    # Build schedule and portfolio
    sched = Schedule(money_market=mm_ticker)
    sched.automate_investment_schedule(
        start=str(start_date), stop=str(end_date),
        freq=deposit_freq, amount=deposit_amount,
    )

    mm = MoneyManager(schedule=sched, money_market_ticker=mm_ticker)
    for inv in fresh.values():
        mm.add_investment(inv)

    mm.run_schedule()
    _execute_proportional_buys(mm, allocations)

    logger.info(f'Strategy simulation complete: {len(mm.activity_lst)} transactions')
    return mm


def _execute_proportional_buys(mm, allocations):
    '''After deposit schedule is run, buy stocks proportionally from
    the money market on each date that has available funds.

    INPUTS:
    mm - MoneyManager with deposits already executed
    allocations - dict of {ticker: fraction} (fractions should sum to <=1.0)'''
    mm_inv = mm.investments.get(mm.money_market_ticker)
    if mm_inv is None or mm_inv.history.empty:
        return

    for _, row in mm_inv.history.iterrows():
        date = row['date']
        try:
            available = mm_inv.get_funds_available(date=date)
        except (ValueError, KeyError):
            continue
        if available <= 10:
            continue
        for ticker, frac in allocations.items():
            if ticker not in mm.investments:
                continue
            dollars = available * frac
            if dollars > 1:
                try:
                    mm.buy_ticket(dollars=dollars, date=date, ticket=ticker)
                except (ValueError, KeyError):
                    pass
