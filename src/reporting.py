'''This module provides financial summary reports and strategy comparison
functionality for the investment assessment application.

By: David Martin
On: July 17, 2024'''

import pandas as pd
import logging

logger = logging.getLogger('main_logger')


def financial_summary(money_manager, date=None):
    '''Generate a financial summary report for a portfolio.

    Per the design document, the summary includes:
    - Total money invested (principal)
    - Total money earned via dividends (income-taxable)
    - Total money earned via interest from CDs (income-taxable)
    - Total unrealized capital gains/losses per holding
    - Total portfolio value

    INPUTS:
    money_manager - a MoneyManager instance
    date - optional date for the snapshot (defaults to latest)

    OUTPUTS:
    dict with the summary data'''
    state = money_manager.return_current_financial_state(date=date)

    total_principal = 0.0
    total_dividends = 0.0
    total_cd_interest = 0.0
    total_savings_interest = 0.0
    total_unrealized_gains = 0.0
    holdings_detail = []

    for ticker, info in state['holdings'].items():
        invested = info['invested']
        sold = info['sold']
        value = info['value']
        dividends = info['dividends']
        shares = info['shares']
        price = info['price']

        net_invested = invested - sold
        unrealized_gain = value - net_invested

        # Classify interest/dividends by investment type
        inv = money_manager.investments.get(ticker)
        is_cd = hasattr(inv, 'rate') and hasattr(inv, 'end_date')
        is_savings = hasattr(inv, 'apy')

        if is_cd:
            total_cd_interest += dividends
            inv_type = 'CD'
        elif is_savings:
            total_savings_interest += dividends
            inv_type = 'Savings'
        else:
            total_dividends += dividends
            inv_type = 'Stock'

        total_principal += net_invested
        total_unrealized_gains += unrealized_gain

        holdings_detail.append({
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'value': value,
            'invested': invested,
            'sold': sold,
            'net_invested': net_invested,
            'unrealized_gain': unrealized_gain,
            'dividends': dividends,
            'type': inv_type,
        })

    summary = {
        'date': date,
        'total_principal': total_principal,
        'total_dividends': total_dividends,
        'total_cd_interest': total_cd_interest,
        'total_savings_interest': total_savings_interest,
        'total_income_taxable': total_dividends + total_cd_interest + total_savings_interest,
        'total_unrealized_gains': total_unrealized_gains,
        'total_portfolio_value': state['total_value'],
        'cash': state['cash'],
        'holdings': holdings_detail,
    }

    logger.info(f'Financial summary: ${summary["total_portfolio_value"]:.2f} total, '
                 f'${summary["total_principal"]:.2f} invested, '
                 f'${summary["total_unrealized_gains"]:.2f} unrealized gains')
    return summary


def format_summary(summary):
    '''Format a financial summary dict into a human-readable string report.

    INPUTS:
    summary - dict from financial_summary()

    OUTPUTS:
    str - formatted report'''
    lines = []
    lines.append('=' * 60)
    lines.append('INVESTMENT PORTFOLIO SUMMARY')
    if summary['date']:
        lines.append(f'As of: {summary["date"]}')
    lines.append('=' * 60)
    lines.append('')

    lines.append(f'Total Portfolio Value:     ${summary["total_portfolio_value"]:>12,.2f}')
    lines.append(f'Cash (Money Market):       ${summary["cash"]:>12,.2f}')
    lines.append(f'Total Principal Invested:  ${summary["total_principal"]:>12,.2f}')
    lines.append(f'Total Unrealized Gains:    ${summary["total_unrealized_gains"]:>12,.2f}')
    lines.append('')

    lines.append('--- Income (Taxable) ---')
    lines.append(f'Stock Dividends:           ${summary["total_dividends"]:>12,.2f}')
    lines.append(f'CD Interest:               ${summary["total_cd_interest"]:>12,.2f}')
    lines.append(f'Savings Interest:          ${summary["total_savings_interest"]:>12,.2f}')
    lines.append(f'Total Taxable Income:      ${summary["total_income_taxable"]:>12,.2f}')
    lines.append('')

    if summary['holdings']:
        lines.append('--- Holdings Detail ---')
        lines.append(f'{"Ticker":<10} {"Type":<6} {"Shares":>10} {"Price":>10} '
                      f'{"Value":>12} {"Invested":>12} {"Gain/Loss":>12}')
        lines.append('-' * 72)
        for h in summary['holdings']:
            lines.append(f'{h["ticker"]:<10} {h["type"]:<6} {h["shares"]:>10.2f} '
                          f'${h["price"]:>9.2f} ${h["value"]:>11,.2f} '
                          f'${h["net_invested"]:>11,.2f} ${h["unrealized_gain"]:>11,.2f}')

    lines.append('')
    lines.append('=' * 60)
    return '\n'.join(lines)


def compare_strategies(strategy_configs, date=None):
    '''Run multiple investment strategies against the same parameters and
    compare their outcomes.

    Each strategy config should be a dict with:
    - 'name': string label for the strategy
    - 'money_manager': a fully configured MoneyManager instance
      (with schedule already executed)

    INPUTS:
    strategy_configs - list of strategy config dicts
    date - optional date for the comparison snapshot

    OUTPUTS:
    dict with:
    - 'strategies': list of {name, summary} for each strategy
    - 'comparison_table': DataFrame comparing key metrics side by side'''
    results = []

    for config in strategy_configs:
        name = config['name']
        mm = config['money_manager']
        summary = financial_summary(mm, date=date)
        results.append({
            'name': name,
            'summary': summary,
        })

    # Build comparison table
    rows = []
    for r in results:
        s = r['summary']
        rows.append({
            'Strategy': r['name'],
            'Total Value': s['total_portfolio_value'],
            'Principal': s['total_principal'],
            'Unrealized Gains': s['total_unrealized_gains'],
            'Dividends': s['total_dividends'],
            'CD Interest': s['total_cd_interest'],
            'Savings Interest': s['total_savings_interest'],
            'Cash': s['cash'],
        })

    comparison_table = pd.DataFrame(rows)

    if len(comparison_table) > 1:
        # Add a row showing the difference from the first strategy
        baseline = comparison_table.iloc[0]
        for i in range(1, len(comparison_table)):
            for col in ['Total Value', 'Principal', 'Unrealized Gains',
                        'Dividends', 'CD Interest', 'Savings Interest',
                        'Cash']:
                diff = comparison_table.iloc[i][col] - baseline[col]
                comparison_table.at[comparison_table.index[i],
                                     f'{col} (vs {results[0]["name"]})'] = diff

    logger.info(f'Compared {len(results)} strategies')

    return {
        'strategies': results,
        'comparison_table': comparison_table,
    }


def format_comparison(comparison):
    '''Format a strategy comparison into a human-readable string report.

    INPUTS:
    comparison - dict from compare_strategies()

    OUTPUTS:
    str - formatted comparison report'''
    lines = []
    lines.append('=' * 70)
    lines.append('STRATEGY COMPARISON')
    lines.append('=' * 70)
    lines.append('')

    table = comparison['comparison_table']
    base_cols = ['Strategy', 'Total Value', 'Principal', 'Unrealized Gains',
                 'Dividends', 'Cash']
    cols = [c for c in base_cols if c in table.columns]

    # Header
    header = f'{"Strategy":<20}'
    for col in cols[1:]:
        header += f'{col:>14}'
    lines.append(header)
    lines.append('-' * len(header))

    # Rows
    for _, row in table.iterrows():
        line = f'{str(row["Strategy"]):<20}'
        for col in cols[1:]:
            line += f'${row[col]:>13,.2f}'
        lines.append(line)

    lines.append('')
    lines.append('=' * 70)
    return '\n'.join(lines)
