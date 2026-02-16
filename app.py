"""Streamlit application for the Investment Assessment tool.

Provides a web UI for all operations described in the project README:
1. Fetch stock data from Alpha Vantage
2. Create investments (stocks from CSV, CDs)
3. Set up investment schedules with auto-buy strategies
4. Manage and execute a portfolio
5. Generate financial summary reports and strategy comparisons
6. Visualize portfolio value over time
7. Strategy Wizard – one-page strategy comparison
8. Parameter Sweep – sensitivity analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import subprocess
import json
import logging
import tempfile
import time
import copy

from src.investment import Investment, CDInvestment
from src.money_manager import MoneyManager
from src.schedule import Schedule
from src.reporting import financial_summary, format_summary, compare_strategies, format_comparison
from src.strategy_templates import (
    STRATEGY_TEMPLATES,
    get_compatible_templates,
    save_strategy_config,
    load_strategy_config,
    list_saved_strategies,
    delete_saved_strategy,
    build_and_run_strategy,
)
from src.vanguard_etfs import VANGUARD_ETFS, get_etf_options, get_ticker_from_option

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    """Ensure all required session-state keys exist."""
    defaults = {
        "investments": {},       # ticker -> Investment
        "schedules": {},         # name  -> Schedule
        "managers": {},          # name  -> MoneyManager
        "csv_files": {},         # ticker -> path
        "summaries": {},         # name  -> summary dict
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_portfolio_chart(managers_dict):
    """Build a Plotly figure showing portfolio value, principal, and gains
    over time for one or more MoneyManager instances.

    INPUTS:
    managers_dict - dict of {label: MoneyManager}

    OUTPUTS:
    plotly Figure"""
    fig = go.Figure()

    for mgr_name, mm in managers_dict.items():
        all_dates = set()
        for inv in mm.investments.values():
            if not inv.history.empty and "date" in inv.history.columns:
                all_dates.update(inv.history["date"].tolist())

        if not all_dates:
            continue

        sorted_dates = sorted(all_dates)
        total_values = []
        total_invested = []

        for d in sorted_dates:
            tv = 0.0
            ti = 0.0
            for inv in mm.investments.values():
                try:
                    idx = inv._get_row_index(d)
                    tv += inv.history.at[idx, "total_value"]
                    ti += inv.history.loc[:idx, "purchase_amt"].sum()
                except (ValueError, KeyError):
                    pass
            total_values.append(tv)
            total_invested.append(ti)

        fig.add_trace(go.Scatter(
            x=sorted_dates, y=total_values,
            mode="lines", name=f"{mgr_name} - Total Value",
            line=dict(width=2),
        ))
        fig.add_trace(go.Scatter(
            x=sorted_dates, y=total_invested,
            mode="lines", name=f"{mgr_name} - Principal",
            line=dict(width=1, dash="dash"),
        ))
        gains = [v - p for v, p in zip(total_values, total_invested)]
        fig.add_trace(go.Scatter(
            x=sorted_dates, y=gains,
            mode="lines", name=f"{mgr_name} - Unrealized Gain/Loss",
            line=dict(width=1, dash="dot"),
        ))

    fig.update_layout(
        title="Portfolio Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        hovermode="x unified",
    )
    return fig


def _export_section(dataframe=None, text_report=None, json_data=None, prefix="export"):
    """Render download buttons for various export formats."""
    cols = st.columns(3)
    if dataframe is not None:
        with cols[0]:
            csv = dataframe.to_csv(index=False)
            st.download_button(
                "Download CSV", csv, f"{prefix}.csv", "text/csv",
                key=f"dl_{prefix}_csv",
            )
    if text_report is not None:
        with cols[1]:
            st.download_button(
                "Download Report (TXT)", text_report, f"{prefix}.txt",
                "text/plain", key=f"dl_{prefix}_txt",
            )
    if json_data is not None:
        with cols[2]:
            st.download_button(
                "Download Config (JSON)", json_data, f"{prefix}.json",
                "application/json", key=f"dl_{prefix}_json",
            )


def _handle_redeploy():
    """Pull the latest code from git, install dependencies, and restart the app."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(app_dir, "requirements.txt")

    # Step 1: git pull
    with st.sidebar:
        with st.spinner("Pulling latest changes..."):
            git_result = subprocess.run(
                ["git", "pull"],
                cwd=app_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

        if git_result.returncode != 0:
            st.error(f"git pull failed:\n```\n{git_result.stderr}\n```")
            return

        st.success(f"git pull: {git_result.stdout.strip()}")

        # Step 2: pip install (best-effort — don't block restart on failure)
        with st.spinner("Installing dependencies..."):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path],
                cwd=app_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )

        st.info("Restarting application...")

    # Step 3: Restart the process by re-exec'ing the original command line.
    # Reading /proc/self/cmdline preserves any flags Streamlit was started
    # with (e.g. --server.port).
    try:
        with open("/proc/self/cmdline", "rb") as f:
            raw = f.read().split(b"\x00")
            cmdline = [arg.decode() for arg in raw if arg]
    except (IOError, OSError):
        # Fallback: reconstruct a reasonable default command.
        cmdline = [
            sys.executable, "-m", "streamlit", "run",
            os.path.join(app_dir, "app.py"),
        ]

    os.chdir(app_dir)
    os.execv(cmdline[0], cmdline)


def _ticker_input(label, key, default_value=""):
    """Render a ticker input that combines a Vanguard ETF dropdown with a
    manual text entry fallback.  Returns the selected/entered ticker string
    (uppercased) or empty string if nothing chosen."""
    etf_options = get_etf_options()
    choices = ["-- Manual entry --"] + etf_options

    selected = st.selectbox(
        f"{label} (Vanguard ETFs)",
        choices,
        key=f"{key}_vanguard_select",
    )

    if selected == "-- Manual entry --":
        ticker = st.text_input(label, value=default_value, key=key)
    else:
        ticker = get_ticker_from_option(selected)
        st.caption(f"Selected: **{ticker}** — {VANGUARD_ETFS.get(ticker, '')}")

    return ticker.strip().upper() if ticker else ""


# ---------------------------------------------------------------------------
# Page: Fetch Stock Data
# ---------------------------------------------------------------------------

def page_fetch_data():
    st.header("Fetch Stock Data")
    st.markdown(
        "Download weekly adjusted price data from Alpha Vantage and "
        "load it into the application."
    )

    tab_download, tab_upload = st.tabs(["Download from API", "Upload CSV"])

    # --- Download tab ---
    with tab_download:
        api_key = st.text_input(
            "Alpha Vantage API Key",
            type="password",
            help="Get a free key at https://www.alphavantage.co/support/#api-key",
        )
        etf_options = get_etf_options()
        selected_etfs = st.multiselect(
            "Select Vanguard ETFs",
            etf_options,
            default=[
                opt for opt in etf_options
                if get_ticker_from_option(opt) in ("VTI", "VNQ", "VOO")
            ],
            key="download_vanguard_etfs",
        )
        tickers_input = st.text_input(
            "Additional tickers (comma-separated)",
            value="",
            help="Enter non-Vanguard or additional ticker symbols here",
        )
        if st.button("Download", disabled=not api_key):
            tickers = [get_ticker_from_option(opt) for opt in selected_etfs]
            manual = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            tickers.extend(t for t in manual if t not in tickers)
            from src.ticker_collection import download_ticker

            progress = st.progress(0, text="Starting downloads...")
            for i, ticker in enumerate(tickers):
                # Wait between API calls to avoid Alpha Vantage rate limits
                if i > 0:
                    time.sleep(1.5)
                progress.progress(
                    (i) / len(tickers),
                    text=f"Downloading {ticker}...",
                )
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        path = download_ticker(ticker, api_key, write_loc=tmp)
                        inv = Investment.from_csv(ticker, path)
                        st.session_state["investments"][ticker] = inv
                        st.session_state["csv_files"][ticker] = path
                        st.success(f"Downloaded **{ticker}** ({len(inv.history)} rows)")
                except Exception as e:
                    st.error(f"Failed to download {ticker}: {e}")
            progress.progress(1.0, text="Done")

    # --- Upload tab ---
    with tab_upload:
        ticker = _ticker_input("Ticker symbol", key="upload_ticker")
        uploaded = st.file_uploader(
            "Upload Alpha Vantage CSV",
            type=["csv"],
            key="csv_upload",
        )
        divisible = st.checkbox("Allow fractional shares", value=True, key="upload_div")
        if st.button("Load CSV") and uploaded and ticker:
            try:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                inv = Investment.from_csv(ticker.upper(), tmp_path, divisible=divisible)
                st.session_state["investments"][ticker.upper()] = inv
                st.success(
                    f"Loaded **{ticker.upper()}** with {len(inv.history)} price records"
                )
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error loading CSV: {e}")

    # --- Show loaded investments ---
    if st.session_state["investments"]:
        st.subheader("Loaded Investments")
        for t, inv in st.session_state["investments"].items():
            with st.expander(f"{t} ({len(inv.history)} rows)"):
                st.dataframe(
                    inv.history[["date", "ticker_price", "per_stock_return"]].head(20),
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Page: Create Investments
# ---------------------------------------------------------------------------

def page_create_investments():
    st.header("Create Investments")

    tab_stock, tab_cd, tab_mm = st.tabs(["Stock Investment", "CD Investment", "Money Market"])

    # --- Stock from manual price data ---
    with tab_stock:
        st.markdown("Create a stock investment with manually entered price data.")
        ticker = _ticker_input("Ticker symbol", key="manual_ticker")
        divisible = st.checkbox("Fractional shares allowed", value=True, key="manual_div")
        num_rows = st.number_input("Number of price rows", 2, 52, 12, key="manual_rows")
        start_date = st.date_input("Start date", value=pd.Timestamp("2023-01-02"), key="manual_start")
        freq = st.selectbox("Frequency", ["W-MON", "MS", "W-FRI", "D"], key="manual_freq")
        base_price = st.number_input("Starting price ($)", value=100.0, min_value=0.01, key="manual_price")

        if st.button("Create Stock Investment", key="btn_create_stock") and ticker:
            dates = pd.date_range(start=start_date, periods=num_rows, freq=freq)
            np.random.seed(42)
            prices = base_price * np.cumprod(1 + np.random.normal(0.002, 0.03, num_rows))
            ph = pd.DataFrame({
                "date": dates,
                "ticker_price": prices,
                "per_stock_return": 0.0,
            })
            inv = Investment(ticker.upper(), price_history=ph, divisible=divisible)
            st.session_state["investments"][ticker.upper()] = inv
            st.success(f"Created stock investment **{ticker.upper()}**")

    # --- CD Investment ---
    with tab_cd:
        st.markdown("Create a Certificate of Deposit investment.")
        cd_ticker = st.text_input("CD name", value="CD-5.0%-2023", key="cd_ticker")
        col1, col2 = st.columns(2)
        with col1:
            cd_start = st.date_input("Start date", key="cd_start")
        with col2:
            cd_end = st.date_input("End date", value=pd.Timestamp.today() + pd.DateOffset(years=1), key="cd_end")
        cd_rate = st.number_input("Annual rate (%)", value=5.0, min_value=0.0, max_value=100.0, key="cd_rate") / 100.0
        cd_principal = st.number_input("Principal ($)", value=10000.0, min_value=0.0, key="cd_principal")

        if st.button("Create CD", key="btn_create_cd"):
            try:
                cd = CDInvestment(
                    cd_ticker, start_date=str(cd_start), end_date=str(cd_end),
                    rate=cd_rate, principal=cd_principal,
                )
                st.session_state["investments"][cd_ticker] = cd
                st.success(f"Created CD **{cd_ticker}** (${cd_principal:,.2f} at {cd_rate*100:.1f}%)")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Money Market ---
    with tab_mm:
        st.markdown("Create a money market fund (price fixed at $1.00).")
        mm_ticker = st.text_input("Money market ticker", value="VMFXX", key="mm_ticker")
        mm_periods = st.number_input("Number of periods", 12, 520, 52, key="mm_periods")
        mm_start = st.date_input("Start date", value=pd.Timestamp("2023-01-02"), key="mm_start")

        if st.button("Create Money Market", key="btn_create_mm"):
            dates = pd.date_range(start=mm_start, periods=mm_periods, freq="W-MON")
            ph = pd.DataFrame({
                "date": dates,
                "ticker_price": 1.0,
                "per_stock_return": 0.0,
            })
            inv = Investment(mm_ticker.upper(), price_history=ph, divisible=True)
            st.session_state["investments"][mm_ticker.upper()] = inv
            st.success(f"Created money market **{mm_ticker.upper()}** ({mm_periods} periods)")

    # Show loaded investments
    if st.session_state["investments"]:
        st.divider()
        st.subheader("All Investments")
        rows = []
        for t, inv in st.session_state["investments"].items():
            is_cd = hasattr(inv, "rate") and hasattr(inv, "end_date")
            rows.append({
                "Ticker": t,
                "Type": "CD" if is_cd else "Stock/Fund",
                "Records": len(inv.history),
                "Divisible": inv.divisible,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Investment Schedule
# ---------------------------------------------------------------------------

def page_schedule():
    st.header("Investment Schedule")
    st.markdown("Define recurring deposit schedules and automatic buying strategies.")

    available_tickers = list(st.session_state["investments"].keys())
    if not available_tickers:
        st.warning("Create or load investments first.")
        return

    mm_ticker = st.selectbox(
        "Money market ticker",
        available_tickers,
        index=available_tickers.index("VMFXX") if "VMFXX" in available_tickers else 0,
        key="sched_mm",
    )

    st.subheader("Recurring Deposits")
    col1, col2 = st.columns(2)
    with col1:
        dep_start = st.date_input("Start date", key="dep_start")
        dep_freq = st.selectbox("Frequency", ["MS", "W-MON", "2W", "QS"], key="dep_freq",
                                format_func=lambda x: {"MS": "Monthly", "W-MON": "Weekly", "2W": "Bi-weekly", "QS": "Quarterly"}[x])
    with col2:
        dep_end = st.date_input("End date", value=pd.Timestamp.today(), key="dep_end")
        dep_amount = st.number_input("Deposit amount ($)", value=1000.0, min_value=0.0, key="dep_amt")

    schedule_name = st.text_input("Schedule name", value="default", key="sched_name")

    if st.button("Create Schedule", key="btn_create_sched"):
        sched = Schedule(money_market=mm_ticker)
        sched.automate_investment_schedule(
            start=str(dep_start), stop=str(dep_end),
            freq=dep_freq, amount=dep_amount,
        )
        st.session_state["schedules"][schedule_name] = sched
        st.success(
            f"Created schedule **{schedule_name}** with "
            f"{len(sched.hist)} deposits totalling "
            f"${sched.hist['dollars'].sum():,.2f}"
        )

    # Show schedules
    if st.session_state["schedules"]:
        st.divider()
        st.subheader("Existing Schedules")
        for name, sched in st.session_state["schedules"].items():
            with st.expander(f"Schedule: {name} ({len(sched.hist)} transactions)"):
                st.dataframe(sched.hist, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Portfolio Management
# ---------------------------------------------------------------------------

def page_portfolio():
    st.header("Portfolio Management")
    st.markdown("Build a portfolio, execute the schedule, and inspect the state.")

    available_tickers = list(st.session_state["investments"].keys())
    schedule_names = list(st.session_state["schedules"].keys())

    if not available_tickers:
        st.warning("Create or load investments first.")
        return

    st.subheader("Create Portfolio")
    mgr_name = st.text_input("Portfolio name", value="portfolio_1", key="mgr_name")
    mm_ticker = st.selectbox(
        "Money market ticker",
        available_tickers,
        index=available_tickers.index("VMFXX") if "VMFXX" in available_tickers else 0,
        key="port_mm",
    )
    selected_tickers = st.multiselect(
        "Select investments to include",
        available_tickers,
        default=available_tickers,
        key="port_tickers",
    )
    selected_schedule = st.selectbox(
        "Schedule to attach",
        ["(none)"] + schedule_names,
        key="port_schedule",
    )

    if st.button("Create Portfolio", key="btn_create_port"):
        sched = st.session_state["schedules"].get(selected_schedule) if selected_schedule != "(none)" else None
        mm = MoneyManager(schedule=sched, money_market_ticker=mm_ticker)
        for t in selected_tickers:
            mm.add_investment(st.session_state["investments"][t])
        st.session_state["managers"][mgr_name] = mm
        st.success(
            f"Created portfolio **{mgr_name}** with {len(mm.investments)} investments"
        )

    # Run schedule
    if st.session_state["managers"]:
        st.divider()
        st.subheader("Execute Schedule")
        mgr_to_run = st.selectbox("Select portfolio", list(st.session_state["managers"].keys()), key="run_mgr")
        if st.button("Run Schedule", key="btn_run_sched"):
            mm = st.session_state["managers"][mgr_to_run]
            if mm.schedule is None:
                st.error("This portfolio has no schedule attached.")
            else:
                try:
                    mm.run_schedule()
                    st.success(
                        f"Executed {len(mm.activity_lst)} transactions on **{mgr_to_run}**"
                    )
                except Exception as e:
                    st.error(f"Error running schedule: {e}")

        # Manual operations
        st.divider()
        st.subheader("Manual Operations")
        mgr_manual = st.selectbox("Portfolio", list(st.session_state["managers"].keys()), key="manual_mgr")
        mm = st.session_state["managers"][mgr_manual]

        op = st.radio("Operation", ["Transfer Cash", "Buy Ticket", "Sell Ticket"], horizontal=True, key="manual_op")
        col1, col2 = st.columns(2)
        with col1:
            op_date = st.date_input("Date", key="op_date")
            op_dollars = st.number_input("Dollars ($)", value=1000.0, min_value=0.0, key="op_dollars")
        with col2:
            op_ticker = st.selectbox("Ticker", list(mm.investments.keys()), key="op_ticker")

        if st.button("Execute", key="btn_exec_op"):
            try:
                if op == "Transfer Cash":
                    shares, dollars = mm.transfer_cash(time=str(op_date), amount=op_dollars, into=op_ticker)
                    st.success(f"Transferred ${dollars:,.2f} into {op_ticker}")
                elif op == "Buy Ticket":
                    shares, dollars = mm.buy_ticket(dollars=op_dollars, date=str(op_date), ticket=op_ticker)
                    st.success(f"Bought {shares:.4f} shares of {op_ticker} for ${dollars:,.2f}")
                elif op == "Sell Ticket":
                    shares, dollars = mm.sell_ticket(dollars=op_dollars, date=str(op_date), ticket=op_ticker)
                    st.success(f"Sold {shares:.4f} shares of {op_ticker} for ${dollars:,.2f}")
            except Exception as e:
                st.error(f"Error: {e}")

        # Show portfolio state
        st.divider()
        st.subheader("Portfolio State")
        mgr_state = st.selectbox("Portfolio", list(st.session_state["managers"].keys()), key="state_mgr")
        state_date = st.date_input("As of date (leave blank for latest)", value=None, key="state_date")
        if st.button("Show State", key="btn_show_state"):
            mm = st.session_state["managers"][mgr_state]
            try:
                state = mm.return_current_financial_state(date=str(state_date) if state_date else None)
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Value", f"${state['total_value']:,.2f}")
                col2.metric("Cash", f"${state['cash']:,.2f}")
                col3.metric("Total Dividends", f"${state['total_dividends']:,.2f}")
                if state["holdings"]:
                    holdings_df = pd.DataFrame([
                        {"Ticker": t, "Shares": h["shares"], "Price": h["price"],
                         "Value": h["value"], "Invested": h["invested"],
                         "Dividends": h["dividends"]}
                        for t, h in state["holdings"].items()
                    ])
                    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Page: Reports
# ---------------------------------------------------------------------------

def page_reports():
    st.header("Financial Reports")

    if not st.session_state["managers"]:
        st.warning("Create a portfolio first.")
        return

    tab_single, tab_compare = st.tabs(["Single Summary", "Strategy Comparison"])

    with tab_single:
        mgr_name = st.selectbox("Portfolio", list(st.session_state["managers"].keys()), key="rpt_mgr")
        rpt_date = st.date_input("Report date (blank for latest)", value=None, key="rpt_date")
        if st.button("Generate Summary", key="btn_gen_summary"):
            mm = st.session_state["managers"][mgr_name]
            try:
                summary = financial_summary(mm, date=str(rpt_date) if rpt_date else None)
                st.session_state["summaries"][mgr_name] = summary

                col1, col2 = st.columns(2)
                col1.metric("Total Portfolio Value", f"${summary['total_portfolio_value']:,.2f}")
                col2.metric("Cash Position", f"${summary['cash']:,.2f}")

                col3, col4, col5 = st.columns(3)
                col3.metric("Principal Invested", f"${summary['total_principal']:,.2f}")
                col4.metric("Unrealized Gains", f"${summary['total_unrealized_gains']:,.2f}")
                col5.metric("Taxable Income", f"${summary['total_income_taxable']:,.2f}")

                if summary["holdings"]:
                    st.subheader("Holdings Detail")
                    df = pd.DataFrame(summary["holdings"])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Full Text Report"):
                    st.code(format_summary(summary))

                # Export buttons
                st.subheader("Export")
                text_rpt = format_summary(summary)
                holdings_df = pd.DataFrame(summary["holdings"]) if summary["holdings"] else None
                _export_section(
                    dataframe=holdings_df,
                    text_report=text_rpt,
                    prefix=f"summary_{mgr_name}",
                )
            except Exception as e:
                st.error(f"Error generating summary: {e}")

    with tab_compare:
        st.markdown("Compare multiple portfolios side by side.")
        selected = st.multiselect(
            "Select portfolios to compare",
            list(st.session_state["managers"].keys()),
            key="cmp_mgrs",
        )
        cmp_date = st.date_input("Comparison date (blank for latest)", value=None, key="cmp_date")
        if st.button("Compare", key="btn_compare") and len(selected) >= 2:
            configs = [
                {"name": name, "money_manager": st.session_state["managers"][name]}
                for name in selected
            ]
            try:
                result = compare_strategies(configs, date=str(cmp_date) if cmp_date else None)
                st.subheader("Comparison Table")
                st.dataframe(result["comparison_table"], use_container_width=True, hide_index=True)

                with st.expander("Full Text Comparison"):
                    st.code(format_comparison(result))

                # Export buttons
                st.subheader("Export")
                _export_section(
                    dataframe=result["comparison_table"],
                    text_report=format_comparison(result),
                    prefix="strategy_comparison",
                )
            except Exception as e:
                st.error(f"Error: {e}")
        elif st.session_state.get("btn_compare") and len(selected) < 2:
            st.info("Select at least 2 portfolios to compare.")


# ---------------------------------------------------------------------------
# Page: Financial Impact Graph
# ---------------------------------------------------------------------------

def page_graph():
    st.header("Financial Impact Graph")
    st.markdown("Visualize portfolio value, principal invested, and gains over time.")

    if not st.session_state["managers"]:
        st.warning("Create and run a portfolio first.")
        return

    selected = st.multiselect(
        "Select portfolios to graph",
        list(st.session_state["managers"].keys()),
        default=list(st.session_state["managers"].keys())[:1],
        key="graph_mgrs",
    )

    if not selected:
        return

    managers_to_plot = {name: st.session_state["managers"][name] for name in selected}
    fig = _build_portfolio_chart(managers_to_plot)
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics table
    if selected:
        st.subheader("Current Snapshot")
        rows = []
        for mgr_name in selected:
            mm = st.session_state["managers"][mgr_name]
            try:
                state = mm.return_current_financial_state()
                rows.append({
                    "Portfolio": mgr_name,
                    "Total Value": f"${state['total_value']:,.2f}",
                    "Cash": f"${state['cash']:,.2f}",
                    "Invested": f"${state['total_invested']:,.2f}",
                    "Dividends": f"${state['total_dividends']:,.2f}",
                })
            except Exception:
                pass
        if rows:
            snapshot_df = pd.DataFrame(rows)
            st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

            # Export
            st.subheader("Export")
            _export_section(dataframe=snapshot_df, prefix="graph_snapshot")


# ---------------------------------------------------------------------------
# Page: Demo (quick start with sample data)
# ---------------------------------------------------------------------------

def page_demo():
    st.header("Quick Demo")
    st.markdown(
        "Generate sample data and run a complete simulation to see the "
        "application in action without needing an API key."
    )

    if st.button("Run Demo", key="btn_demo"):
        with st.spinner("Building demo portfolio..."):
            np.random.seed(42)
            weeks = 104  # 2 years

            # Money market
            mm_dates = pd.date_range(start="2022-01-03", periods=weeks, freq="W-MON")
            mm_ph = pd.DataFrame({
                "date": mm_dates, "ticker_price": 1.0, "per_stock_return": 0.0,
            })
            vmfxx = Investment("VMFXX", price_history=mm_ph, divisible=True)

            # VTI-like stock
            vti_prices = 200.0 * np.cumprod(1 + np.random.normal(0.002, 0.025, weeks))
            vti_divs = np.zeros(weeks)
            vti_divs[12::13] = 0.75  # quarterly-ish dividends
            vti_ph = pd.DataFrame({
                "date": mm_dates, "ticker_price": vti_prices, "per_stock_return": vti_divs,
            })
            vti = Investment("VTI", price_history=vti_ph, divisible=True)

            # VNQ-like REIT
            vnq_prices = 90.0 * np.cumprod(1 + np.random.normal(0.001, 0.03, weeks))
            vnq_divs = np.zeros(weeks)
            vnq_divs[12::13] = 0.55
            vnq_ph = pd.DataFrame({
                "date": mm_dates, "ticker_price": vnq_prices, "per_stock_return": vnq_divs,
            })
            vnq = Investment("VNQ", price_history=vnq_ph, divisible=True)

            # Store investments
            st.session_state["investments"]["VMFXX"] = vmfxx
            st.session_state["investments"]["VTI"] = vti
            st.session_state["investments"]["VNQ"] = vnq

            # Strategy 1: 70/30 VTI/VNQ
            sched1 = Schedule(money_market="VMFXX")
            sched1.automate_investment_schedule(
                start="2022-01-03", stop="2023-12-25", freq="MS", amount=1000.0,
            )
            st.session_state["schedules"]["demo_70_30"] = sched1

            mm1 = MoneyManager(schedule=sched1, money_market_ticker="VMFXX")
            mm1.add_investment(Investment("VMFXX", price_history=mm_ph.copy(), divisible=True))
            mm1.add_investment(Investment("VTI", price_history=vti_ph.copy(), divisible=True))
            mm1.add_investment(Investment("VNQ", price_history=vnq_ph.copy(), divisible=True))
            mm1.run_schedule()

            # After deposits, simulate proportional buys by manually executing
            # buy operations for each deposit date
            _simulate_proportional_buys(mm1, {"VTI": 0.7, "VNQ": 0.3})
            st.session_state["managers"]["Demo 70/30"] = mm1

            # Strategy 2: 100% VTI
            sched2 = Schedule(money_market="VMFXX")
            sched2.automate_investment_schedule(
                start="2022-01-03", stop="2023-12-25", freq="MS", amount=1000.0,
            )
            st.session_state["schedules"]["demo_100_vti"] = sched2

            mm2 = MoneyManager(schedule=sched2, money_market_ticker="VMFXX")
            mm2.add_investment(Investment("VMFXX", price_history=mm_ph.copy(), divisible=True))
            mm2.add_investment(Investment("VTI", price_history=vti_ph.copy(), divisible=True))
            mm2.run_schedule()
            _simulate_proportional_buys(mm2, {"VTI": 1.0})
            st.session_state["managers"]["Demo 100% VTI"] = mm2

        st.success("Demo portfolios created! Use the sidebar to explore Reports and Financial Graph pages.")


def _simulate_proportional_buys(mm, allocations):
    """After deposits are executed, buy stocks proportionally from the money market."""
    mm_inv = mm.investments.get(mm.money_market_ticker)
    if mm_inv is None or mm_inv.history.empty:
        return

    for _, row in mm_inv.history.iterrows():
        date = row["date"]
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


# ---------------------------------------------------------------------------
# Page: Strategy Wizard (Features 1 + 2 + 3)
# ---------------------------------------------------------------------------

FREQ_OPTIONS = {
    "MS": "Monthly",
    "W-MON": "Weekly",
    "2W": "Bi-weekly",
    "QS": "Quarterly",
}


def page_strategy_wizard():
    st.header("Strategy Wizard")
    st.markdown(
        "Define multiple investment strategies and compare them side-by-side "
        "with a single click. Pick from built-in templates, load saved "
        "strategies, or build your own."
    )

    available_tickers = list(st.session_state["investments"].keys())
    if not available_tickers:
        st.warning(
            "No investments loaded. Use **Quick Demo** for sample data, "
            "**Fetch Stock Data** to download real data, or "
            "**Create Investments** to build manual data."
        )
        return

    # ---- Shared settings ----
    st.subheader("Shared Settings")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        mm_ticker = st.selectbox(
            "Money market",
            available_tickers,
            index=available_tickers.index("VMFXX") if "VMFXX" in available_tickers else 0,
            key="wiz_mm",
        )
    with col_s2:
        wiz_start = st.date_input("Start date", value=pd.Timestamp("2022-01-03"), key="wiz_start")
    with col_s3:
        wiz_end = st.date_input("End date", value=pd.Timestamp("2023-12-25"), key="wiz_end")
    with col_s4:
        wiz_freq = st.selectbox(
            "Deposit frequency",
            list(FREQ_OPTIONS.keys()),
            format_func=lambda x: FREQ_OPTIONS[x],
            key="wiz_freq",
        )
    wiz_amount = st.number_input("Deposit amount ($)", value=1000.0, min_value=0.0, step=100.0, key="wiz_amount")

    non_mm_tickers = [t for t in available_tickers if t != mm_ticker]

    # ---- Build template choices ----
    compatible_templates = get_compatible_templates(available_tickers)
    saved_strategies = list_saved_strategies()
    compatible_saved = {
        name: cfg for name, cfg in saved_strategies.items()
        if set(cfg.get("allocations", {}).keys()) | {cfg.get("money_market", "VMFXX")}
        <= set(available_tickers)
    }

    template_choices = ["Custom"]
    template_choices += compatible_templates
    if compatible_saved:
        template_choices += [f"[Saved] {name}" for name in compatible_saved.keys()]

    # ---- Strategy definitions ----
    st.divider()
    st.subheader("Define Strategies")
    num_strategies = st.number_input("Number of strategies to compare", 2, 4, 2, key="wiz_num")

    strategy_configs = []
    tabs = st.tabs([f"Strategy {i + 1}" for i in range(num_strategies)])

    for i, tab in enumerate(tabs):
        with tab:
            selected_tmpl = st.selectbox(
                "Start from template",
                template_choices,
                key=f"wiz_tmpl_{i}",
            )

            # Determine defaults based on selection
            if selected_tmpl == "Custom":
                def_name = f"Strategy {i + 1}"
                def_allocs = {}
            elif selected_tmpl.startswith("[Saved] "):
                saved_name = selected_tmpl[8:]
                cfg = compatible_saved[saved_name]
                def_name = saved_name
                def_allocs = cfg.get("allocations", {})
            else:
                cfg = STRATEGY_TEMPLATES[selected_tmpl]
                def_name = selected_tmpl
                def_allocs = cfg.get("allocations", {})

            name = st.text_input(
                "Strategy name",
                value=def_name,
                key=f"wiz_name_{i}_{selected_tmpl}",
            )

            # Ticker allocation
            default_tickers = [t for t in def_allocs.keys() if t in non_mm_tickers]
            selected_tickers = st.multiselect(
                "Investment tickers",
                non_mm_tickers,
                default=default_tickers,
                key=f"wiz_tk_{i}_{selected_tmpl}",
            )

            allocations = {}
            if selected_tickers:
                alloc_cols = st.columns(len(selected_tickers))
                for j, ticker in enumerate(selected_tickers):
                    with alloc_cols[j]:
                        raw_default = def_allocs.get(ticker, 0.0)
                        # Template values are 0-1 fractions; UI shows 0-100%
                        default_pct = raw_default * 100.0 if raw_default <= 1.0 else raw_default
                        if default_pct == 0.0 and len(selected_tickers) > 0:
                            default_pct = round(100.0 / len(selected_tickers), 1)
                        pct = st.number_input(
                            f"{ticker} %",
                            min_value=0.0,
                            max_value=100.0,
                            value=default_pct,
                            step=5.0,
                            key=f"wiz_pct_{i}_{selected_tmpl}_{ticker}",
                        )
                        allocations[ticker] = pct / 100.0

            total_pct = sum(allocations.values()) * 100
            if allocations and abs(total_pct - 100.0) > 0.1:
                st.caption(f"Allocation total: {total_pct:.1f}% (remainder stays in money market)")

            strategy_configs.append({
                "name": name,
                "allocations": allocations,
                "deposit_freq": wiz_freq,
                "deposit_amount": wiz_amount,
                "money_market": mm_ticker,
                "buy_strategy": "basic",
            })

    # ---- Save strategy ----
    st.divider()
    st.subheader("Save / Manage Strategies")
    col_sv1, col_sv2 = st.columns([3, 1])
    with col_sv1:
        save_idx = st.selectbox(
            "Strategy to save",
            range(num_strategies),
            format_func=lambda i: strategy_configs[i]["name"] if i < len(strategy_configs) else f"Strategy {i + 1}",
            key="wiz_save_idx",
        )
    with col_sv2:
        if st.button("Save to File", key="wiz_save_btn"):
            cfg_to_save = strategy_configs[save_idx]
            try:
                filepath = save_strategy_config(cfg_to_save, cfg_to_save["name"])
                st.success(f"Saved **{cfg_to_save['name']}** to `{filepath}`")
            except Exception as e:
                st.error(f"Error saving: {e}")

    # Show saved strategies with delete option
    if compatible_saved:
        with st.expander("Manage Saved Strategies"):
            for sname in list(compatible_saved.keys()):
                col_d1, col_d2 = st.columns([4, 1])
                col_d1.text(sname)
                if col_d2.button("Delete", key=f"wiz_del_{sname}"):
                    delete_saved_strategy(sname)
                    st.rerun()

    # ---- Run comparison ----
    st.divider()
    if st.button("Run Comparison", type="primary", key="wiz_run"):
        # Validate
        valid = True
        for cfg in strategy_configs:
            if not cfg["allocations"]:
                st.error(f"Strategy **{cfg['name']}** has no ticker allocations. Select at least one ticker.")
                valid = False

        if valid:
            results = {}
            progress = st.progress(0, text="Running simulations...")
            for idx, cfg in enumerate(strategy_configs):
                progress.progress(idx / len(strategy_configs), text=f"Running {cfg['name']}...")
                try:
                    mm = build_and_run_strategy(
                        cfg,
                        st.session_state["investments"],
                        str(wiz_start),
                        str(wiz_end),
                    )
                    results[cfg["name"]] = mm
                    st.session_state["managers"][cfg["name"]] = mm
                except Exception as e:
                    st.error(f"Error running **{cfg['name']}**: {e}")
            progress.progress(1.0, text="Done!")

            if len(results) >= 2:
                configs_for_cmp = [
                    {"name": name, "money_manager": mm}
                    for name, mm in results.items()
                ]
                comparison = compare_strategies(
                    configs_for_cmp, date=str(wiz_end),
                )
                st.session_state["wiz_comparison"] = comparison
                st.session_state["wiz_results"] = results
                st.session_state["wiz_configs"] = strategy_configs
                st.session_state["wiz_end_date"] = str(wiz_end)
            elif len(results) == 1:
                st.info("Only one strategy ran successfully. Need at least 2 to compare.")

    # ---- Display results ----
    if "wiz_comparison" in st.session_state:
        st.divider()
        st.subheader("Comparison Results")

        comparison = st.session_state["wiz_comparison"]
        results = st.session_state["wiz_results"]
        end_date = st.session_state["wiz_end_date"]

        # Enhanced comparison table with return %
        table = comparison["comparison_table"].copy()
        if "Total Value" in table.columns and "Principal" in table.columns:
            table["Total Return %"] = table.apply(
                lambda row: (
                    (row["Total Value"] - row["Principal"]) / row["Principal"] * 100
                    if row["Principal"] > 0 else 0.0
                ),
                axis=1,
            )

        st.dataframe(
            table.style.format({
                "Total Value": "${:,.2f}",
                "Principal": "${:,.2f}",
                "Unrealized Gains": "${:,.2f}",
                "Dividends": "${:,.2f}",
                "Cash": "${:,.2f}",
                "Total Return %": "{:.2f}%",
            }, na_rep=""),
            use_container_width=True,
            hide_index=True,
        )

        # Performance chart
        fig = _build_portfolio_chart(results)
        st.plotly_chart(fig, use_container_width=True)

        # Export section
        st.subheader("Export Results")
        text_rpt = format_comparison(comparison)
        json_cfg = json.dumps(st.session_state.get("wiz_configs", []), indent=2)
        _export_section(
            dataframe=table,
            text_report=text_rpt,
            json_data=json_cfg,
            prefix="wizard_comparison",
        )


# ---------------------------------------------------------------------------
# Page: Parameter Sweep (Feature 5)
# ---------------------------------------------------------------------------

def page_parameter_sweep():
    st.header("Parameter Sweep")
    st.markdown(
        "See how changing one parameter (deposit amount or allocation "
        "weight) affects investment outcomes across a range of values."
    )

    available_tickers = list(st.session_state["investments"].keys())
    if not available_tickers:
        st.warning(
            "No investments loaded. Use **Quick Demo** for sample data."
        )
        return

    # ---- Base strategy selection ----
    st.subheader("Base Strategy")

    compatible = get_compatible_templates(available_tickers)
    saved = list_saved_strategies()
    compatible_saved = {
        name: cfg for name, cfg in saved.items()
        if set(cfg.get("allocations", {}).keys()) | {cfg.get("money_market", "VMFXX")}
        <= set(available_tickers)
    }

    all_choices = {}
    for name in compatible:
        all_choices[name] = STRATEGY_TEMPLATES[name]
    for name, cfg in compatible_saved.items():
        all_choices[f"[Saved] {name}"] = cfg

    if not all_choices:
        st.warning(
            "No compatible templates found for loaded tickers. "
            "Load more tickers or create a saved strategy via the Strategy Wizard."
        )
        return

    base_name = st.selectbox("Select base strategy", list(all_choices.keys()), key="sweep_base")
    base_config = copy.deepcopy(all_choices[base_name])

    with st.expander("Base strategy details"):
        st.json(base_config)

    # ---- Date range ----
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        sweep_start = st.date_input("Start date", value=pd.Timestamp("2022-01-03"), key="sweep_start")
    with col_d2:
        sweep_end = st.date_input("End date", value=pd.Timestamp("2023-12-25"), key="sweep_end")

    # ---- Sweep parameter ----
    st.subheader("Parameter to Vary")
    sweep_type = st.radio(
        "Sweep type",
        ["Deposit Amount", "Allocation Weight"],
        horizontal=True,
        key="sweep_type",
    )

    if sweep_type == "Deposit Amount":
        col1, col2, col3 = st.columns(3)
        with col1:
            sweep_min = st.number_input("Min ($)", value=500.0, min_value=0.0, step=100.0, key="sweep_min_amt")
        with col2:
            sweep_max = st.number_input("Max ($)", value=3000.0, min_value=0.0, step=100.0, key="sweep_max_amt")
        with col3:
            sweep_step = st.number_input("Step ($)", value=500.0, min_value=1.0, step=100.0, key="sweep_step_amt")

        if sweep_min > sweep_max:
            st.error("Min must be less than or equal to Max.")
            return
        values = list(np.arange(sweep_min, sweep_max + sweep_step / 2, sweep_step))

    else:  # Allocation Weight
        alloc_tickers = list(base_config["allocations"].keys())
        if len(alloc_tickers) < 2:
            st.warning(
                "Allocation sweep requires a base strategy with at least 2 tickers. "
                "Pick a different base strategy."
            )
            return

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            sweep_ticker = st.selectbox("Ticker to vary", alloc_tickers, key="sweep_ticker")
        with col_t2:
            other_tickers = [t for t in alloc_tickers if t != sweep_ticker]
            complement_ticker = st.selectbox(
                "Complementary ticker (gets the remainder)",
                other_tickers,
                key="sweep_complement",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            sweep_min = st.number_input("Min %", value=0.0, min_value=0.0, max_value=100.0, step=5.0, key="sweep_min_pct")
        with col2:
            sweep_max = st.number_input("Max %", value=100.0, min_value=0.0, max_value=100.0, step=5.0, key="sweep_max_pct")
        with col3:
            sweep_step = st.number_input("Step %", value=10.0, min_value=1.0, max_value=50.0, step=5.0, key="sweep_step_pct")

        if sweep_min > sweep_max:
            st.error("Min must be less than or equal to Max.")
            return
        values = list(np.arange(sweep_min, sweep_max + sweep_step / 2, sweep_step))

    st.caption(f"{len(values)} simulations will be run.")

    # ---- Run sweep ----
    if st.button("Run Sweep", type="primary", key="sweep_run"):
        results = []
        progress = st.progress(0, text="Running parameter sweep...")

        for idx, val in enumerate(values):
            config = copy.deepcopy(base_config)

            if sweep_type == "Deposit Amount":
                config["deposit_amount"] = float(val)
                label = f"${val:,.0f}/mo"
            else:
                frac = float(val) / 100.0
                config["allocations"][sweep_ticker] = frac
                config["allocations"][complement_ticker] = 1.0 - frac
                label = f"{val:.0f}% {sweep_ticker}"

            progress.progress(idx / len(values), text=f"Running: {label}...")

            try:
                mm = build_and_run_strategy(
                    config,
                    st.session_state["investments"],
                    str(sweep_start),
                    str(sweep_end),
                )
                summary = financial_summary(mm, date=str(sweep_end))
                principal = summary["total_principal"]
                total_val = summary["total_portfolio_value"]
                return_pct = (
                    (total_val - principal) / principal * 100
                    if principal > 0 else 0.0
                )
                results.append({
                    "Parameter": label,
                    "Total Value": total_val,
                    "Principal": principal,
                    "Unrealized Gains": summary["total_unrealized_gains"],
                    "Total Return %": return_pct,
                    "Dividends": summary["total_dividends"],
                    "Cash": summary["cash"],
                })
            except Exception as e:
                st.warning(f"Failed for {label}: {e}")

        progress.progress(1.0, text="Done!")

        if results:
            st.session_state["sweep_results"] = results

    # ---- Display results ----
    if "sweep_results" in st.session_state:
        results = st.session_state["sweep_results"]
        df = pd.DataFrame(results)

        st.divider()
        st.subheader("Sweep Results")
        st.dataframe(
            df.style.format({
                "Total Value": "${:,.2f}",
                "Principal": "${:,.2f}",
                "Unrealized Gains": "${:,.2f}",
                "Total Return %": "{:.2f}%",
                "Dividends": "${:,.2f}",
                "Cash": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Bar + line chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Parameter"],
            y=df["Total Value"],
            name="Total Value",
            marker_color="steelblue",
        ))
        fig.add_trace(go.Scatter(
            x=df["Parameter"],
            y=df["Principal"],
            name="Principal Invested",
            mode="lines+markers",
            line=dict(color="orange", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=df["Parameter"],
            y=df["Unrealized Gains"],
            name="Unrealized Gains",
            mode="lines+markers",
            line=dict(color="green", dash="dot"),
        ))
        fig.update_layout(
            title="Parameter Sweep Results",
            xaxis_title="Parameter Value",
            yaxis_title="Value ($)",
            yaxis_tickformat="$,.0f",
            barmode="overlay",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Return % chart
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(
            x=df["Parameter"],
            y=df["Total Return %"],
            mode="lines+markers",
            name="Total Return %",
            line=dict(color="purple", width=2),
        ))
        fig_ret.update_layout(
            title="Total Return % by Parameter",
            xaxis_title="Parameter Value",
            yaxis_title="Return %",
            yaxis_tickformat=".1f",
            height=350,
        )
        st.plotly_chart(fig_ret, use_container_width=True)

        # Export
        st.subheader("Export")
        _export_section(dataframe=df, prefix="parameter_sweep")


# ---------------------------------------------------------------------------
# Main / Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Quick Demo": page_demo,
    "Strategy Wizard": page_strategy_wizard,
    "Parameter Sweep": page_parameter_sweep,
    "Fetch Stock Data": page_fetch_data,
    "Create Investments": page_create_investments,
    "Investment Schedule": page_schedule,
    "Portfolio Management": page_portfolio,
    "Financial Reports": page_reports,
    "Financial Impact Graph": page_graph,
}


def main():
    st.set_page_config(
        page_title="Investment Assessment",
        page_icon="$",
        layout="wide",
    )

    _init_state()

    st.sidebar.title("Investment Assessment")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", list(PAGES.keys()))

    # Sidebar summary
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Investments: {len(st.session_state['investments'])} | "
        f"Schedules: {len(st.session_state['schedules'])} | "
        f"Portfolios: {len(st.session_state['managers'])}"
    )

    # Redeploy section
    st.sidebar.markdown("---")
    with st.sidebar.expander("Update & Redeploy"):
        st.caption(
            "Pull the latest code from git and restart the application. "
            "All active sessions will be disconnected."
        )
        if st.button("Update & Redeploy", type="primary", key="btn_redeploy"):
            _handle_redeploy()

    PAGES[page]()


if __name__ == "__main__":
    main()
