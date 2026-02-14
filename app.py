"""Streamlit application for the Investment Assessment tool.

Provides a web UI for all operations described in the project README:
1. Fetch stock data from Alpha Vantage
2. Create investments (stocks from CSV, CDs)
3. Set up investment schedules with auto-buy strategies
4. Manage and execute a portfolio
5. Generate financial summary reports and strategy comparisons
6. Visualize portfolio value over time
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import logging
import tempfile

from src.investment import Investment, CDInvestment
from src.money_manager import MoneyManager
from src.schedule import Schedule
from src.reporting import financial_summary, format_summary, compare_strategies, format_comparison

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
        tickers_input = st.text_input(
            "Tickers (comma-separated)",
            value="VTI, VNQ, VOO",
        )
        if st.button("Download", disabled=not api_key):
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            from src.ticker_collection import download_ticker

            progress = st.progress(0, text="Starting downloads...")
            for i, ticker in enumerate(tickers):
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
        ticker = st.text_input("Ticker symbol", key="upload_ticker")
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
        ticker = st.text_input("Ticker symbol", key="manual_ticker")
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

    fig = go.Figure()

    for mgr_name in selected:
        mm = st.session_state["managers"][mgr_name]

        # Collect value over time across all holdings
        all_dates = set()
        for inv in mm.investments.values():
            if not inv.history.empty and "date" in inv.history.columns:
                all_dates.update(inv.history["date"].tolist())

        if not all_dates:
            st.info(f"No data for {mgr_name}")
            continue

        sorted_dates = sorted(all_dates)
        total_values = []
        total_invested = []
        total_dividends = []

        for d in sorted_dates:
            tv = 0.0
            ti = 0.0
            td = 0.0
            for inv in mm.investments.values():
                try:
                    idx = inv._get_row_index(d)
                    tv += inv.history.at[idx, "total_value"]
                    ti += inv.history.loc[:idx, "purchase_amt"].sum()
                    td += inv.history.at[idx, "total_dividend"]
                except (ValueError, KeyError):
                    pass
            total_values.append(tv)
            total_invested.append(ti)
            total_dividends.append(td)

        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=sorted_dates, y=total_values,
            mode="lines", name=f"{mgr_name} - Total Value",
            line=dict(width=2),
        ))

        # Principal invested line
        fig.add_trace(go.Scatter(
            x=sorted_dates, y=total_invested,
            mode="lines", name=f"{mgr_name} - Principal",
            line=dict(width=1, dash="dash"),
        ))

        # Gains (value - principal)
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
# Main / Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Quick Demo": page_demo,
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

    PAGES[page]()


if __name__ == "__main__":
    main()
