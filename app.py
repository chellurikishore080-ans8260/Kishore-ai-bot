import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Kishore Bot - Rupee P&L & Max Drawdown", layout="wide")
st.title("🇮🇳 Kishore Bot — Indian Stock & Index Scanner + Rupee Backtester")
st.caption("Strategy: Kishore Bot Trailing Stop + 200 EMA Trend Filter + ADX Strength Filter")

ASSETS = {
    "NIFTY 50 Index": "^NSEI",
    "BANK NIFTY Index": "^NSEBANK",
    "NIFTY 500 Index": "^CRSLDX",
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "State Bank of India": "SBIN.NS",
    "Larsen & Toubro": "LT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "Titan Company": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Wipro": "WIPRO.NS",
    "HCL Tech": "HCLTECH.NS",
    "Adani Enterprises": "ADANIENT.NS"
}

st.sidebar.header("⚙️ Kishore Bot Parameters")
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h", "15m", "5m"], index=0)
period_map = {"1d": "2y", "1h": "6mo", "15m": "60d", "5m": "30d"}
period = period_map[timeframe]

key_val = st.sidebar.number_input("Key Value (Sensitivity)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
atr_period = st.sidebar.number_input("ATR Period", min_value=5, max_value=30, value=10, step=1)
use_ema = st.sidebar.checkbox("Enable 200 EMA Filter", value=True)
ema_len = st.sidebar.number_input("EMA Length", min_value=20, max_value=300, value=200, step=10)
use_adx = st.sidebar.checkbox("Enable ADX Filter", value=True)
adx_min = st.sidebar.slider("Minimum ADX Threshold", min_value=10, max_value=40, value=20)

st.sidebar.header("💰 Rupee Backtest Sizing")
sizing_mode = st.sidebar.radio("Position Sizing Mode", ["Fixed Qty (Nifty Futures Lot)", "Capital Allocation (₹)"])
if sizing_mode == "Fixed Qty (Nifty Futures Lot)":
    nifty_lots = st.sidebar.number_input("Nifty Lots (1 Lot = 65 Qty)", min_value=1, max_value=50, value=1, step=1)
    fixed_qty = int(nifty_lots * 65)
    initial_capital = 100000.0
    st.sidebar.info(f"Active Quantity: **{fixed_qty} units** ({nifty_lots} Nifty lot)")
else:
    initial_capital = float(st.sidebar.number_input("Starting Capital (₹)", min_value=10000, max_value=10000000, value=100000, step=10000))
    fixed_qty = 0


def compute_kishore_bot(df, key_val, atr_period, ema_len, adx_len=14):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    if len(df) < ema_len + 10:
        return None

    df["EMA"] = df["Close"].ewm(span=ema_len, adjust=False).mean()

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/atr_period, adjust=False).mean()

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_adx = tr.ewm(alpha=1/adx_len, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/adx_len, adjust=False).mean() / tr_adx)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/adx_len, adjust=False).mean() / tr_adx)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=1/adx_len, adjust=False).mean().fillna(0)

    n_loss = key_val * df["ATR"]
    close = df["Close"].values
    nloss = n_loss.values
    stop = np.zeros(len(df))
    stop[0] = close[0] - nloss[0]

    for i in range(1, len(df)):
        if close[i] > stop[i-1] and close[i-1] > stop[i-1]:
            stop[i] = max(stop[i-1], close[i] - nloss[i])
        elif close[i] < stop[i-1] and close[i-1] < stop[i-1]:
            stop[i] = min(stop[i-1], close[i] + nloss[i])
        else:
            stop[i] = close[i] - nloss[i] if close[i] > stop[i-1] else close[i] + nloss[i]

    df["TrailingStop"] = stop

    raw_buy = (df["Close"] > df["TrailingStop"]) & (df["Close"].shift(1) <= df["TrailingStop"].shift(1))
    raw_sell = (df["Close"] < df["TrailingStop"]) & (df["Close"].shift(1) >= df["TrailingStop"].shift(1))

    ema_long_ok = (df["Close"] > df["EMA"]) if use_ema else True
    ema_short_ok = (df["Close"] < df["EMA"]) if use_ema else True
    adx_ok = (df["ADX"] >= adx_min) if use_adx else True

    df["Buy_Signal"] = raw_buy & ema_long_ok & adx_ok
    df["Sell_Signal"] = raw_sell & ema_short_ok & adx_ok
    df["Trend"] = np.where(df["Close"] > df["TrailingStop"], "BULLISH", "BEARISH")
    return df


@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False)


def run_backtest_rupees(df, mode, qty_fixed, start_cap):
    trades = []
    pos = None
    entry_price = 0.0
    entry_date = None
    trade_qty = 0

    equity = start_cap
    peak_equity = start_cap
    max_dd_inr = 0.0
    max_dd_pct = 0.0
    equity_curve = []

    for idx, row in df.iterrows():
        price = float(row["Close"])
        stop = float(row["TrailingStop"])

        if pos == "LONG":
            mtm_equity = equity + (price - entry_price) * trade_qty
        elif pos == "SHORT":
            mtm_equity = equity + (entry_price - price) * trade_qty
        else:
            mtm_equity = equity

        if mtm_equity > peak_equity:
            peak_equity = mtm_equity
        dd_inr = peak_equity - mtm_equity
        dd_pct = (dd_inr / peak_equity * 100) if peak_equity > 0 else 0.0
        if dd_inr > max_dd_inr:
            max_dd_inr = dd_inr
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
        equity_curve.append({"Date": idx, "Equity": mtm_equity, "Peak": peak_equity})

        if pos == "LONG" and (price < stop or row["Sell_Signal"]):
            pnl_inr = (price - entry_price) * trade_qty
            equity += pnl_inr
            trades.append({
                "Type": "LONG", "Entry Date": entry_date, "Entry": round(entry_price, 2),
                "Exit Date": idx, "Exit": round(price, 2), "Qty": trade_qty,
                "P&L (₹)": round(pnl_inr, 2)
            })
            pos = None
            trade_qty = 0
        elif pos == "SHORT" and (price > stop or row["Buy_Signal"]):
            pnl_inr = (entry_price - price) * trade_qty
            equity += pnl_inr
            trades.append({
                "Type": "SHORT", "Entry Date": entry_date, "Entry": round(entry_price, 2),
                "Exit Date": idx, "Exit": round(price, 2), "Qty": trade_qty,
                "P&L (₹)": round(pnl_inr, 2)
            })
            pos = None
            trade_qty = 0

        if pos is None and (row["Buy_Signal"] or row["Sell_Signal"]):
            if mode == "Fixed Qty (Nifty Futures Lot)":
                calc_qty = int(qty_fixed)
            else:
                calc_qty = int(max(1, np.floor(equity / price))) if price > 0 else 1

            if row["Buy_Signal"]:
                pos = "LONG"
                entry_price = price
                entry_date = idx
                trade_qty = calc_qty
            elif row["Sell_Signal"]:
                pos = "SHORT"
                entry_price = price
                entry_date = idx
                trade_qty = calc_qty

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve)

    if trades_df.empty:
        return trades_df, eq_df, 0, 0.0, 0.0, 0.0, 0.0

    total_trades = len(trades_df)
    win_rate = (trades_df["P&L (₹)"] > 0).mean() * 100
    net_pnl_inr = trades_df["P&L (₹)"].sum()
    return (
        trades_df,
        eq_df,
        total_trades,
        round(win_rate, 2),
        round(net_pnl_inr, 2),
        round(max_dd_inr, 2),
        round(max_dd_pct, 2),
    )


tab1, tab2 = st.tabs(["📡 Kishore Bot Live Scanner (28 Assets)", "📊 Rupee Backtester & Max Drawdown"])

with tab1:
    st.subheader(f"Kishore Bot Live Scan ({timeframe} Timeframe)")
    if st.button("🔄 Scan All Indices & Stocks Now"):
        scan_rows = []
        progress = st.progress(0)
        items = list(ASSETS.items())
        for i, (name, ticker) in enumerate(items):
            raw_df = fetch_data(ticker, period, timeframe)
            res_df = compute_kishore_bot(raw_df, key_val, atr_period, ema_len)
            if res_df is not None:
                last = res_df.iloc[-1]
                prev = res_df.iloc[-2]
                signal = "⚪ HOLD"
                if last["Buy_Signal"] or prev["Buy_Signal"]:
                    signal = "🟢 FRESH BUY"
                elif last["Sell_Signal"] or prev["Sell_Signal"]:
                    signal = "🔴 FRESH SELL"

                scan_rows.append({
                    "Asset": name,
                    "Ticker": ticker,
                    "Close": round(float(last["Close"]), 2),
                    "200 EMA": round(float(last["EMA"]), 2),
                    "EMA Status": "Above 200 EMA" if last["Close"] > last["EMA"] else "Below 200 EMA",
                    "ADX": round(float(last["ADX"]), 1),
                    "Kishore Stop": round(float(last["TrailingStop"]), 2),
                    "Trend": last["Trend"],
                    "Signal (Last 2 Bars)": signal
                })
            progress.progress((i + 1) / len(items))

        st.dataframe(pd.DataFrame(scan_rows), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Rupee P&L & Max Drawdown Backtest")
    selected_asset = st.selectbox("Select Index or Stock to Backtest", list(ASSETS.keys()), index=0)
    ticker = ASSETS[selected_asset]

    raw_df = fetch_data(ticker, period, timeframe)
    bt_df = compute_kishore_bot(raw_df, key_val, atr_period, ema_len)

    if bt_df is not None:
        trades_df, eq_df, total_trades, win_rate, net_pnl_inr, max_dd_inr, max_dd_pct = run_backtest_rupees(
            bt_df, sizing_mode, fixed_qty, initial_capital
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Profit / Loss (₹)", f"₹{net_pnl_inr:,.2f}")
        c2.metric("Max Drawdown (₹)", f"₹{max_dd_inr:,.2f}", delta=f"-{max_dd_pct}%", delta_color="inverse")
        c3.metric("Win Rate (%)", f"{win_rate}%")
        c4.metric("Total Filtered Trades", total_trades)

        if not eq_df.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq_df["Date"], y=eq_df["Equity"], mode="lines", name="Equity (₹)", line=dict(color="#00e676", width=2)))
            fig_eq.add_trace(go.Scatter(x=eq_df["Date"], y=eq_df["Peak"], mode="lines", name="Peak Equity (₹)", line=dict(color="#ffa726", width=1.2, dash="dash")))
            fig_eq.update_layout(height=320, template="plotly_dark", title="Equity Curve & Peak Drawdown Tracking (₹)", yaxis_title="Amount (₹)")
            st.plotly_chart(fig_eq, use_container_width=True)

        plot_df = bt_df.tail(150)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"], name="Price"
        ))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA"], line=dict(color="orange", width=2), name="200 EMA"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["TrailingStop"], line=dict(color="cyan", width=1.5), name="Kishore Bot Stop"))

        buys = plot_df[plot_df["Buy_Signal"]]
        sells = plot_df[plot_df["Sell_Signal"]]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Low"]*0.995, mode="markers", marker=dict(symbol="triangle-up", size=12, color="lime"), name="Kishore Bot BUY"))
        fig.add_trace(go.Scatter(x=sells.index, y=sells["High"]*1.005, mode="markers", marker=dict(symbol="triangle-down", size=12, color="red"), name="Kishore Bot SELL"))

        fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", title=f"{selected_asset} ({timeframe}) - Kishore Bot Signals")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📜 Backtest Trade Log (Rupee P&L)")
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Not enough historical bars to calculate 200 EMA for this timeframe.")
