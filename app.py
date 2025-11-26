import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime

# --- CONFIG ---
REFRESH_RATE = 60 # Slower refresh to handle heavier historical data load
MIN_DAYS = 2.0  
SPOT_EXEC_BTC = "BTC_USDC" 

st.set_page_config(page_title="HFT Relative Value Desk", layout="wide")

# --- 1. DATA LAYER: LIVE SNAPSHOTS ---
def get_yield_curve(currency, index_name):
    url_ticker = "https://www.deribit.com/api/v2/public/ticker"
    url_index = "https://www.deribit.com/api/v2/public/get_index_price"
    url_inst = "https://www.deribit.com/api/v2/public/get_instruments"
    
    try:
        i_resp = requests.get(url_index, params={"index_name": index_name}).json()['result']
        index_price = i_resp['index_price']
    except: return pd.DataFrame(), 0

    try:
        futures = requests.get(url_inst, params={"currency": currency, "kind": "future", "expired": "false"}).json()['result']
        futures = [f for f in futures if f['settlement_period'] != 'perpetual']
    except: return pd.DataFrame(), index_price

    rows = []
    for f in futures:
        try:
            t_resp = requests.get(url_ticker, params={"instrument_name": f['instrument_name']}).json()['result']
            mid_price = (t_resp['best_bid_price'] + t_resp['best_ask_price']) / 2
            
            expiry_ts = f['expiration_timestamp']
            days_left = (datetime.fromtimestamp(expiry_ts/1000) - datetime.now()).total_seconds() / (24 * 3600)
            
            if days_left < MIN_DAYS: continue

            basis_pct = (mid_price - index_price) / index_price
            apy = basis_pct * (365 / days_left) * 100
            
            date_key = f['instrument_name'].split("-")[1]

            rows.append({
                "Contract": f['instrument_name'],
                "Date_Key": date_key,
                "Days_Left": days_left,
                "APY": apy,
                "Price": mid_price
            })
        except: continue
        
    return pd.DataFrame(rows), index_price

def get_perp_data():
    try:
        url = "https://www.deribit.com/api/v2/public/ticker"
        p_resp = requests.get(url, params={"instrument_name": "BTC-PERPETUAL"}).json()['result']
        return p_resp['current_funding'] * 3 * 365 * 100
    except: return 0

def get_btc_spot_exec():
    try:
        url = "https://www.deribit.com/api/v2/public/ticker"
        s_resp = requests.get(url, params={"instrument_name": SPOT_EXEC_BTC}).json()['result']
        return (s_resp['best_bid_price'] + s_resp['best_ask_price']) / 2
    except: return 0

# --- 2. DATA LAYER: SPREAD HISTORY (The Time Series) ---
@st.cache_data(ttl=3600) # Cache for 1 hour strictly
def get_spread_history(date_key):
    """
    Fetches 30-day hourly history for BTC vs ETH for a specific expiry date.
    """
    btc_instr = f"BTC-{date_key}"
    eth_instr = f"ETH-{date_key}"
    
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    now_ts = int(time.time() * 1000)
    start_ts = now_ts - (30 * 24 * 60 * 60 * 1000) # 30 Days
    
    # 1. Fetch Futures Prices
    try:
        f_btc = requests.get(url, params={"instrument_name": btc_instr, "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "60"}).json()['result']
        f_eth = requests.get(url, params={"instrument_name": eth_instr, "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "60"}).json()['result']
        
        # 2. Fetch Index Prices (Required for accurate APY calc)
        i_btc = requests.get(url, params={"instrument_name": "btc_usd", "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "60"}).json()['result']
        i_eth = requests.get(url, params={"instrument_name": "eth_usd", "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "60"}).json()['result']
    except: return pd.DataFrame()

    if f_btc['status'] == 'no_data' or f_eth['status'] == 'no_data': return pd.DataFrame()

    # 3. Create DataFrames
    df_btc = pd.DataFrame({'Ts': f_btc['ticks'], 'Fut_BTC': f_btc['close']})
    df_eth = pd.DataFrame({'Ts': f_eth['ticks'], 'Fut_ETH': f_eth['close']})
    df_ibtc = pd.DataFrame({'Ts': i_btc['ticks'], 'Idx_BTC': i_btc['close']})
    df_ieth = pd.DataFrame({'Ts': i_eth['ticks'], 'Idx_ETH': i_eth['close']})
    
    # 4. Merge Everything
    df = pd.merge(df_btc, df_eth, on='Ts', how='inner')
    df = pd.merge(df, df_ibtc, on='Ts', how='inner')
    df = pd.merge(df, df_ieth, on='Ts', how='inner')
    
    # 5. Calculate Historical APY & Spread
    # We need the Expiry Timestamp to calculate 'Days Left' for every historical hour
    # We can approximate expiry from the Date Key (e.g. 27MAR26)
    try:
        expiry_dt = datetime.strptime(date_key, "%d%b%y")
    except: return pd.DataFrame() # Fail safe
    
    df['Date'] = pd.to_datetime(df['Ts'], unit='ms')
    df['Days_Left'] = (expiry_dt - df['Date']).dt.total_seconds() / (24 * 3600)
    
    # Filter out near-expiry noise if historical data covers it
    df = df[df['Days_Left'] > 7]

    # APY = ((Fut - Idx) / Idx) * (365/Days)
    df['APY_BTC'] = ((df['Fut_BTC'] - df['Idx_BTC']) / df['Idx_BTC']) * (365 / df['Days_Left']) * 100
    df['APY_ETH'] = ((df['Fut_ETH'] - df['Idx_ETH']) / df['Idx_ETH']) * (365 / df['Days_Left']) * 100
    
    df['Spread_BPS'] = (df['APY_ETH'] - df['APY_BTC']) * 100
    
    return df

# --- 3. MATH LAYER: FITS & BANDS ---
def fit_curves(df):
    if df.empty or len(df) < 3: return None, None
    x = df['Days_Left'].values
    y = df['APY'].values
    try:
        coeffs = np.polyfit(np.log(x), y, 1)
        return coeffs[0] * np.log(x) + coeffs[1], None
    except: return y, None

# --- 4. VISUALIZATION ---
st.title("⚡ HFT Relative Value Desk")

btc_df, btc_index = get_yield_curve("BTC", "btc_usd")
eth_df, eth_index = get_yield_curve("ETH", "eth_usd")
btc_perp_apy = get_perp_data()
btc_exec = get_btc_spot_exec()

if btc_df.empty:
    st.error("Connecting to Exchange...")
    time.sleep(2)
    st.rerun()

# Header
dislocation = btc_exec - btc_index
c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC Index", f"${btc_index:,.2f}")
c2.metric("BTC Spot Exec", f"${btc_exec:,.2f}", delta=f"{dislocation:.2f}", delta_color="inverse")
c3.metric("ETH Index", f"${eth_index:,.2f}")
c4.metric("Spread Dislocation", f"{dislocation:.2f}", delta="Warning" if abs(dislocation)>10 else "Normal", delta_color="off")
st.markdown("---")

# Data Prep
btc_df = btc_df.sort_values("Days_Left")
log_y, _ = fit_curves(btc_df)
if log_y is not None:
    btc_df['Fair_Log'] = log_y
    btc_df['Residual'] = btc_df['APY'] - btc_df['Fair_Log']

# --- ROW 1: STRUCTURE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. BTC Curve Residuals (Spatial)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=btc_df['Days_Left'], y=btc_df['APY'],
        mode='markers+text', text=btc_df['Date_Key'], textposition="top center",
        marker=dict(size=14, color=btc_df['Residual'], colorscale="RdBu_r", cmin=-2, cmax=2),
        name='BTC Live'
    ))
    if log_y is not None:
        fig1.add_trace(go.Scatter(x=btc_df['Days_Left'], y=btc_df['Fair_Log'], mode='lines', line=dict(color='cyan', dash='dash'), name='Stiff Fit'))
    fig1.update_layout(xaxis=dict(autorange="reversed"), height=400, xaxis_title="Days to Expiry", yaxis_title="BTC APY (%)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("2. Basis Spread Snapshot (ETH - BTC)")
    spread_df = pd.merge(eth_df, btc_df, on="Date_Key", suffixes=("_ETH", "_BTC"))
    if not spread_df.empty:
        spread_df['Spread_BPS'] = (spread_df['APY_ETH'] - spread_df['APY_BTC']) * 100
        spread_df = spread_df.sort_values("Days_Left_BTC")
        colors = ['#00FF00' if x > 0 else '#FF4444' for x in spread_df['Spread_BPS']]
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=spread_df['Date_Key'], y=spread_df['Spread_BPS'], marker_color=colors, text=spread_df['Spread_BPS'].apply(lambda x: f"{x:.0f} bps")))
        fig3.update_layout(height=400, yaxis_title="Spread (bps)")
        fig3.add_hline(y=0, line_color="white", opacity=0.3)
        st.plotly_chart(fig3, use_container_width=True)

# --- ROW 2: TIME SERIES ---
st.subheader("3. Spread Trend Monitor (30-Day History)")

# Automatic Selection: Pick the first liquid contract > 30 Days out
valid_contracts = spread_df[spread_df['Days_Left_BTC'] > 30]
if not valid_contracts.empty:
    target_contract = valid_contracts.iloc[0]['Date_Key'] # e.g. "27MAR26"
    
    st.caption(f"Analyzing Benchmark Pair: **BTC-{target_contract} vs ETH-{target_contract}**")
    
    hist_df = get_spread_history(target_contract)
    
    if not hist_df.empty:
        # Bollinger Bands Calc
        hist_df['MA_7D'] = hist_df['Spread_BPS'].rolling(window=24*7).mean()
        hist_df['STD_7D'] = hist_df['Spread_BPS'].rolling(window=24*7).std()
        hist_df['Upper'] = hist_df['MA_7D'] + (2 * hist_df['STD_7D'])
        hist_df['Lower'] = hist_df['MA_7D'] - (2 * hist_df['STD_7D'])
        
        fig_ts = go.Figure()
        
        # Bands
        fig_ts.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Upper'], line=dict(width=0), showlegend=False, name='Upper'))
        fig_ts.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Lower'], line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', showlegend=False, name='Lower'))
        
        # Main Line
        fig_ts.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Spread_BPS'], line=dict(color='yellow', width=1.5), name='Spread (Hourly)'))
        fig_ts.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['MA_7D'], line=dict(color='cyan', width=2, dash='dash'), name='7D Trend'))
        
        fig_ts.update_layout(height=450, yaxis_title="Spread (bps)", title=f"Historical Spread: {target_contract}")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Loading historical data... (First run may take 10s)")
else:
    st.warning("No valid contracts > 30 days found for trend analysis.")

time.sleep(REFRESH_RATE)
st.rerun()
