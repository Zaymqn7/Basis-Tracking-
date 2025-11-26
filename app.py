import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import numpy as np

# --- 0. QUANT CONFIGURATION ---
# "HFT" Refresh Rate: We pull the live ticker every 15 seconds. 
# Pulling faster (e.g. 1s) on a public web-app will get you IP banned by Deribit.
REFRESH_RATE = 15 
SPOT_INSTRUMENT = "BTC_USDC"

st.set_page_config(page_title="HFT Basis Quant", layout="wide")

# --- 1. DATA LAYER: BENCHMARK (The Map) ---
@st.cache_data
def load_benchmark():
    try:
        df = pd.read_csv("benchmark_hft.csv")
        # Interpolation function for precise Z-Score lookups later
        df = df.sort_values("Days_Left", ascending=False)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 2. DATA LAYER: HISTORICAL CONTEXT (The Body) ---
# We cache this for 10 minutes. The "Body" of the snake doesn't need to update every second.
@st.cache_data(ttl=600) 
def get_curve_history():
    url = "https://www.deribit.com/api/v2/public/get_instruments"
    tv_url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    
    # A. Get Active Futures
    try:
        resp = requests.get(url, params={"currency": "BTC", "kind": "future", "expired": "false"}).json()['result']
        futures = [i for i in resp if i['settlement_period'] != 'perpetual']
    except: return pd.DataFrame()

    live_rows = []
    now_ts = int(time.time() * 1000)
    # Get last 180 days of HOURLY data (matches benchmark resolution)
    start_ts = now_ts - (180 * 24 * 60 * 60 * 1000) 

    for f in futures:
        try:
            # 1. Fetch Future & Spot (Hourly)
            p_fut = {"instrument_name": f['instrument_name'], "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "60"}
            p_spot = {"instrument_name": SPOT_INSTRUMENT, "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "60"}
            
            d_fut = requests.get(tv_url, params=p_fut).json()
            d_spot = requests.get(tv_url, params=p_spot).json()
            
            if d_fut['result']['status'] == 'no_data' or d_spot['result']['status'] == 'no_data': continue

            # 2. Merge
            df_f = pd.DataFrame(d_fut['result'])[['ticks', 'close']].rename(columns={'close': 'Future', 'ticks': 'Timestamp'})
            df_s = pd.DataFrame(d_spot['result'])[['ticks', 'close']].rename(columns={'close': 'Spot', 'ticks': 'Timestamp'})
            df = pd.merge(df_f, df_s, on='Timestamp', how='inner')
            
            # 3. Calc APY
            expiry_ts = f['expiration_timestamp']
            expiry_date = datetime.fromtimestamp(expiry_ts/1000)
            
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df['Days_Left'] = (expiry_date - df['Date']).dt.total_seconds() / (24 * 3600)
            
            df['Basis_Pct'] = (df['Future'] - df['Spot']) / df['Spot']
            df['APY'] = df['Basis_Pct'] * (365 / df['Days_Left']) * 100
            
            df['Contract'] = f['instrument_name']
            df['Type'] = 'History' # Tag as historical candle
            
            # Filter valid range
            df = df[(df['Days_Left'] > 0.5) & (df['Days_Left'] <= 180)]
            live_rows.append(df)
        except: continue
        
    if live_rows: return pd.concat(live_rows)
    return pd.DataFrame()

# --- 3. DATA LAYER: LIVE TICKER (The Head) ---
# This runs every refresh cycle (15s). Light and fast.
def get_live_snapshot(active_contracts):
    snapshot_rows = []
    ticker_url = "https://www.deribit.com/api/v2/public/ticker"
    
    # 1. Get Live Spot Mid-Price
    try:
        s_resp = requests.get(ticker_url, params={"instrument_name": SPOT_INSTRUMENT}).json()['result']
        spot_mid = (s_resp['best_bid_price'] + s_resp['best_ask_price']) / 2
    except: return pd.DataFrame()

    # 2. Get Live Future Mid-Prices
    for contract in active_contracts:
        try:
            f_resp = requests.get(ticker_url, params={"instrument_name": contract}).json()['result']
            fut_mid = (f_resp['best_bid_price'] + f_resp['best_ask_price']) / 2
            
            # Calc Time
            expiry_ts = f_resp['expiration_timestamp']
            days_left = (datetime.fromtimestamp(expiry_ts/1000) - datetime.now()).total_seconds() / (24 * 3600)
            
            # Calc Instant APY
            apy = ((fut_mid - spot_mid) / spot_mid) * (365 / days_left) * 100
            
            snapshot_rows.append({
                "Timestamp": int(time.time()*1000),
                "Date": datetime.now(),
                "Future": fut_mid,
                "Spot": spot_mid,
                "Days_Left": days_left,
                "APY": apy,
                "Contract": contract,
                "Type": "Live" # Tag as live tip
            })
        except: continue
        
    return pd.DataFrame(snapshot_rows)

# --- 4. ENGINE: Z-SCORE CALCULATION ---
def calculate_quant_signals(live_df, bench_df):
    if bench_df.empty or live_df.empty: return live_df
    
    # Sort benchmark for interpolation
    bench_df = bench_df.sort_values("Days_Left")
    
    # Create Interpolation Functions
    from scipy.interpolate import interp1d
    # We use linear interpolation to find the exact Median/Q1/Q3 for fractional days (e.g. 45.3 days)
    f_med = interp1d(bench_df['Days_Left'], bench_df['Median_APY'], fill_value="extrapolate")
    f_q1 = interp1d(bench_df['Days_Left'], bench_df['Q1_APY'], fill_value="extrapolate")
    f_q3 = interp1d(bench_df['Days_Left'], bench_df['Q3_APY'], fill_value="extrapolate")
    
    # Apply to Live Data
    live_df['Exp_Median'] = f_med(live_df['Days_Left'])
    live_df['Exp_Q1'] = f_q1(live_df['Days_Left'])
    live_df['Exp_Q3'] = f_q3(live_df['Days_Left'])
    
    # Calculate Robust Z-Score (using IQR)
    # Standard Normal Dist: IQR approx 1.35 * Sigma
    live_df['IQR'] = live_df['Exp_Q3'] - live_df['Exp_Q1']
    live_df['Sigma_Proxy'] = live_df['IQR'] / 1.35
    
    # Avoid div by zero
    live_df['Sigma_Proxy'] = live_df['Sigma_Proxy'].replace(0, 1)
    
    live_df['Z_Score'] = (live_df['APY'] - live_df['Exp_Median']) / live_df['Sigma_Proxy']
    
    return live_df

# --- 5. VISUALIZATION ---
st.title("🛡️ Systematic Basis Engine (Hourly + RealTime)")

# Load Data
bench_df = load_benchmark()
history_df = get_curve_history()

if history_df.empty:
    st.warning("Initializing data...")
    time.sleep(2)
    st.rerun()

# Get Live Snapshot
active_contracts = history_df['Contract'].unique()
live_tip_df = get_live_snapshot(active_contracts)

# STITCH: Combine History + Live Tip
full_df = pd.concat([history_df, live_tip_df], ignore_index=True)

# Run Quant Stats
scored_df = calculate_quant_signals(full_df, bench_df)
latest_scores = scored_df[scored_df['Type'] == 'Live'].sort_values("Days_Left")

# --- DASHBOARD LAYOUT ---

# Top Metric: Z-Score Heatmap
st.subheader("1. Live Signal Matrix (Z-Scores)")
cols = st.columns(len(latest_scores))
for i, (idx, row) in enumerate(latest_scores.iterrows()):
    contract_clean = row['Contract'].replace("BTC-", "")
    z = row['Z_Score']
    
    # Color Logic
    color = "off"
    if z > 1.5: color = "normal" # Rich
    if z > 2.5: color = "inverse" # Very Rich (Short Signal)
    if z < -1.5: color = "off" # Cheap
    
    with cols[i]:
        st.metric(
            label=contract_clean, 
            value=f"{row['APY']:.2f}%", 
            delta=f"Z: {z:.2f}",
            delta_color=color
        )

# Main Chart
st.subheader("2. Term Structure (Hybrid Resolution)")
fig = go.Figure()

# A. Draw Tunnel (Benchmark)
if not bench_df.empty:
    fig.add_trace(go.Scatter(
        x=pd.concat([bench_df['Days_Left'], bench_df['Days_Left'][::-1]]),
        y=pd.concat([bench_df['Q3_APY'], bench_df['Q1_APY'][::-1]]),
        fill='toself', fillcolor='rgba(0, 255, 255, 0.08)', line=dict(width=0),
        hoverinfo="skip", name='Fair Value Tunnel'
    ))
    fig.add_trace(go.Scatter(
        x=bench_df['Days_Left'], y=bench_df['Median_APY'],
        mode='lines', line=dict(color='cyan', width=1, dash='dash'), name='Median'
    ))

# B. Draw Live Curves
colors = px.colors.qualitative.Bold
for i, c in enumerate(active_contracts):
    # Plot History (Solid Line)
    hist = scored_df[(scored_df['Contract'] == c) & (scored_df['Type'] == 'History')]
    fig.add_trace(go.Scatter(
        x=hist['Days_Left'], y=hist['APY'],
        mode='lines', line=dict(color=colors[i % len(colors)], width=2),
        name=c, showlegend=False
    ))
    
    # Plot Live Tip (Pulsing Dot)
    tip = scored_df[(scored_df['Contract'] == c) & (scored_df['Type'] == 'Live')]
    if not tip.empty:
        fig.add_trace(go.Scatter(
            x=tip['Days_Left'], y=tip['APY'],
            mode='markers', marker=dict(color=colors[i % len(colors)], size=10, symbol="diamond"),
            name=f"{c} (Live)", showlegend=True
        ))

fig.update_layout(
    xaxis_title="Days Until Expiration", yaxis_title="Annualized Yield (%)",
    xaxis=dict(autorange="reversed"), height=650, hovermode="closest"
)
st.plotly_chart(fig, use_container_width=True)

# Footer info
st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')} | Refresh Rate: {REFRESH_RATE}s | Model: Robust Z-Score (IQR)")

time.sleep(REFRESH_RATE)
st.rerun()
