import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import numpy as np
from scipy.interpolate import interp1d

# --- 0. QUANT CONFIGURATION ---
REFRESH_RATE = 15 
SPOT_INSTRUMENT = "BTC_USDC"
MIN_DAYS_THRESHOLD = 7.0  # Crucial: We ignore the last week to prevent gamma noise/math singularities

st.set_page_config(page_title="HFT Basis Quant", layout="wide")

# --- 1. DATA LAYER: BENCHMARK ---
@st.cache_data
def load_benchmark():
    try:
        df = pd.read_csv("benchmark_hft.csv")
        df = df.sort_values("Days_Left", ascending=False)
        # Filter benchmark to match our threshold
        df = df[df['Days_Left'] >= MIN_DAYS_THRESHOLD]
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 2. DATA LAYER: HISTORY (Context) ---
@st.cache_data(ttl=600) 
def get_curve_history():
    url = "https://www.deribit.com/api/v2/public/get_instruments"
    tv_url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    
    try:
        resp = requests.get(url, params={"currency": "BTC", "kind": "future", "expired": "false"}).json()['result']
        futures = [i for i in resp if i['settlement_period'] != 'perpetual']
    except: return pd.DataFrame()

    live_rows = []
    now_ts = int(time.time() * 1000)
    start_ts = now_ts - (180 * 24 * 60 * 60 * 1000) 

    for f in futures:
        try:
            # Fetch
            p_fut = {"instrument_name": f['instrument_name'], "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "60"}
            p_spot = {"instrument_name": SPOT_INSTRUMENT, "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "60"}
            
            d_fut = requests.get(tv_url, params=p_fut).json()
            d_spot = requests.get(tv_url, params=p_spot).json()
            
            if d_fut['result']['status'] == 'no_data' or d_spot['result']['status'] == 'no_data': continue

            # Merge
            df_f = pd.DataFrame(d_fut['result'])[['ticks', 'close']].rename(columns={'close': 'Future', 'ticks': 'Timestamp'})
            df_s = pd.DataFrame(d_spot['result'])[['ticks', 'close']].rename(columns={'close': 'Spot', 'ticks': 'Timestamp'})
            df = pd.merge(df_f, df_s, on='Timestamp', how='inner')
            
            # Calculations
            expiry_ts = f['expiration_timestamp']
            expiry_date = datetime.fromtimestamp(expiry_ts/1000)
            
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df['Days_Left'] = (expiry_date - df['Date']).dt.total_seconds() / (24 * 3600)
            
            # Filter: Apply the Gamma Cutoff
            df = df[(df['Days_Left'] >= MIN_DAYS_THRESHOLD) & (df['Days_Left'] <= 180)]
            
            if not df.empty:
                df['Basis_Pct'] = (df['Future'] - df['Spot']) / df['Spot']
                df['APY'] = df['Basis_Pct'] * (365 / df['Days_Left']) * 100
                df['Contract'] = f['instrument_name']
                df['Type'] = 'History' 
                live_rows.append(df)
        except: continue
        
    if live_rows: return pd.concat(live_rows)
    return pd.DataFrame()

# --- 3. DATA LAYER: LIVE TICKER (Signal) ---
def get_live_snapshot(active_contracts):
    snapshot_rows = []
    ticker_url = "https://www.deribit.com/api/v2/public/ticker"
    
    try:
        s_resp = requests.get(ticker_url, params={"instrument_name": SPOT_INSTRUMENT}).json()['result']
        spot_mid = (s_resp['best_bid_price'] + s_resp['best_ask_price']) / 2
    except: return pd.DataFrame()

    for contract in active_contracts:
        try:
            f_resp = requests.get(ticker_url, params={"instrument_name": contract}).json()['result']
            fut_mid = (f_resp['best_bid_price'] + f_resp['best_ask_price']) / 2
            
            expiry_ts = f_resp['expiration_timestamp']
            days_left = (datetime.fromtimestamp(expiry_ts/1000) - datetime.now()).total_seconds() / (24 * 3600)
            
            # Filter: Apply Gamma Cutoff to live data too
            if days_left < MIN_DAYS_THRESHOLD: continue

            apy = ((fut_mid - spot_mid) / spot_mid) * (365 / days_left) * 100
            
            snapshot_rows.append({
                "Timestamp": int(time.time()*1000),
                "Date": datetime.now(),
                "Future": fut_mid,
                "Spot": spot_mid,
                "Days_Left": days_left,
                "APY": apy,
                "Contract": contract,
                "Type": "Live"
            })
        except: continue
        
    return pd.DataFrame(snapshot_rows)

# --- 4. ENGINE: Z-SCORE ---
def calculate_quant_signals(live_df, bench_df):
    if bench_df.empty or live_df.empty: return live_df
    
    bench_df = bench_df.sort_values("Days_Left")
    
    # Strict Extrapolation to handle edge cases
    f_med = interp1d(bench_df['Days_Left'], bench_df['Median_APY'], fill_value="extrapolate")
    f_q1 = interp1d(bench_df['Days_Left'], bench_df['Q1_APY'], fill_value="extrapolate")
    f_q3 = interp1d(bench_df['Days_Left'], bench_df['Q3_APY'], fill_value="extrapolate")
    
    live_df['Exp_Median'] = f_med(live_df['Days_Left'])
    live_df['Exp_Q1'] = f_q1(live_df['Days_Left'])
    live_df['Exp_Q3'] = f_q3(live_df['Days_Left'])
    
    live_df['IQR'] = live_df['Exp_Q3'] - live_df['Exp_Q1']
    live_df['Sigma_Proxy'] = live_df['IQR'] / 1.35
    live_df['Sigma_Proxy'] = live_df['Sigma_Proxy'].replace(0, 1) # Safety
    
    live_df['Z_Score'] = (live_df['APY'] - live_df['Exp_Median']) / live_df['Sigma_Proxy']
    
    return live_df

# --- 5. VISUALIZATION ---
st.title("🛡️ Systematic Basis Engine")

bench_df = load_benchmark()
history_df = get_curve_history()

if history_df.empty:
    st.info("System Initializing...")
    time.sleep(2)
    st.rerun()

active_contracts = history_df['Contract'].unique()
live_tip_df = get_live_snapshot(active_contracts)

full_df = pd.concat([history_df, live_tip_df], ignore_index=True)
scored_df = calculate_quant_signals(full_df, bench_df)
latest_scores = scored_df[scored_df['Type'] == 'Live'].sort_values("Days_Left")

# --- HEATMAP ---
st.subheader("1. Live Z-Score Matrix")
if not latest_scores.empty:
    cols = st.columns(len(latest_scores))
    for i, (idx, row) in enumerate(latest_scores.iterrows()):
        c_name = row['Contract'].replace("BTC-", "")
        z = row['Z_Score']
        val = row['APY']
        
        # Quant Color Logic
        color = "off"
        if z > 2.0: color = "inverse" # Sell Signal
        elif z < -2.0: color = "normal" # Buy Signal
        
        with cols[i]:
            st.metric(label=c_name, value=f"{val:.2f}%", delta=f"{z:.2f} σ", delta_color=color)
else:
    st.warning(f"No contracts found outside the {MIN_DAYS_THRESHOLD}-day volatility cutoff.")

# --- CHART ---
st.subheader("2. Yield Term Structure")
fig = go.Figure()

# Benchmark Tunnel
if not bench_df.empty:
    fig.add_trace(go.Scatter(
        x=pd.concat([bench_df['Days_Left'], bench_df['Days_Left'][::-1]]),
        y=pd.concat([bench_df['Q3_APY'], bench_df['Q1_APY'][::-1]]),
        fill='toself', fillcolor='rgba(0, 255, 255, 0.05)', line=dict(width=0),
        hoverinfo="skip", name='Normal Regime (IQR)'
    ))
    fig.add_trace(go.Scatter(
        x=bench_df['Days_Left'], y=bench_df['Median_APY'],
        mode='lines', line=dict(color='cyan', width=1, dash='dash'), name='Fair Value'
    ))

# Active Contracts (Grouped)
colors = px.colors.qualitative.Bold
for i, c in enumerate(active_contracts):
    # Common color for this contract
    c_color = colors[i % len(colors)]
    
    # History Line
    hist = scored_df[(scored_df['Contract'] == c) & (scored_df['Type'] == 'History')]
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist['Days_Left'], y=hist['APY'],
            mode='lines', line=dict(color=c_color, width=2),
            name=c, legendgroup=c 
        ))
    
    # Live Dot
    tip = scored_df[(scored_df['Contract'] == c) & (scored_df['Type'] == 'Live')]
    if not tip.empty:
        fig.add_trace(go.Scatter(
            x=tip['Days_Left'], y=tip['APY'],
            mode='markers', marker=dict(color=c_color, size=12, symbol="diamond", line=dict(width=1, color='white')),
            name=c, legendgroup=c, showlegend=False 
        ))

fig.update_layout(
    xaxis_title="Days Until Expiration", yaxis_title="Annualized Yield (%)",
    xaxis=dict(autorange="reversed"), height=700, hovermode="closest",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Cutoff: {MIN_DAYS_THRESHOLD} Days | Model: Robust Z-Score")

time.sleep(REFRESH_RATE)
st.rerun()
