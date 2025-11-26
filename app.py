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
REFRESH_RATE = 60 
SPOT_INSTRUMENT = "BTC_USDC"
MIN_DAYS_THRESHOLD = 7.0  # HARD FILTER: Contracts expiring within 7 days are BLACKLISTED

st.set_page_config(page_title="HFT Basis Quant (Stable)", layout="wide")

# --- 1. DATA LAYER: BENCHMARK ---
@st.cache_data
def load_benchmark():
    try:
        df = pd.read_csv("benchmark_hft.csv")
        df = df.sort_values("Days_Left", ascending=False)
        df = df[df['Days_Left'] >= MIN_DAYS_THRESHOLD]
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 2. DATA LAYER: HOURLY HISTORY ---
@st.cache_data(ttl=300) 
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
            # --- CRITICAL FIX: CONTRACT LEVEL FILTER ---
            # Check expiry BEFORE fetching data. 
            expiry_ts = f['expiration_timestamp']
            days_until_expiry = (datetime.fromtimestamp(expiry_ts/1000) - datetime.now()).total_seconds() / (24 * 3600)
            
            # If the snake is too close to the wall, kill it immediately.
            if days_until_expiry < MIN_DAYS_THRESHOLD:
                continue 

            # Fetch Future & Spot
            p_fut = {"instrument_name": f['instrument_name'], "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "60"}
            p_spot = {"instrument_name": SPOT_INSTRUMENT, "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "60"}
            
            d_fut = requests.get(tv_url, params=p_fut).json()
            d_spot = requests.get(tv_url, params=p_spot).json()
            
            if 'result' not in d_fut or d_fut['result']['status'] == 'no_data': continue
            if 'result' not in d_spot or d_spot['result']['status'] == 'no_data': continue

            # Merge
            df_f = pd.DataFrame(d_fut['result'])[['ticks', 'close']].rename(columns={'close': 'Future', 'ticks': 'Timestamp'})
            df_s = pd.DataFrame(d_spot['result'])[['ticks', 'close']].rename(columns={'close': 'Spot', 'ticks': 'Timestamp'})
            df = pd.merge(df_f, df_s, on='Timestamp', how='inner')
            
            # Calculations
            expiry_date = datetime.fromtimestamp(expiry_ts/1000)
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df['Days_Left'] = (expiry_date - df['Date']).dt.total_seconds() / (24 * 3600)
            
            # Double check: Filter rows too (for the history part)
            df = df[(df['Days_Left'] >= MIN_DAYS_THRESHOLD) & (df['Days_Left'] <= 180)]
            
            if not df.empty:
                df['Basis_Pct'] = (df['Future'] - df['Spot']) / df['Spot']
                df['APY'] = df['Basis_Pct'] * (365 / df['Days_Left']) * 100
                df['Contract'] = f['instrument_name']
                live_rows.append(df)
        except: continue
        
    if live_rows: return pd.concat(live_rows)
    return pd.DataFrame()

# --- 3. ENGINE: Z-SCORE ---
def calculate_quant_signals(live_df, bench_df):
    if bench_df.empty or live_df.empty: return live_df
    
    bench_df = bench_df.sort_values("Days_Left")
    
    f_med = interp1d(bench_df['Days_Left'], bench_df['Median_APY'], fill_value="extrapolate")
    f_q1 = interp1d(bench_df['Days_Left'], bench_df['Q1_APY'], fill_value="extrapolate")
    f_q3 = interp1d(bench_df['Days_Left'], bench_df['Q3_APY'], fill_value="extrapolate")
    
    live_df['Exp_Median'] = f_med(live_df['Days_Left'])
    live_df['Exp_Q1'] = f_q1(live_df['Days_Left'])
    live_df['Exp_Q3'] = f_q3(live_df['Days_Left'])
    
    live_df['IQR'] = live_df['Exp_Q3'] - live_df['Exp_Q1']
    live_df['Sigma_Proxy'] = live_df['IQR'] / 1.35
    live_df['Sigma_Proxy'] = live_df['Sigma_Proxy'].replace(0, 1) 
    
    live_df['Z_Score'] = (live_df['APY'] - live_df['Exp_Median']) / live_df['Sigma_Proxy']
    
    return live_df

# --- 4. VISUALIZATION ---
st.title("🛡️ Systematic Basis Engine (Strict Filter)")

bench_df = load_benchmark()
full_df = get_curve_history()

if full_df.empty:
    st.info("No contracts found meeting the > 7 days safety criteria.")
    st.stop()

scored_df = calculate_quant_signals(full_df, bench_df)

# Extract Latest Signal
latest_scores = scored_df.sort_values('Timestamp').groupby('Contract').tail(1).sort_values("Days_Left")

# --- HEATMAP ---
st.subheader(f"1. Regime Scoreboard (> {int(MIN_DAYS_THRESHOLD)} Days)")

if not latest_scores.empty:
    cols = st.columns(len(latest_scores))
    for i, (idx, row) in enumerate(latest_scores.iterrows()):
        c_name = row['Contract'].replace("BTC-", "")
        z = row['Z_Score']
        val = row['APY']
        
        color = "off"
        if z > 2.0: color = "inverse" # Sell
        elif z < -2.0: color = "normal" # Buy
        
        with cols[i]:
            st.metric(label=c_name, value=f"{val:.2f}%", delta=f"{z:.2f} σ", delta_color=color)
else:
    st.warning("No active contracts outside the volatility zone.")

# --- CHART ---
st.subheader("2. Yield Term Structure")
fig = go.Figure()

if not bench_df.empty:
    fig.add_trace(go.Scatter(
        x=pd.concat([bench_df['Days_Left'], bench_df['Days_Left'][::-1]]),
        y=pd.concat([bench_df['Q3_APY'], bench_df['Q1_APY'][::-1]]),
        fill='toself', fillcolor='rgba(0, 255, 255, 0.05)', line=dict(width=0),
        hoverinfo="skip", name='Normal Regime (IQR)'
    ))
    fig.add_trace(go.Scatter(
        x=bench_df['Days_Left'], y=bench_df['Median_APY'],
        mode='lines', line=dict(color='cyan', width=1, dash='dash'), name='Fair Value (Median)'
    ))

colors = px.colors.qualitative.Bold
active_contracts = scored_df['Contract'].unique()

for i, c in enumerate(active_contracts):
    c_df = scored_df[scored_df['Contract'] == c]
    fig.add_trace(go.Scatter(
        x=c_df['Days_Left'], y=c_df['APY'],
        mode='lines', line=dict(color=colors[i % len(colors)], width=2),
        name=c
    ))

fig.update_layout(
    xaxis_title="Days Until Expiration", yaxis_title="Annualized Yield (%)",
    xaxis=dict(autorange="reversed"), height=700, hovermode="closest",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')} | Latency: ~1 Hour | Model: Robust Median Z-Score")

time.sleep(REFRESH_RATE)
st.rerun()
