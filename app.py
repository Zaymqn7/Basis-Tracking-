import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# --- CONFIG ---
SPOT_INSTRUMENT = "BTC_USDC"
st.set_page_config(page_title="BTC EFP Valuator", layout="wide")

# --- 1. Load Static Benchmark (Fast) ---
@st.cache_data
def load_benchmark():
    try:
        # Reads the file you just uploaded
        df = pd.read_csv("benchmark.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 2. Fetch Active Contracts (Live) ---
@st.cache_data(ttl=300)
def get_live_data():
    # A. Get Instruments
    url = "https://www.deribit.com/api/v2/public/get_instruments"
    try:
        resp = requests.get(url, params={"currency": "BTC", "kind": "future", "expired": "false"}).json()['result']
        futures = [i for i in resp if i['settlement_period'] != 'perpetual']
    except: return pd.DataFrame()

    # B. Get History for each active contract
    live_rows = []
    tv_url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    now_ts = int(time.time() * 1000)
    start_ts = now_ts - (180 * 24 * 60 * 60 * 1000) # Look back 6 months

    for f in futures:
        try:
            # 1. Get Future History
            d_fut = requests.get(tv_url, params={"instrument_name": f['instrument_name'], "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "1D"}).json()
            if 'result' not in d_fut or d_fut['result']['status'] == 'no_data': continue
            df_f = pd.DataFrame(d_fut['result'])[['ticks', 'close']].rename(columns={'close': 'Future', 'ticks': 'Timestamp'})

            # 2. Get Spot History
            d_spot = requests.get(tv_url, params={"instrument_name": SPOT_INSTRUMENT, "start_timestamp": start_ts, "end_timestamp": now_ts, "resolution": "1D"}).json()
            if 'result' not in d_spot or d_spot['result']['status'] == 'no_data': continue
            df_s = pd.DataFrame(d_spot['result'])[['ticks', 'close']].rename(columns={'close': 'Spot', 'ticks': 'Timestamp'})

            # 3. Merge & Calc
            df = pd.merge(df_f, df_s, on='Timestamp', how='inner')
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            
            expiry_ts = f['expiration_timestamp']
            expiry_date = datetime.fromtimestamp(expiry_ts/1000)
            
            df['Days_Left'] = (expiry_date - df['Date']).dt.days
            df['Basis'] = df['Future'] - df['Spot']
            df['Contract'] = f['instrument_name']
            
            # Filter valid days
            df = df[df['Days_Left'] >= 0]
            live_rows.append(df)
        except: continue
        
    if live_rows:
        return pd.concat(live_rows)
    return pd.DataFrame()

# --- 3. Dashboard Interface ---
st.title("BTC Basis: Live vs. Fair Value Model")
st.markdown("Comparing **Live EFP Decay** against the **Historical Median (2023-2024)**.")

# Load Data
bench_df = load_benchmark()
live_df = get_live_data()

if live_df.empty:
    st.error("Could not fetch live Deribit data. API might be busy.")
    st.stop()

# --- The Chart ---
fig = go.Figure()

# A. Draw the "Ghost" Benchmark (The Tunnel)
if not bench_df.empty:
    # 1. The Grey Zone (25th-75th Percentile)
    fig.add_trace(go.Scatter(
        x=pd.concat([bench_df['Days_Left'], bench_df['Days_Left'][::-1]]), 
        y=pd.concat([bench_df['Q3'], bench_df['Q1'][::-1]]),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.15)', # Light Grey
        line=dict(width=0),
        hoverinfo="skip",
        name='Normal Range'
    ))
    # 2. The White Median Line
    fig.add_trace(go.Scatter(
        x=bench_df['Days_Left'],
        y=bench_df['Median'],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Fair Value (Median)'
    ))
else:
    st.warning("⚠️ 'benchmark.csv' not detected in GitHub repo. Using Live Data only.")

# B. Draw Live Contracts
colors = px.colors.qualitative.Bold
contracts = live_df['Contract'].unique()

for i, c in enumerate(contracts):
    d = live_df[live_df['Contract'] == c]
    fig.add_trace(go.Scatter(
        x=d['Days_Left'],
        y=d['Basis'],
        mode='lines',
        line=dict(color=colors[i % len(colors)], width=2.5),
        name=c
    ))

# C. Formatting
fig.update_layout(
    title="EFP Term Structure Decay",
    xaxis_title="Days Until Expiration",
    yaxis_title="Basis Spread ($)",
    xaxis=dict(autorange="reversed"), # 0 is on the Right
    hovermode="x unified",
    height=650,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

# Explanation
st.info("""
**How to Read:**
* **White Dashed Line:** The "Fair Value" (Median of previous expired contracts).
* **Grey Tunnel:** The "Boring Zone".
* **Strategy:** * If a live line is way **ABOVE** the tunnel → **Expensive** (Consider Short Basis).
    * If a live line is way **BELOW** the tunnel → **Cheap** (Consider Long Basis).
""")
