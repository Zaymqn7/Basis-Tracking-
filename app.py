import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime

# --- CONFIG ---
REFRESH_RATE = 10
SPOT_INSTRUMENT_TRADABLE = "BTC_USDC" # Execution Reality
INDEX_NAME = "btc_usd"                # Settlement Truth
PERP_INSTRUMENT = "BTC-PERPETUAL"
MIN_DAYS = 2.0  # Lowered to 2 days so you can see the 28NOV contract

st.set_page_config(page_title="HFT Relative Value Desk", layout="wide")

# --- 1. DATA LAYER: LIVE SNAPSHOTS ---
def get_market_snapshot():
    url_ticker = "https://www.deribit.com/api/v2/public/ticker"
    url_index = "https://www.deribit.com/api/v2/public/get_index_price"
    url_inst = "https://www.deribit.com/api/v2/public/get_instruments"
    
    # A. Get Index Price (Source of Truth)
    try:
        i_resp = requests.get(url_index, params={"index_name": INDEX_NAME}).json()['result']
        index_price = i_resp['index_price']
    except: return pd.DataFrame(), {}, 0, 0

    # B. Get Tradable Spot Price (Execution Reality)
    try:
        s_resp = requests.get(url_ticker, params={"instrument_name": SPOT_INSTRUMENT_TRADABLE}).json()['result']
        spot_price_tradable = (s_resp['best_bid_price'] + s_resp['best_ask_price']) / 2
    except: spot_price_tradable = 0

    # C. Get Active Instruments
    try:
        futures = requests.get(url_inst, params={"currency": "BTC", "kind": "future", "expired": "false"}).json()['result']
        futures = [f for f in futures if f['settlement_period'] != 'perpetual']
    except: return pd.DataFrame(), {}, 0, 0

    # D. Get Perp Data
    perp_data = {}
    try:
        p_resp = requests.get(url_ticker, params={"instrument_name": PERP_INSTRUMENT}).json()['result']
        # Rabbit (Instant)
        perp_data['live_funding_apy'] = p_resp['current_funding'] * 3 * 365 * 100
    except: 
        perp_data = {'live_funding_apy': 0}

    # E. Get Futures Yields (Calculated vs INDEX)
    rows = []
    for f in futures:
        try:
            t_resp = requests.get(url_ticker, params={"instrument_name": f['instrument_name']}).json()['result']
            mid_price = (t_resp['best_bid_price'] + t_resp['best_ask_price']) / 2
            
            expiry_ts = f['expiration_timestamp']
            days_left = (datetime.fromtimestamp(expiry_ts/1000) - datetime.now()).total_seconds() / (24 * 3600)
            
            if days_left < MIN_DAYS: continue

            # APY Calc using INDEX PRICE (Clean Curve)
            basis_pct = (mid_price - index_price) / index_price
            apy = basis_pct * (365 / days_left) * 100
            
            rows.append({
                "Contract": f['instrument_name'],
                "Days_Left": days_left,
                "APY": apy,
                "Price": mid_price
            })
        except: continue
        
    return pd.DataFrame(rows), perp_data, index_price, spot_price_tradable

# --- 2. MATH LAYER: CURVE FITTING ---
def fit_curves(df):
    if df.empty or len(df) < 3: return None, None
    x = df['Days_Left'].values
    y = df['APY'].values
    
    # Stiff Fit (Log)
    try:
        coeffs_log = np.polyfit(np.log(x), y, 1)
        fit_log_y = coeffs_log[0] * np.log(x) + coeffs_log[1]
    except: fit_log_y = y

    # Flex Fit (Quadratic)
    try:
        coeffs_poly = np.polyfit(x, y, 2)
        fit_poly_y = np.polyval(coeffs_poly, x)
    except: fit_poly_y = y
    
    return fit_log_y, fit_poly_y

# --- 3. VISUALIZATION ---
st.title("⚡ HFT Relative Value Desk")

live_df, perp_data, index_price, spot_exec = get_market_snapshot()

if live_df.empty:
    st.error("Waiting for data...")
    time.sleep(3)
    st.rerun()

# --- HEADER METRICS: EXECUTION MONITOR ---
# This tells you if your "Clean Model" is disconnected from "Dirty Reality"
dislocation = spot_exec - index_price
c1, c2, c3 = st.columns(3)
c1.metric("Settlement Index (BTC-USD)", f"${index_price:,.2f}")
c2.metric("Execution Spot (BTC_USDC)", f"${spot_exec:,.2f}", delta=f"{dislocation:.2f}", delta_color="inverse")
c3.metric("Spot Dislocation", f"{dislocation:.2f} USD", 
          delta="Normal" if abs(dislocation) < 10 else "High Slippage",
          delta_color="off" if abs(dislocation) < 10 else "inverse")

st.markdown("---")

# Run Fits
live_df = live_df.sort_values("Days_Left")
log_y, poly_y = fit_curves(live_df)

if log_y is not None:
    live_df['Fair_Log'] = log_y
    live_df['Fair_Poly'] = poly_y
    live_df['Residual'] = live_df['APY'] - live_df['Fair_Log']

# --- LAYOUT ---
col1, col2 = st.columns(2)

# --- CHART 1: CURVE RESIDUALS ---
with col1:
    st.subheader("1. Curve Residuals (Spatial Arb)")
    fig_curve = go.Figure()

    fig_curve.add_trace(go.Scatter(
        x=live_df['Days_Left'], y=live_df['APY'],
        mode='markers+text', text=live_df['Contract'].str.replace("BTC-",""),
        textposition="top center",
        marker=dict(size=14, color=live_df['Residual'], colorscale="RdBu_r", cmin=-2, cmax=2),
        name='Live Contracts'
    ))

    if log_y is not None:
        fig_curve.add_trace(go.Scatter(
            x=live_df['Days_Left'], y=live_df['Fair_Log'],
            mode='lines', line=dict(color='cyan', width=2, dash='dash'),
            name='Stiff Fit'
        ))
        fig_curve.add_trace(go.Scatter(
            x=live_df['Days_Left'], y=live_df['Fair_Poly'],
            mode='lines', line=dict(color='gray', width=1, dash='dot'),
            name='Flex Fit'
        ))

    fig_curve.update_layout(
        xaxis_title="Days Until Expiration", yaxis_title="Annualized Yield (Index Basis)",
        xaxis=dict(autorange="reversed"), height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
    )
    st.plotly_chart(fig_curve, use_container_width=True)

# --- CHART 2: FUNDING ARB ---
with col2:
    st.subheader("2. Cost of Carry (Funding Arb)")
    
    carry_data = []
    carry_data.append({
        "Instrument": "PERP (Live)", "APY": perp_data.get('live_funding_apy', 0), "Color": "crimson"
    })
    for idx, row in live_df.iterrows():
        c_name = row['Contract'].replace("BTC-", "")
        carry_data.append({
            "Instrument": c_name, "APY": row['APY'], "Color": "teal"
        })
        
    df_carry = pd.DataFrame(carry_data)
    
    fig_carry = go.Figure()
    fig_carry.add_trace(go.Bar(
        x=df_carry['Instrument'], y=df_carry['APY'],
        marker_color=df_carry['Color'], text=df_carry['APY'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto'
    ))
    
    fig_carry.update_layout(yaxis_title="Annualized Cost (%)", height=500)
    st.plotly_chart(fig_carry, use_container_width=True)

# --- RAW DATA ---
with st.expander("Quant Monitor (Live Details)"):
    format_mapping = {
        "APY": "{:.2f}%", "Price": "${:,.2f}", 
        "Days_Left": "{:.1f}", "Fair_Log": "{:.2f}%", 
        "Fair_Poly": "{:.2f}%", "Residual": "{:.2f}"
    }
    valid_format = {k: v for k, v in format_mapping.items() if k in live_df.columns}
    st.dataframe(live_df.style.format(valid_format))

time.sleep(REFRESH_RATE)
st.rerun()
