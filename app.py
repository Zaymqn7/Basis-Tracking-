import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime

# --- CONFIG ---
REFRESH_RATE = 15
MIN_DAYS = 2.0  # Gamma Filter
# Execution Reality (BTC Only for header)
SPOT_EXEC_BTC = "BTC_USDC" 
INDEX_BTC = "btc_usd"

st.set_page_config(page_title="HFT Relative Value Desk", layout="wide")

# --- 1. DATA LAYER: GENERIC FETCHER ---
def get_yield_curve(currency, index_name):
    """
    Fetches the Term Structure for a specific currency (BTC or ETH)
    using the Settlement Index as the source of truth.
    """
    url_ticker = "https://www.deribit.com/api/v2/public/ticker"
    url_index = "https://www.deribit.com/api/v2/public/get_index_price"
    url_inst = "https://www.deribit.com/api/v2/public/get_instruments"
    
    # A. Get Index Price
    try:
        i_resp = requests.get(url_index, params={"index_name": index_name}).json()['result']
        index_price = i_resp['index_price']
    except: return pd.DataFrame(), 0

    # B. Get Active Futures
    try:
        futures = requests.get(url_inst, params={"currency": currency, "kind": "future", "expired": "false"}).json()['result']
        futures = [f for f in futures if f['settlement_period'] != 'perpetual']
    except: return pd.DataFrame(), index_price

    # C. Calculate Yields
    rows = []
    for f in futures:
        try:
            t_resp = requests.get(url_ticker, params={"instrument_name": f['instrument_name']}).json()['result']
            mid_price = (t_resp['best_bid_price'] + t_resp['best_ask_price']) / 2
            
            expiry_ts = f['expiration_timestamp']
            days_left = (datetime.fromtimestamp(expiry_ts/1000) - datetime.now()).total_seconds() / (24 * 3600)
            
            if days_left < MIN_DAYS: continue

            # APY Calc using INDEX
            basis_pct = (mid_price - index_price) / index_price
            apy = basis_pct * (365 / days_left) * 100
            
            # Extract common date key (e.g., "26DEC25") for matching
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
    # Helper for BTC Perp only (for Chart 2)
    try:
        url = "https://www.deribit.com/api/v2/public/ticker"
        p_resp = requests.get(url, params={"instrument_name": "BTC-PERPETUAL"}).json()['result']
        return p_resp['current_funding'] * 3 * 365 * 100
    except: return 0

def get_btc_spot_exec():
    # Helper for Execution Monitor
    try:
        url = "https://www.deribit.com/api/v2/public/ticker"
        s_resp = requests.get(url, params={"instrument_name": SPOT_EXEC_BTC}).json()['result']
        return (s_resp['best_bid_price'] + s_resp['best_ask_price']) / 2
    except: return 0

# --- 2. MATH LAYER: CURVE FITTING ---
def fit_curves(df):
    if df.empty or len(df) < 3: return None, None
    x = df['Days_Left'].values
    y = df['APY'].values
    
    # Stiff (Log)
    try:
        coeffs_log = np.polyfit(np.log(x), y, 1)
        fit_log_y = coeffs_log[0] * np.log(x) + coeffs_log[1]
    except: fit_log_y = y

    # Flex (Quad)
    try:
        coeffs_poly = np.polyfit(x, y, 2)
        fit_poly_y = np.polyval(coeffs_poly, x)
    except: fit_poly_y = y
    
    return fit_log_y, fit_poly_y

# --- 3. VISUALIZATION ---
st.title("⚡ HFT Relative Value Desk")

# Fetch Data
btc_df, btc_index = get_yield_curve("BTC", "btc_usd")
eth_df, eth_index = get_yield_curve("ETH", "eth_usd")
btc_perp_apy = get_perp_data()
btc_exec = get_btc_spot_exec()

if btc_df.empty:
    st.error("Waiting for API Data...")
    time.sleep(3)
    st.rerun()

# --- HEADER: EXECUTION MONITOR ---
dislocation = btc_exec - btc_index
c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC Index", f"${btc_index:,.2f}")
c2.metric("BTC Spot Exec", f"${btc_exec:,.2f}", delta=f"{dislocation:.2f}", delta_color="inverse")
c3.metric("ETH Index", f"${eth_index:,.2f}")
c4.metric("Spread Dislocation", f"{dislocation:.2f}", delta="Warning" if abs(dislocation)>10 else "Normal", delta_color="off")
st.markdown("---")

# Prepare BTC Data for Chart 1
btc_df = btc_df.sort_values("Days_Left")
log_y, poly_y = fit_curves(btc_df)
if log_y is not None:
    btc_df['Fair_Log'] = log_y
    btc_df['Fair_Poly'] = poly_y
    btc_df['Residual'] = btc_df['APY'] - btc_df['Fair_Log']

# --- LAYOUT ---
col1, col2 = st.columns(2)

# CHART 1: BTC CURVE RESIDUALS
with col1:
    st.subheader("1. BTC Curve Geometry (Spatial Arb)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=btc_df['Days_Left'], y=btc_df['APY'],
        mode='markers+text', text=btc_df['Date_Key'], textposition="top center",
        marker=dict(size=14, color=btc_df['Residual'], colorscale="RdBu_r", cmin=-2, cmax=2),
        name='BTC Live'
    ))
    if log_y is not None:
        fig1.add_trace(go.Scatter(x=btc_df['Days_Left'], y=btc_df['Fair_Log'], mode='lines', line=dict(color='cyan', dash='dash'), name='Stiff Fit'))
        fig1.add_trace(go.Scatter(x=btc_df['Days_Left'], y=btc_df['Fair_Poly'], mode='lines', line=dict(color='gray', dash='dot'), name='Flex Fit'))
    
    fig1.update_layout(xaxis=dict(autorange="reversed"), height=450, xaxis_title="Days to Expiry", yaxis_title="BTC APY (%)")
    st.plotly_chart(fig1, use_container_width=True)

# CHART 2: BTC FUNDING ARB
with col2:
    st.subheader("2. BTC Cost of Carry (Cash & Carry)")
    carry_data = [{"Instrument": "PERP", "APY": btc_perp_apy, "Color": "crimson"}]
    for _, row in btc_df.iterrows():
        carry_data.append({"Instrument": row['Date_Key'], "APY": row['APY'], "Color": "teal"})
    df_carry = pd.DataFrame(carry_data)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df_carry['Instrument'], y=df_carry['APY'], marker_color=df_carry['Color'], text=df_carry['APY'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
    fig2.update_layout(height=450, yaxis_title="Annualized Cost (%)")
    st.plotly_chart(fig2, use_container_width=True)

# --- CHART 3: STAT ARB (ETH - BTC SPREAD) ---
st.subheader("3. Statistical Arb: ETH vs BTC Spread")

# Match Logic: Inner Join on Date_Key (e.g. "26DEC25")
spread_df = pd.merge(eth_df, btc_df, on="Date_Key", suffixes=("_ETH", "_BTC"))

if not spread_df.empty:
    # Calc Spread
    spread_df['Spread_BPS'] = (spread_df['APY_ETH'] - spread_df['APY_BTC']) * 100 # In Basis Points
    spread_df = spread_df.sort_values("Days_Left_BTC")
    
    fig3 = go.Figure()
    
    # Color Logic: Green if ETH > BTC (Risk On), Red if ETH < BTC (Risk Off)
    colors = ['#00FF00' if x > 0 else '#FF4444' for x in spread_df['Spread_BPS']]
    
    fig3.add_trace(go.Bar(
        x=spread_df['Date_Key'], 
        y=spread_df['Spread_BPS'],
        marker_color=colors,
        text=spread_df['Spread_BPS'].apply(lambda x: f"{x:.0f} bps"),
        textposition='auto'
    ))
    
    fig3.update_layout(
        title="Basis Spread Structure (ETH Yield minus BTC Yield)",
        yaxis_title="Spread (Basis Points)",
        xaxis_title="Contract Expiry",
        height=400
    )
    
    # Zero Line
    fig3.add_hline(y=0, line_color="white", opacity=0.3)
    
    st.plotly_chart(fig3, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    col_a.info("**Green Bars (Positive):** ETH Yield > BTC Yield. Market is chasing Beta/Alts. **Trade:** Short ETH Basis / Long BTC Basis.")
    col_b.warning("**Red Bars (Negative):** ETH Yield < BTC Yield. Market is fleeing to Safety (BTC). **Trade:** Long ETH Basis / Short BTC Basis.")

else:
    st.warning("No matching BTC/ETH contracts found to calculate spreads.")

time.sleep(REFRESH_RATE)
st.rerun()
