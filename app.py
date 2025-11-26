import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time
from datetime import datetime

# --- Configuration ---
REFRESH_RATE = 6  # Seconds
st.set_page_config(page_title="Live Basis Monitor", layout="wide")

# --- 1. Get Instruments (Cached) ---
@st.cache_data(ttl=3600)
def get_instruments():
    # We fetch ALL BTC futures
    url = "https://www.deribit.com/api/v2/public/get_instruments"
    params = {"currency": "BTC", "kind": "future", "expired": "false"}
    try:
        resp = requests.get(url, params=params).json()['result']
        # Filter out Perpetuals, we only want Dated Futures
        return [i for i in resp if i['settlement_period'] != 'perpetual']
    except Exception as e:
        st.error(f"Failed to fetch instruments: {e}")
        return []

# --- 2. Get Live Prices (Spot & Futures) ---
def get_live_data(futures_list):
    # A. Get SPOT Price (BTC_USDC) - "Tradable"
    spot_url = "https://www.deribit.com/api/v2/public/ticker"
    try:
        spot_resp = requests.get(spot_url, params={"instrument_name": "BTC_USDC"}).json()['result']
        spot_mid = (spot_resp['best_bid_price'] + spot_resp['best_ask_price']) / 2
    except Exception:
        st.error("Could not fetch Spot BTC_USDC. Market might be down.")
        return 0, pd.DataFrame()

    # B. Get Futures Prices
    rows = []
    for f in futures_list:
        try:
            # Fetch ticker for this specific future
            ticker = requests.get(spot_url, params={"instrument_name": f['instrument_name']}).json()['result']
            
            # Calculate Mid Price
            fut_mid = (ticker['best_bid_price'] + ticker['best_ask_price']) / 2
            
            # Calculate Expiry
            expiry_ts = f['expiration_timestamp'] / 1000
            expiry_date = datetime.fromtimestamp(expiry_ts)
            days_left = (expiry_date - datetime.now()).days
            if days_left <= 0: days_left = 0.5 # Avoid division by zero

            # --- THE MATH ---
            basis_usd = fut_mid - spot_mid
            # Annualized % = (Basis / Spot) * (365 / Days)
            basis_apr = (basis_usd / spot_mid) * (365 / days_left) * 100

            rows.append({
                "Contract": f['instrument_name'],
                "Date": expiry_date.strftime('%Y-%m-%d'),
                "Days": days_left,
                "Fut Price": fut_mid,
                "Basis ($)": basis_usd,
                "Annualized (%)": basis_apr
            })
        except Exception:
            continue
            
    return spot_mid, pd.DataFrame(rows)

# --- 3. The Dashboard Layout ---
st.title("⚡ BTC Basis Term Structure (Mid-Price)")

# Fetch Data
futures = get_instruments()
if futures:
    spot_price, df = get_live_data(futures)

    # Top Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Spot BTC/USDC (Mid)", f"${spot_price:,.2f}")
    c2.metric("Contracts Tracked", len(df))
    if not df.empty:
        # Sort by date so the line chart connects correctly
        df = df.sort_values("Days")
        
        # Best yield highlight
        best_yield = df.loc[df['Annualized (%)'].idxmax()]
        c3.metric("Best Yield", f"{best_yield['Annualized (%)']:.2f}%", best_yield['Contract'])

        # --- Visualizations ---
        st.markdown("---")
        
        # Chart 1: The Yield Curve (Annualized)
        st.subheader("1. Annualized Basis Yield Curve (%)")
        st.caption("This curve shows the 'Cost of Carry'. If it slopes up, the market expects higher prices later.")
        
        fig_curve = px.line(df, x="Date", y="Annualized (%)", markers=True, 
                            text="Annualized (%)", hover_data=["Contract", "Basis ($)"])
        fig_curve.update_traces(texttemplate='%{y:.2f}%', textposition="top center", line_shape="spline")
        fig_curve.update_layout(height=500, xaxis_title="Expiration Date", yaxis_title="Annualized Return (%)")
        st.plotly_chart(fig_curve, use_container_width=True)

        # Chart 2: Dollar Basis (Bar)
        st.subheader("2. Raw Price Difference ($)")
        st.caption("How many dollars more expensive is the Future compared to Spot?")
        
        fig_bar = px.bar(df, x="Contract", y="Basis ($)", color="Basis ($)", 
                         color_continuous_scale="Tealgrn")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Raw Data
        with st.expander("Show Raw Data Table"):
            st.dataframe(df.style.format({"Fut Price": "{:.2f}", "Basis ($)": "{:.2f}", "Annualized (%)": "{:.2f}"}))

# Auto-Refresh
time.sleep(REFRESH_RATE)
st.rerun()
