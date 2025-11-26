import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time
from datetime import datetime

# --- Configuration ---
SPOT_INSTRUMENT = "BTC_USDC"  # The actual Spot Asset
ST_CONFIG = st.set_page_config(page_title="BTC EFP Decay (Spot)", layout="wide")

# --- 1. Fetch Active Futures (Cached) ---
@st.cache_data(ttl=3600)
def get_instruments():
    url = "https://www.deribit.com/api/v2/public/get_instruments"
    params = {"currency": "BTC", "kind": "future", "expired": "false"}
    try:
        resp = requests.get(url, params=params).json()['result']
        # Filter for dated futures only (exclude perpetuals)
        return [i for i in resp if i['settlement_period'] != 'perpetual']
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return []

# --- 2. Fetch Historical Candles (Spot vs Future) ---
@st.cache_data(ttl=600)
def get_historical_basis(contract_name, expiry_timestamp):
    # End time = Now
    end_ts = int(time.time() * 1000)
    # Look back 365 days (max)
    start_ts = end_ts - (365 * 24 * 60 * 60 * 1000) 
    
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    
    # A. Get Future History (Daily Close)
    params_fut = {
        "instrument_name": contract_name, 
        "start_timestamp": start_ts, 
        "end_timestamp": end_ts, 
        "resolution": "1D"
    }
    try:
        data_fut = requests.get(url, params=params_fut).json()
        if 'result' not in data_fut or data_fut['result']['status'] == 'no_data':
            return pd.DataFrame()
        
        df_fut = pd.DataFrame(data_fut['result'])
        df_fut = df_fut[['ticks', 'close']].rename(columns={'close': 'Future_Price', 'ticks': 'Timestamp'})
    except:
        return pd.DataFrame()

    # B. Get Spot BTC_USDC History (Daily Close)
    params_spot = {
        "instrument_name": SPOT_INSTRUMENT, 
        "start_timestamp": start_ts, 
        "end_timestamp": end_ts, 
        "resolution": "1D"
    }
    try:
        data_spot = requests.get(url, params=params_spot).json()
        if 'result' not in data_spot or data_spot['result']['status'] == 'no_data':
            return pd.DataFrame() # If spot is down or empty
            
        df_spot = pd.DataFrame(data_spot['result'])
        df_spot = df_spot[['ticks', 'close']].rename(columns={'close': 'Spot_Price', 'ticks': 'Timestamp'})
    except:
        return pd.DataFrame()

    # C. Merge Future & Spot on Date
    # We use inner join: we only want days where BOTH traded
    df = pd.merge(df_fut, df_spot, on='Timestamp', how='inner')
    
    # Convert Timestamp to human date
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    
    # Calculate Days to Expiry
    expiry_date = datetime.fromtimestamp(expiry_timestamp / 1000)
    df['Days_to_Expiry'] = (expiry_date - df['Date']).dt.days
    
    # Filter: Keep only positive days (ignore post-expiry data)
    df = df[df['Days_to_Expiry'] >= 0]
    
    # Calculate Basis ($)
    df['Basis_USD'] = df['Future_Price'] - df['Spot_Price']
    df['Contract'] = contract_name
    
    return df

# --- 3. Main Dashboard ---
st.title("📉 BTC EFP Decay Tracker (Spot vs Future)")
st.markdown(f"""
**Strategy:** Tracking the **Exchange for Physical (EFP)** spread decay.
* **Spot Leg:** {SPOT_INSTRUMENT}
* **Future Leg:** Dated Futures (e.g., BTC-27DEC24)
* **Logic:** As time passes (moving right on the chart), the Basis should decay to $0.
""")

instruments = get_instruments()

if not instruments:
    st.warning("No active futures found.")
    st.stop()

# Progress Bar
progress_text = "Analyzing EFP history..."
my_bar = st.progress(0, text=progress_text)
all_history = []

for index, instr in enumerate(instruments):
    # Update Progress
    my_bar.progress((index + 1) / len(instruments), text=f"Fetching {instr['instrument_name']}...")
    
    # Fetch
    df_contract = get_historical_basis(instr['instrument_name'], instr['expiration_timestamp'])
    if not df_contract.empty:
        all_history.append(df_contract)

my_bar.empty()

if all_history:
    full_df = pd.concat(all_history)
    
    # --- The "EFP Decay" Chart ---
    st.subheader("Historical Basis Decay (Convergence)")
    
    # We construct the chart
    fig = px.line(full_df, 
                  x="Days_to_Expiry", 
                  y="Basis_USD", 
                  color="Contract",
                  title="EFP Term Structure Decay",
                  hover_data=["Date", "Future_Price", "Spot_Price"])
    
    # CRITICAL: Reverse X Axis so 0 (Expiry) is on the RIGHT
    fig.update_layout(xaxis=dict(autorange="reversed"), hovermode="x unified")
    
    # Add Zero Line (Target)
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, annotation_text="Parity ($0)")
    
    fig.update_xaxes(title_text="Days Until Expiration")
    fig.update_yaxes(title_text="Basis Spread ($)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Context Analysis ---
    st.info("""
    **How to spot 'Cheap' or 'Expensive' Basis:**
    Look vertically at a specific time (e.g., '60 Days Left'). 
    If the current contract's line is significantly **higher** than previous contracts were at 60 days, 
    the EFP is currently expensive (Good to Short Future / Long Spot).
    """)

else:
    st.error("Could not retrieve historical data. Note: BTC_USDC spot history on Deribit starts ~April 2023.")

# Refresh
if st.button('Refresh Data'):
    st.rerun()
