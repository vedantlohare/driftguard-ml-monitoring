import streamlit as st
import pandas as pd
import requests
import time

# Set page config
st.set_page_config(page_title="DriftGuard Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ DriftGuard: Real-Time ML Monitoring")

API_URL = "http://localhost:8000/api"

# Refresh button
if st.button("Refresh Data"):
    st.rerun()

# 1. Fetch Summary
try:
    summary_resp = requests.get(f"{API_URL}/dashboard/summary")
    summary = summary_resp.json()
except Exception as e:
    st.error(f"Could not connect to API Server at {API_URL}. Is it running?")
    st.stop()

# Display Model Health
st.header("Model Health")
col1, col2, col3, col4, col5 = st.columns(5)
health = summary.get("model_health", {})

with col1:
    st.metric("Data Quality", f"{health.get('data_quality_percent', 0)}%")
with col2:
    st.metric("Feature Drift Alerts", health.get('feature_drift_alerts', 0))
with col3:
    st.metric("Prediction Drift Alerts", health.get('prediction_drift_alerts', 0))
with col4:
    st.metric("Data Freshness", health.get('freshness', 'Unknown'))
with col5:
    st.metric("Schema Status", health.get('schema', 'Unknown'))

st.divider()

# 2. Display Feature Status
st.header("Feature-Level Drift Status")
feature_status = summary.get("feature_status", {})
if feature_status:
    # Convert to dataframe for nice display
    df_features = pd.DataFrame.from_dict(feature_status, orient='index').reset_index()
    df_features.columns = ['Feature', 'Drift Score (PSI/KS)', 'Status']
    
    # Color coding
    def color_status(val):
        color = 'green' if val == 'Healthy' else ('orange' if val == 'Warning' else 'red')
        return f'color: {color}'
        
    st.dataframe(df_features.style.applymap(color_status, subset=['Status']), use_container_width=True)
else:
    st.info("No drift alerts registered yet. Waiting for data window...")

st.divider()

# 3. Display Recent Alerts
st.header("Recent Data Quality Alerts")
try:
    dq_resp = requests.get(f"{API_URL}/alerts/data_quality?limit=10")
    dq_alerts = dq_resp.json()
    if dq_alerts:
        df_dq = pd.DataFrame(dq_alerts)[['timestamp', 'severity', 'issue_type', 'description']]
        st.dataframe(df_dq, use_container_width=True)
    else:
        st.info("No data quality alerts found. Stream looks clean!")
except Exception as e:
    st.warning("Failed to fetch Data Quality Alerts.")

# Auto refresh via st.empty loop (optional, keeping it simple for now)
