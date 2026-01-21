import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from baseline_forecasting import run_baseline_forecasting

# =====================================================
# APPLICATION CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Cashflow Forecasting Model",
    layout="wide"
)

st.title("Cashflow Forecasting Model")

# =====================================================
# DATA INGESTION
# =====================================================
uploaded_file = st.file_uploader(
    "Upload Cashflow Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a cashflow CSV file to proceed.")
    st.stop()

# =====================================================
# LOAD & VALIDATE DATA
# =====================================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data(uploaded_file)

required_columns = {
    "Date",
    "Account_ID",
    "Inflow_INR",
    "Outflow_INR",
    "Balance_INR"
}

if not required_columns.issubset(df.columns):
    st.error(
        "The uploaded dataset does not match the required schema.\n\n"
        "Expected columns:\n"
        "- Date\n- Account_ID\n- Inflow_INR\n- Outflow_INR\n- Balance_INR"
    )
    st.stop()

# =====================================================
# SIDEBAR — USER CONTROLS
# =====================================================
st.sidebar.header("Analysis Controls")

view_mode = st.sidebar.radio(
    "Analysis Level",
    ["Bank Level", "Account Level"]
)

date_range = st.sidebar.date_input(
    "Analysis Period",
    [df["Date"].min().date(), df["Date"].max().date()]
)

stress_pct = st.sidebar.slider(
    "Outflow Stress Scenario (%)",
    min_value=0,
    max_value=30,
    value=10
)

unit_choice = st.sidebar.selectbox(
    "Monetary Units",
    ["INR", "Lakhs", "Millions"]
)

UNIT_CONFIG = {
    "INR": (1, "₹"),
    "Lakhs": (100_000, "₹ Lakhs"),
    "Millions": (1_000_000, "₹ Millions")
}

unit_divisor, unit_label = UNIT_CONFIG[unit_choice]

# =====================================================
# DATE FILTERING
# =====================================================
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

df_filtered = df[
    (df["Date"] >= start_date) &
    (df["Date"] <= end_date)
]

if df_filtered.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

# =====================================================
# FORECAST GENERATION (BASELINE MODEL)
# =====================================================
account_fc_all, bank_fc_all = run_baseline_forecasting(df, horizon=60)

bank_fc = bank_fc_all[
    (bank_fc_all["Date"] >= start_date) &
    (bank_fc_all["Date"] <= end_date)
]

account_fc = account_fc_all[
    (account_fc_all["Date"] >= start_date) &
    (account_fc_all["Date"] <= end_date)
]

# =====================================================
# CONFIDENCE BAND CALCULATION
# =====================================================
def add_confidence_band(forecast_df, historical_df, z=1.65):
    std_outflow = historical_df["Outflow_INR"].std()
    forecast_df["Upper_Bound"] = forecast_df["Predicted_Outflow"] + z * std_outflow
    forecast_df["Lower_Bound"] = forecast_df["Predicted_Outflow"] - z * std_outflow
    return forecast_df

# =====================================================
# EXECUTIVE SUMMARY LOGIC
# =====================================================
def executive_summary(net_position, stress_level):
    if net_position > 0 and stress_level < 10:
        return (
            "Liquidity conditions are healthy. Forecasted cashflows indicate surplus "
            "availability, supporting investment or redeployment decisions."
        )
    elif stress_level >= 20:
        return (
            "Stress scenarios indicate elevated funding risk. Additional liquidity buffers "
            "or proactive funding actions are recommended."
        )
    else:
        return (
            "Liquidity conditions are stable but warrant close monitoring, particularly "
            "under adverse outflow scenarios."
        )

# =====================================================
# BANK-LEVEL ANALYSIS
# =====================================================
if view_mode == "Bank Level":

    st.subheader("Bank-Level Liquidity Overview (Selected Period)")

    bank_hist = (
        df_filtered
        .groupby("Date")[["Inflow_INR", "Outflow_INR"]]
        .sum()
        .reset_index()
    )

    bank_hist["Net_Cash"] = bank_hist["Inflow_INR"] - bank_hist["Outflow_INR"]

    for col in ["Inflow_INR", "Outflow_INR", "Net_Cash"]:
        bank_hist[col] /= unit_divisor

    total_inflow = bank_hist["Inflow_INR"].sum()
    total_outflow = bank_hist["Outflow_INR"].sum()
    net_position = total_inflow - total_outflow

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total Inflow ({unit_label})", f"{total_inflow:,.2f}")
    c2.metric(f"Total Outflow ({unit_label})", f"{total_outflow:,.2f}")
    c3.metric(f"Net Position ({unit_label})", f"{net_position:,.2f}")

    st.markdown("#### Net Cash Position Over Time")

    fig_net = px.line(
        bank_hist,
        x="Date",
        y="Net_Cash",
        markers=True
    )

    fig_net.add_hline(y=0, line_dash="dash")
    fig_net.update_yaxes(
        title=f"Net Cash ({unit_label})",
        tickformat=",.2f",
        separatethousands=True
    )

    fig_net.update_traces(
        hovertemplate="Date=%{x}<br>Net Cash="
        + unit_label + " %{y:,.2f}<extra></extra>"
    )

    st.plotly_chart(fig_net, use_container_width=True)

    if not bank_fc.empty:
        bank_fc = add_confidence_band(bank_fc, df_filtered)

        for col in ["Predicted_Outflow", "Upper_Bound", "Lower_Bound"]:
            bank_fc[col] /= unit_divisor

        bank_fc["Stress_Outflow"] = (
            bank_fc["Predicted_Outflow"] * (1 + stress_pct / 100)
        )

        st.markdown("#### Forecasted Outflows with Stress Scenario")

        fig_fc = px.line(
            bank_fc,
            x="Date",
            y=["Predicted_Outflow", "Stress_Outflow"]
        )

        fig_fc.add_scatter(
            x=bank_fc["Date"],
            y=bank_fc["Upper_Bound"],
            mode="lines",
            name="Upper Confidence Bound",
            line=dict(dash="dash")
        )

        fig_fc.add_scatter(
            x=bank_fc["Date"],
            y=bank_fc["Lower_Bound"],
            mode="lines",
            name="Lower Confidence Bound",
            line=dict(dash="dash")
        )

        fig_fc.update_yaxes(
            title=f"Outflow Amount ({unit_label})",
            tickformat=",.2f",
            separatethousands=True
        )

        fig_fc.update_traces(
            hovertemplate="Date=%{x}<br>Amount="
            + unit_label + " %{y:,.2f}<extra></extra>"
        )

        st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("#### Executive Summary")
    st.info(executive_summary(net_position, stress_pct))

# =====================================================
# ACCOUNT-LEVEL ANALYSIS
# =====================================================
else:

    st.subheader("Account-Level Liquidity Overview")

    account_id = st.sidebar.selectbox(
        "Select Account",
        sorted(df_filtered["Account_ID"].unique())
    )

    acc = df_filtered[df_filtered["Account_ID"] == account_id].copy()
    acc["Net_Cash"] = acc["Inflow_INR"] - acc["Outflow_INR"]

    for col in ["Inflow_INR", "Outflow_INR", "Balance_INR", "Net_Cash"]:
        acc[col] /= unit_divisor

    avg_inflow = acc["Inflow_INR"].mean()
    avg_outflow = acc["Outflow_INR"].mean()
    current_balance = acc["Balance_INR"].iloc[-1]

    risk_level = "Low"
    if current_balance < 0:
        risk_level = "High"
    elif current_balance < avg_outflow * 3:
        risk_level = "Medium"

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Average Inflow ({unit_label})", f"{avg_inflow:,.2f}")
    c2.metric(f"Average Outflow ({unit_label})", f"{avg_outflow:,.2f}")
    c3.metric(f"Current Balance ({unit_label})", f"{current_balance:,.2f}")

    st.markdown(f"**Funding Risk Classification:** {risk_level}")

    fig_acc = px.line(
        acc,
        x="Date",
        y="Net_Cash",
        markers=True
    )

    fig_acc.add_hline(y=0, line_dash="dash")
    fig_acc.update_yaxes(
        title=f"Net Cash ({unit_label})",
        tickformat=",.2f",
        separatethousands=True
    )

    fig_acc.update_traces(
        hovertemplate="Date=%{x}<br>Net Cash="
        + unit_label + " %{y:,.2f}<extra></extra>"
    )

    st.plotly_chart(fig_acc, use_container_width=True)

# =====================================================
# APPLICATION FOOTER
# =====================================================
st.markdown("---")
st.success(
    "System status: Operational. Forecasts, stress scenarios, and liquidity insights "
    "are based on historical behavior and baseline statistical modeling."
)
