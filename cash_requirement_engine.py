import pandas as pd
import numpy as np

# =====================================================
# CASH REQUIREMENT & FUNDING GAP ENGINE
# =====================================================

def run_cash_requirement_engine(
    account_forecasts,
    account_behavior_metrics,
    structural_cash,
    balances,
    stress_pct=0.15,
    confidence_factor=1.65
):
    """
    Computes:
    - Required cash
    - Funding gap
    - Idle cash
    - Action signal

    Inputs:
    - account_forecasts: Date, Account_ID, Predicted_Inflow, Predicted_Outflow
    - account_behavior_metrics: Account_ID, Outflow_Volatility
    - structural_cash: Account_ID, Structural_Inflow
    - balances: Account_ID, Balance_INR
    """

    # -----------------------------
    # MERGE ALL INPUTS
    # -----------------------------
    df = account_forecasts.merge(
        account_behavior_metrics[["Account_ID", "Outflow_Volatility"]],
        on="Account_ID",
        how="left"
    )

    df = df.merge(
        structural_cash[["Account_ID", "Structural_Inflow"]],
        on="Account_ID",
        how="left"
    )

    df = df.merge(
        balances[["Account_ID", "Balance_INR"]],
        on="Account_ID",
        how="left"
    )

    # Fill NaNs safely
    df["Outflow_Volatility"] = df["Outflow_Volatility"].fillna(0)
    df["Structural_Inflow"] = df["Structural_Inflow"].fillna(0)
    df["Balance_INR"] = df["Balance_INR"].fillna(0)

    # -----------------------------
    # BUFFERS
    # -----------------------------
    df["Safety_Buffer"] = df["Outflow_Volatility"] * confidence_factor
    df["Stress_Buffer"] = df["Predicted_Outflow"] * stress_pct
    df["Reliable_Inflow"] = df["Structural_Inflow"]

    # -----------------------------
    # REQUIRED CASH
    # -----------------------------
    df["Required_Cash"] = (
        df["Predicted_Outflow"]
        + df["Safety_Buffer"]
        + df["Stress_Buffer"]
        - df["Reliable_Inflow"]
    )

    # -----------------------------
    # FUNDING GAP & IDLE CASH
    # -----------------------------
    df["Funding_Gap"] = df["Required_Cash"] - df["Balance_INR"]

    df["Idle_Cash"] = np.where(
        df["Funding_Gap"] < 0,
        abs(df["Funding_Gap"]),
        0
    )

    # -----------------------------
    # ACTION SIGNAL
    # -----------------------------
    df["Action"] = np.select(
        [
            df["Funding_Gap"] > 0,
            df["Funding_Gap"] <= 0
        ],
        [
            "RAISE_FUNDS",
            "INVEST_SURPLUS"
        ],
        default="MONITOR"
    )

    # -----------------------------
    # BANK LEVEL AGGREGATION
    # -----------------------------
    bank_df = (
        df.groupby("Date")[[
            "Predicted_Inflow",
            "Predicted_Outflow",
            "Required_Cash",
            "Funding_Gap",
            "Idle_Cash"
        ]]
        .sum()
        .reset_index()
    )

    bank_df["Action"] = np.select(
        [
            bank_df["Funding_Gap"] > 0,
            bank_df["Funding_Gap"] <= 0
        ],
        [
            "BANK_FUNDING_REQUIRED",
            "BANK_HAS_EXCESS_LIQUIDITY"
        ],
        default="BANK_MONITOR"
    )

    return df, bank_df


# =====================================================
# TEST RUN
# =====================================================
if __name__ == "__main__":
    print("Cash Requirement Engine loaded successfully.")
