import pandas as pd
import numpy as np

# =====================================================
# BASELINE INFLOW / OUTFLOW FORECASTING
# =====================================================

def run_baseline_forecasting(df, horizon=14, rolling_window=14, alpha=0.7):
    """
    Baseline forecast using:
    - Rolling mean
    - Day-of-week seasonality

    Returns:
    - account_level_forecast
    - bank_level_forecast
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Account_ID", "Date"])
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    forecasts = []

    # -------------------------------------------------
    # ACCOUNT LEVEL BASELINE
    # -------------------------------------------------
    for acc, g in df.groupby("Account_ID"):
        g = g.sort_values("Date")

        last_date = g["Date"].max()

        # Rolling means
        roll_in = g["Inflow_INR"].tail(rolling_window).mean()
        roll_out = g["Outflow_INR"].tail(rolling_window).mean()

        # Day-of-week averages
        dow_profile = (
            g.groupby("DayOfWeek")[["Inflow_INR", "Outflow_INR"]]
            .mean()
            .to_dict()
        )

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )

        for d in future_dates:
            dow = d.dayofweek

            dow_in = dow_profile["Inflow_INR"].get(dow, roll_in)
            dow_out = dow_profile["Outflow_INR"].get(dow, roll_out)

            pred_in = alpha * roll_in + (1 - alpha) * dow_in
            pred_out = alpha * roll_out + (1 - alpha) * dow_out

            forecasts.append({
                "Date": d,
                "Account_ID": acc,
                "Predicted_Inflow": max(pred_in, 0),
                "Predicted_Outflow": max(pred_out, 0),
                "Model": "BASELINE"
            })

    account_forecast = pd.DataFrame(forecasts)

    # -------------------------------------------------
    # BANK LEVEL BASELINE (AGGREGATED)
    # -------------------------------------------------
    bank_forecast = (
        account_forecast
        .groupby("Date")[["Predicted_Inflow", "Predicted_Outflow"]]
        .sum()
        .reset_index()
    )

    bank_forecast["Model"] = "BASELINE_BANK"

    return account_forecast, bank_forecast


# =====================================================
# TEST RUN
# =====================================================
if __name__ == "__main__":
    df = pd.read_csv("sample_cashflow.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    acc_fc, bank_fc = run_baseline_forecasting(df)

    print("ACCOUNT LEVEL BASELINE:")
    print(acc_fc.head())

    print("\nBANK LEVEL BASELINE:")
    print(bank_fc.head())
