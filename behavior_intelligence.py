import pandas as pd
import numpy as np

# =====================================================
# 1. PREPROCESS
# =====================================================
def preprocess(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Account_ID", "Date"])

    # Net cash movement
    df["Net_Cash"] = df["Inflow_INR"] - df["Outflow_INR"]

    # Calendar features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfMonth"] = df["Date"].dt.day
    df["Is_Month_End"] = (df["DayOfMonth"] >= 25).astype(int)

    return df


# =====================================================
# 2. ACCOUNT-LEVEL BEHAVIOR METRICS
# =====================================================
def account_behavior_metrics(df):
    records = []

    for acc, g in df.groupby("Account_ID"):
        avg_in = g["Inflow_INR"].mean()
        avg_out = g["Outflow_INR"].mean()

        std_in = g["Inflow_INR"].std()
        std_out = g["Outflow_INR"].std()

        cv_in = std_in / avg_in if avg_in > 0 else np.nan
        cv_out = std_out / avg_out if avg_out > 0 else np.nan

        stability_score = 1 / (1 + cv_out) if not np.isnan(cv_out) else 0

        records.append({
            "Account_ID": acc,
            "Avg_Inflow": avg_in,
            "Avg_Outflow": avg_out,
            "Net_Flow": avg_in - avg_out,
            "Outflow_Volatility": std_out,
            "Outflow_CV": cv_out,
            "Stability_Score": stability_score
        })

    return pd.DataFrame(records)


# =====================================================
# 3. STRUCTURAL vs VOLATILE CASH
# =====================================================
def structural_cash_estimation(df, quantile=0.25):
    rows = []

    for acc, g in df.groupby("Account_ID"):
        structural_inflow = g["Inflow_INR"].quantile(quantile)
        mean_inflow = g["Inflow_INR"].mean()

        volatile_part = max(mean_inflow - structural_inflow, 0)

        rows.append({
            "Account_ID": acc,
            "Structural_Inflow": structural_inflow,
            "Volatile_Inflow": volatile_part,
            "Structural_Ratio": structural_inflow / mean_inflow if mean_inflow > 0 else 0
        })

    return pd.DataFrame(rows)


# =====================================================
# 4. SEASONALITY & DATE EFFECTS
# =====================================================
def seasonality_analysis(df):
    # Day-of-week pattern
    dow = (
        df.groupby("DayOfWeek")[["Inflow_INR", "Outflow_INR"]]
        .mean()
        .reset_index()
    )

    # Month-end behavior
    month_end = (
        df.groupby("Is_Month_End")[["Inflow_INR", "Outflow_INR"]]
        .mean()
        .reset_index()
    )

    return dow, month_end


# =====================================================
# 5. BANK-LEVEL ROLLUP
# =====================================================
def bank_level_summary(df):
    bank_daily = (
        df.groupby("Date")[["Inflow_INR", "Outflow_INR", "Net_Cash"]]
        .sum()
        .reset_index()
    )

    summary = {
        "Avg_Daily_Inflow": bank_daily["Inflow_INR"].mean(),
        "Avg_Daily_Outflow": bank_daily["Outflow_INR"].mean(),
        "Net_Position": bank_daily["Net_Cash"].mean(),
        "Outflow_Volatility": bank_daily["Outflow_INR"].std()
    }

    return bank_daily, summary


# =====================================================
# 6. MASTER PIPELINE
# =====================================================
def run_behavior_intelligence(df):
    df = preprocess(df)

    account_metrics = account_behavior_metrics(df)
    structural_cash = structural_cash_estimation(df)
    dow_pattern, month_end_pattern = seasonality_analysis(df)
    bank_daily, bank_summary = bank_level_summary(df)

    return {
        "account_metrics": account_metrics,
        "structural_cash": structural_cash,
        "day_of_week_pattern": dow_pattern,
        "month_end_pattern": month_end_pattern,
        "bank_daily": bank_daily,
        "bank_summary": bank_summary
    }


# =====================================================
# 7. TEST RUN
# =====================================================
if __name__ == "__main__":
    df = pd.read_csv("sample_cashflow.csv")
    results = run_behavior_intelligence(df)

    print("\nACCOUNT BEHAVIOR:")
    print(results["account_metrics"].head())

    print("\nSTRUCTURAL CASH:")
    print(results["structural_cash"].head())

    print("\nBANK SUMMARY:")
    print(results["bank_summary"])
