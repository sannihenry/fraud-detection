import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Toll-Gate Intelligence Platform",
    page_icon="🚧",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🚧 AI-Powered Smart Toll Intelligence Platform")
st.caption("Fraud Detection • Traffic Forecasting • Revenue Forecasting")

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

# MODEL_DIR = Path("models")
# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

try:
    fraud_bundle = joblib.load("fraud_model.joblib")

    fraud_model = fraud_bundle["model"]
    fraud_features = fraud_bundle["feature_columns"]

    volume_model = joblib.load("forecast_volume.joblib")
    revenue_model = joblib.load("forecast_revenue.joblib")

except Exception as e:
    st.error(f"Unable to load models.\n\n{e}")
    st.stop()

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def build_fraud_features(df):

    data = df.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    data["hour"] = data["timestamp"].dt.hour
    data["dayofweek"] = data["timestamp"].dt.dayofweek

    categorical = [
        "plaza_id",
        "lane",
        "vehicle_class",
        "payment_method"
    ]

    data = pd.get_dummies(data, columns=categorical)

    for col in fraud_features:
        if col not in data.columns:
            data[col] = 0

    data = data[fraud_features]

    return data

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Upload Dataset")

uploaded = st.sidebar.file_uploader(
    "Choose Toll Transaction CSV",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.success("Models Loaded")

# ---------------------------------------------------------
# NO DATA
# ---------------------------------------------------------

if uploaded is None:

    st.info("Upload a CSV file to begin.")

    st.stop()

# ---------------------------------------------------------
# READ DATA
# ---------------------------------------------------------

df = pd.read_csv(uploaded)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ---------------------------------------------------------
# KPI
# ---------------------------------------------------------

total_transactions = len(df)

total_revenue = df["amount"].sum()

num_plazas = df["plaza_id"].nunique()

vehicle_count = df["vehicle_id"].nunique()

c1,c2,c3,c4 = st.columns(4)

c1.metric("Transactions", f"{total_transactions:,}")

c2.metric("Revenue", f"₦{total_revenue:,.0f}")

c3.metric("Plazas", num_plazas)

c4.metric("Vehicles", f"{vehicle_count:,}")

st.divider()

# ---------------------------------------------------------
# FRAUD DETECTION
# ---------------------------------------------------------

st.header("Fraud Detection")

X = build_fraud_features(df)

prob = fraud_model.predict_proba(X)[:,1]

pred = (prob >= 0.50).astype(int)

results = df.copy()

results["fraud_probability"] = prob

results["prediction"] = np.where(pred==1,"Fraud","Legitimate")

fraud_cases = (results["prediction"]=="Fraud").sum()

fraud_rate = fraud_cases/len(results)

a,b = st.columns(2)

a.metric(
    "Fraud Alerts",
    fraud_cases
)

b.metric(
    "Fraud Rate",
    f"{fraud_rate:.2%}"
)

st.dataframe(

    results.sort_values(
        "fraud_probability",
        ascending=False
    ).head(25),

    use_container_width=True

)

csv = results.to_csv(index=False)

st.download_button(

    "Download Prediction Results",

    csv,

    "fraud_predictions.csv",

    "text/csv"

)

# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------

st.divider()

left,right = st.columns(2)

with left:

    fig = px.histogram(

        results,

        x="fraud_probability",

        nbins=40,

        title="Fraud Probability Distribution"

    )

    st.plotly_chart(fig,use_container_width=True)

with right:

    fig = px.bar(

        results.groupby("vehicle_class")["amount"]

        .sum()

        .reset_index(),

        x="vehicle_class",

        y="amount",

        title="Revenue by Vehicle Class"

    )

    st.plotly_chart(fig,use_container_width=True)

# ---------------------------------------------------------
# REVENUE BY PLAZA
# ---------------------------------------------------------

fig = px.bar(

    results.groupby("plaza_id")["amount"]

    .sum()

    .reset_index(),

    x="plaza_id",

    y="amount",

    title="Revenue by Plaza"

)

st.plotly_chart(fig,use_container_width=True)

# ---------------------------------------------------------
# TRAFFIC
# ---------------------------------------------------------

traffic = (

    df

    .set_index("timestamp")

    .groupby("plaza_id")

    .resample("H")

    .size()

    .reset_index(name="vehicle_count")

)

fig = px.line(

    traffic,

    x="timestamp",

    y="vehicle_count",

    color="plaza_id",

    title="Hourly Traffic"

)

st.plotly_chart(fig,use_container_width=True)

# ---------------------------------------------------------
# FORECAST
# ---------------------------------------------------------

st.header("Forecast")

st.info(
    "Forecasts are generated using the latest available hourly data."
)

try:

    agg = (

        df

        .set_index("timestamp")

        .groupby("plaza_id")

        .resample("H")

        .agg(

            vehicle_count=("vehicle_id","count"),

            revenue=("amount","sum")

        )

        .reset_index()

    )

    latest = agg.groupby("plaza_id").tail(1).copy()

    latest["hour"] = latest["timestamp"].dt.hour
    latest["dayofweek"] = latest["timestamp"].dt.dayofweek

    latest["lag1"] = latest["vehicle_count"]
    latest["lag24"] = latest["vehicle_count"]
    latest["lag168"] = latest["vehicle_count"]

    latest["rev_lag1"] = latest["revenue"]
    latest["rev_lag24"] = latest["revenue"]
    latest["rev_lag168"] = latest["revenue"]

    volume_prediction = volume_model.predict(
        latest.select_dtypes(include=np.number)
    )

    revenue_prediction = revenue_model.predict(
        latest.select_dtypes(include=np.number)
    )

    latest["Predicted Next Hour Traffic"] = volume_prediction
    latest["Predicted Next Hour Revenue"] = revenue_prediction

    st.dataframe(

        latest[

            [

                "plaza_id",

                "Predicted Next Hour Traffic",

                "Predicted Next Hour Revenue"

            ]

        ],

        use_container_width=True

    )

except Exception as e:

    st.warning(
        f"Forecast could not be generated.\n\n{e}"
    )

# ---------------------------------------------------------
# RAW DATA
# ---------------------------------------------------------

with st.expander("View Uploaded Dataset"):

    st.dataframe(df,use_container_width=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Prototype built for demonstrating AI-powered fraud detection and traffic intelligence for toll management systems."
)
