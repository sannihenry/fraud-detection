import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Toll-Gate Intelligence Platform",
    page_icon="🚧",
    layout="wide",
)

st.title("🚧 Toll-Gate Intelligence Platform")
st.caption("Fraud Detection • Traffic Forecasting • Revenue Forecasting")

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

REQUIRED_COLUMNS = [
    "timestamp",
    "plaza_id",
    "lane",
    "vehicle_class",
    "payment_method",
    "vehicle_id",
    "amount",
]

CATEGORICAL_COLUMNS = ["plaza_id", "lane", "vehicle_class", "payment_method"]

MODEL_DIR = Path(__file__).parent


# ---------------------------------------------------------
# DEBUG / ENVIRONMENT INFO
# (helps catch version-drift crashes early next time)
# ---------------------------------------------------------

def show_environment_info():
    import sklearn
    with st.sidebar.expander("Environment info"):
        st.write(f"streamlit: {st.__version__}")
        st.write(f"pandas: {pd.__version__}")
        st.write(f"numpy: {np.__version__}")
        st.write(f"scikit-learn: {sklearn.__version__}")
        st.write(f"joblib: {joblib.__version__}")


# ---------------------------------------------------------
# MODEL LOADING (cached — only loads once per session)
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    base = Path(__file__).resolve().parent

    fraud_path = base / "fraud_model.joblib"
    volume_path = base / "forecast_volume.joblib"
    revenue_path = base / "forecast_revenue.joblib"

    try:
        fraud_bundle = joblib.load(fraud_path)
        volume_bundle = joblib.load(volume_path)
        revenue_bundle = joblib.load(revenue_path)

        return fraud_bundle, volume_bundle, revenue_bundle

    except Exception as e:
        st.error("Failed to load ML models.")
        st.exception(e)
        st.stop()

# ---------------------------------------------------------
# DATA LOADING / VALIDATION (cached per uploaded file)
# ---------------------------------------------------------

@st.cache_data(show_spinner="Reading CSV...")
def load_data(uploaded_file):
    try:
    df = pd.read_csv(uploaded_file)

    if df.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

except Exception as e:
    st.error("Unable to read CSV.")
    st.exception(e)
    st.stop()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing required column(s): {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        n_bad = df["timestamp"].isna().sum()
        st.warning(f"{n_bad} row(s) had an unparseable timestamp and were dropped.")
        df = df.dropna(subset=["timestamp"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if df["amount"].isna().any():
        n_bad = df["amount"].isna().sum()
        st.warning(f"{n_bad} row(s) had a non-numeric amount and were treated as 0.")
        df["amount"] = df["amount"].fillna(0)

    return df


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def build_fraud_features(df, fraud_features):
    data = df.copy()
    data["hour"] = data["timestamp"].dt.hour
    data["dayofweek"] = data["timestamp"].dt.dayofweek

    data = pd.get_dummies(data, columns=CATEGORICAL_COLUMNS)

    for col in fraud_features:
        if col not in data.columns:
            data[col] = 0

    return data[fraud_features]


# ---------------------------------------------------------
# LOAD MODELS (fail loudly but gracefully, with real reason shown)
# ---------------------------------------------------------

show_environment_info()

try:
    fraud_model, fraud_features, volume_model, revenue_model = load_models()
    st.sidebar.success("Models loaded")
except Exception as e:
    st.error(
        "Unable to load one or more models. This is very often caused by a "
        "scikit-learn / numpy version mismatch between when the model was "
        "trained and what's currently installed on the server.\n\n"
        f"**Details:** {e}"
    )
    st.info(
        "Fix: pin exact versions in requirements.txt (the ones used when you "
        "trained/pickled the model), e.g.\n\n"
        "```\nscikit-learn==<your_version>\nnumpy==<your_version>\n"
        "pandas==<your_version>\njoblib==<your_version>\n```"
    )
    st.stop()

# ---------------------------------------------------------
# SIDEBAR — UPLOAD
# ---------------------------------------------------------

st.sidebar.header("Upload Dataset")
uploaded = st.sidebar.file_uploader("Choose Toll Transaction CSV", type=["csv"])
st.sidebar.markdown("---")

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not read the uploaded CSV.\n\n**Details:** {e}")
    st.stop()

# ---------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(df):,}")
c2.metric("Revenue", f"₦{df['amount'].sum():,.0f}")
c3.metric("Plazas", df["plaza_id"].nunique())
c4.metric("Vehicles", f"{df['vehicle_id'].nunique():,}")

st.divider()

# ---------------------------------------------------------
# FRAUD DETECTION
# ---------------------------------------------------------

st.header("Fraud Detection")

results = df.copy()

try:
    X = build_fraud_features(df, fraud_features)
    prob = fraud_model.predict_proba(X)[:, 1]
    pred = (prob >= 0.50).astype(int)

    results["fraud_probability"] = prob
    results["prediction"] = np.where(pred == 1, "Fraud", "Legitimate")

    fraud_cases = int((results["prediction"] == "Fraud").sum())
    fraud_rate = fraud_cases / len(results) if len(results) else 0

    a, b = st.columns(2)
    a.metric("Fraud Alerts", fraud_cases)
    b.metric("Fraud Rate", f"{fraud_rate:.2%}")

    st.dataframe(
        results.sort_values("fraud_probability", ascending=False).head(25),
        use_container_width=True,
    )

    st.download_button(
        "Download Prediction Results",
        results.to_csv(index=False),
        "fraud_predictions.csv",
        "text/csv",
    )
except Exception as e:
    st.warning(f"Fraud detection could not be run.\n\n**Details:** {e}")

# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------

st.divider()
left, right = st.columns(2)

with left:
    try:
        if "fraud_probability" in results.columns:
            fig = px.histogram(
                results, x="fraud_probability", nbins=40,
                title="Fraud Probability Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render fraud distribution chart.\n\n{e}")

with right:
    try:
        fig = px.bar(
            results.groupby("vehicle_class")["amount"].sum().reset_index(),
            x="vehicle_class", y="amount", title="Revenue by Vehicle Class",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render vehicle class chart.\n\n{e}")

try:
    fig = px.bar(
        results.groupby("plaza_id")["amount"].sum().reset_index(),
        x="plaza_id", y="amount", title="Revenue by Plaza",
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render revenue-by-plaza chart.\n\n{e}")

try:
    traffic = (
        df.set_index("timestamp")
        .groupby("plaza_id")
        .resample("h")
        .size()
        .reset_index(name="vehicle_count")
    )
    fig = px.line(
        traffic, x="timestamp", y="vehicle_count", color="plaza_id",
        title="Hourly Traffic",
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render traffic chart.\n\n{e}")

# ---------------------------------------------------------
# FORECAST
# ---------------------------------------------------------

st.header("Forecast")
st.info("Forecasts are generated using the latest available hourly data.")

try:
    agg = (
        df.set_index("timestamp")
        .groupby("plaza_id")
        .resample("h")
        .agg(vehicle_count=("vehicle_id", "count"), revenue=("amount", "sum"))
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

    numeric_features = latest.select_dtypes(include=np.number)

    latest["Predicted Next Hour Traffic"] = volume_model.predict(numeric_features)
    latest["Predicted Next Hour Revenue"] = revenue_model.predict(numeric_features)

    st.dataframe(
        latest[["plaza_id", "Predicted Next Hour Traffic", "Predicted Next Hour Revenue"]],
        use_container_width=True,
    )
except Exception as e:
    st.warning(f"Forecast could not be generated.\n\n**Details:** {e}")

# ---------------------------------------------------------
# RAW DATA
# ---------------------------------------------------------

with st.expander("View Uploaded Dataset"):
    st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")
st.caption(
    "Prototype built for demonstrating AI-powered fraud detection and "
    "traffic intelligence for toll management systems."
)
