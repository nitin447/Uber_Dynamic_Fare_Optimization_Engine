# app3.py — restored UI with robust download, caching, lazy loads and graceful fallbacks
import os
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow import keras
from fpdf import FPDF
import gdown

# ---------------------------- Config / constants ----------------------------
DRIVE_FILE_ID = "1Y0AgBw5hotx0ucjuzwkBvo-ArpO8ZS2U"
LOCAL_FNAME = "cleaned_dataset.csv"
ANN_MODEL_PATH = "ann_model.keras"  # ensure model file exists or upload

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")
st.title("🚖 Uber's Dynamic Fare Optimization Engine ")

# ---------------------------- Data loader with fallback ----------------------------
@st.cache_data(show_spinner=False)
def get_data_from_drive(file_id: str, fname: str = LOCAL_FNAME) -> pd.DataFrame:
    """Try local file, then gdown id-download, then fallback export=download URL."""
    if os.path.exists(fname):
        return pd.read_csv(fname)

    # Try gdown id download first (preferred)
    try:
        gdown.download(id=file_id, output=fname, quiet=False)
        if os.path.exists(fname):
            return pd.read_csv(fname)
    except Exception:
        pass

    # Fallback export=download
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, fname, quiet=False)
        if os.path.exists(fname):
            return pd.read_csv(fname)
    except Exception:
        pass

    raise FileNotFoundError("Failed to download dataset from Drive. Use uploader or place file in repo root.")

# ---------------------------- Cached model & preprocessor loaders ----------------------------
@st.cache_resource(show_spinner=False)
def load_ann_model(path: str = ANN_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at `{path}`.")
    model = keras.models.load_model(path)
    return model

@st.cache_resource(show_spinner=False)
def build_preprocessor(df: pd.DataFrame, categorical_cols, numerical_cols):
    X_full = df[categorical_cols + numerical_cols]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )
    pre.fit(X_full)
    return pre

# ---------------------------- Dataset selection UI ----------------------------
st.sidebar.header("⚙️ Data & Model")

data_load_mode = st.sidebar.radio("Dataset source:",
                                  ["Download from Drive", "Use uploaded file", "Use repo local (if present)"])

df = None
if data_load_mode == "Use repo local (if present)":
    if os.path.exists(LOCAL_FNAME):
        try:
            df = pd.read_csv(LOCAL_FNAME)
            st.sidebar.success("Loaded dataset from repo local file.")
        except Exception as e:
            st.sidebar.error("Local dataset exists but failed to read.")
            st.sidebar.exception(e)
            df = None
    else:
        st.sidebar.info("No local dataset in repo root.")
        df = None

if data_load_mode == "Use uploaded file":
    uploaded = st.sidebar.file_uploader("Upload cleaned_dataset.csv", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success("Uploaded dataset loaded.")
        except Exception as e:
            st.sidebar.error("Uploaded file couldn't be read as CSV.")
            st.sidebar.exception(e)
            df = None

if data_load_mode == "Download from Drive":
    try:
        with st.spinner("Downloading dataset from Google Drive..."):
            df = get_data_from_drive(DRIVE_FILE_ID, LOCAL_FNAME)
            st.sidebar.success("Dataset downloaded & cached.")
    except Exception as e:
        st.sidebar.error("Failed to download dataset from Drive.")
        st.sidebar.info("You can upload the dataset via 'Use uploaded file' or place cleaned_dataset.csv in the repo root.")
        st.sidebar.exception(e)
        # Show uploader in main UI too for convenience
        uploaded_fallback = st.file_uploader("If automatic download failed, upload cleaned_dataset.csv here", type=["csv"])
        if uploaded_fallback is not None:
            try:
                df = pd.read_csv(uploaded_fallback)
                st.success("Uploaded dataset loaded.")
            except Exception as e2:
                st.error("Uploaded file couldn't be read.")
                st.exception(e2)
        else:
            st.stop()

if df is None:
    st.error("Dataset not loaded. Please provide dataset using sidebar or uploader.")
    st.stop()

# ---------------------------- Preprocessor & model (cached) ----------------------------
categorical_cols = ["cab_type", "source", "destination", "name"]
numerical_cols = ["distance", "surge_multiplier", "temp", "clouds", "pressure",
                  "rain", "humidity", "wind", "hour", "dayofweek", "is_weekend"]

try:
    preprocessor = build_preprocessor(df, categorical_cols, numerical_cols)
except Exception as e:
    st.error("Failed to build preprocessor — check dataset columns.")
    st.exception(e)
    st.stop()

# Load model (lazy, cached)
try:
    ann_model = load_ann_model(ANN_MODEL_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Place 'ann_model.keras' in repo root or upload it (not implemented here).")
    st.stop()
except Exception as e:
    st.error("Failed to load ANN model.")
    st.exception(e)
    st.stop()

# ---------------------------- Compute metrics (avoid hashing preprocessor/model) ----------------------------
@st.cache_data(show_spinner=False)
def compute_metrics_and_encoded(df, _preprocessor, _model, categorical_cols, numerical_cols):
    X_full = df[categorical_cols + numerical_cols]
    y_true = df["price"].values
    X_full_enc = _preprocessor.transform(X_full)
    y_pred = _model.predict(X_full_enc).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "r2": r2, "X_full_enc": X_full_enc, "y_pred": y_pred, "y_true": y_true}

metrics = compute_metrics_and_encoded(df, preprocessor, ann_model, categorical_cols, numerical_cols)
rmse = metrics["rmse"]
r2 = metrics["r2"]

# ---------------------------- Header / description (restored) ----------------------------
st.markdown(f"""
🚖 **Dynamic Pricing Engine** uses AI to make ride fares smarter:  

- 🧠 **Artificial Neural Network (ANN)** predicts ride price from ride + weather conditions.  
- 🤖 **Reinforcement Learning (RL)** finds the **optimal fare** 🎯 that maximizes revenue.  
- 🔍 **Explainable AI (SHAP)** explains *why* the price is high or low.  

📊 *Model Performance:* R² = **{r2*100:.2f}%**, RMSE = **{rmse:.2f}**
""")

# ---------------------------- Sidebar inputs (restored) ----------------------------
st.sidebar.header("📌 Ride Details")
distance = st.sidebar.slider("Distance (km)", 1, 30, 5)
cab_type = st.sidebar.selectbox("Cab Type", df["cab_type"].unique())
ride_type = st.sidebar.selectbox("Ride Option", df["name"].unique())
source = st.sidebar.selectbox("Pickup Location", df["source"].unique())
destination = st.sidebar.selectbox("Drop Location", df["destination"].unique())
hour = st.sidebar.slider("Hour of Day", 0, 23, 18)
dayofweek = st.sidebar.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
is_weekend = 1 if dayofweek in ["Sat","Sun"] else 0
surge = st.sidebar.slider("Surge Multiplier", 1.0, 3.0, 1.0)
temp = st.sidebar.slider("Temperature (°C)", -5, 40, 20)
rain = st.sidebar.slider("Rain (mm)", 0.0, 20.0, 0.0)
humidity = st.sidebar.slider("Humidity", 0.0, 1.0, 0.6)
cost = st.sidebar.slider("Estimated Ride Cost (Driver/Fuel)", 5, 50, 20)

# ---------------------------- Input prep & encoding ----------------------------
input_data = pd.DataFrame([{
    "distance": distance, "cab_type": cab_type, "source": source, "destination": destination,
    "surge_multiplier": surge, "name": ride_type, "temp": temp, "clouds": 0.5, "pressure": 1013,
    "rain": rain, "humidity": humidity, "wind": 5, "hour": hour,
    "dayofweek": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(dayofweek),
    "is_weekend": is_weekend
}])

try:
    X_enc = preprocessor.transform(input_data)
except Exception as e:
    st.error("Failed to transform input data using preprocessor.")
    st.exception(e)
    st.stop()

# ---------------------------- Predictions, RL proxy & metrics (restored) ----------------------------
predicted_price = ann_model.predict(X_enc)[0][0]

# original used predicted_price as demand proxy; keep same
price_range = np.arange(5, 100, 5)
Q_values = [price * predicted_price for price in price_range]
optimal_price = price_range[np.argmax(Q_values)]
optimal_revenue = max(Q_values)
optimal_demand = predicted_price
profit = optimal_price * predicted_price - cost

# ---------------------------- Tabs (full UI restored) ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction", "📈 Revenue Curves", "🔍 Explainability", "📑 Report"])

with tab1:
    st.subheader("📊 Prediction Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Price (ANN)", f"${predicted_price:.2f}")
    with col2:
        st.metric("Optimal Price (RL)", f"${optimal_price:.2f}")
    with col3:
        st.metric("Estimated Profit", f"${profit:.2f}")

    st.markdown("---")
    st.subheader("🌀 Scenario Simulation")
    st.caption("""
    Change inputs to see their effect on predicted fare:  

    - ⚡ **Surge Multiplier** → demand–supply pressure (higher surge → fares UP 💹)  
    - 🌧️ **Rain** → bad weather increases demand & traffic → higher fares  
    """)
    sim_surge = st.slider("Simulated Surge Multiplier", 1.0, 3.0, surge)
    sim_rain = st.slider("Simulated Rain (mm)", 0.0, 20.0, rain)
    sim_input = input_data.copy()
    sim_input["surge_multiplier"] = sim_surge
    sim_input["rain"] = sim_rain
    sim_X_enc = preprocessor.transform(sim_input)
    sim_price = ann_model.predict(sim_X_enc)[0][0]
    st.metric("Simulated Predicted Price", f"${sim_price:.2f}")
    st.caption("🔧 Simulation helps businesses plan strategies for changing market conditions.")

with tab2:
    st.subheader("📈 Revenue & Demand Analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # demand proxy across prices (using predicted_price as proxy)
    ax[0].plot(price_range, [predicted_price for _ in price_range], marker="o")
    ax[0].set_title("Price vs Demand Curve")
    ax[0].set_xlabel("Price ($)"); ax[0].set_ylabel("Demand (proxy)")
    ax[1].plot(price_range, Q_values, marker="o", color="orange")
    ax[1].set_title("Price vs Revenue Curve")
    ax[1].set_xlabel("Price ($)"); ax[1].set_ylabel("Revenue")
    st.pyplot(fig)

    st.markdown(f"""
    ### 📊 Dynamic Revenue Summary  

    - 🎯 **Optimal Price:** ${optimal_price:.2f}  
    - 📉 **Demand at this price (proxy):** ~{optimal_demand:.0f} rides  
    - 💰 **Revenue at this price:** ~${optimal_revenue:.2f}  

    **Interpretation:**  
    - As fares rise, demand falls 📉.  
    - Revenue rises initially 💹 but drops after fares become too high.  
    - The peak point (${optimal_price:.2f}) balances price & demand, ensuring **maximum revenue**.  
    """)

with tab3:
    st.subheader("🔍 Explainable AI (SHAP)")
    st.caption("Positive values increase predicted price, negative values reduce it.")
    # Lazy-import and compute SHAP explanations only when user opens this tab
    try:
        import shap
    except Exception as e:
        st.error("SHAP is not installed in this environment.")
        st.info("Install shap if you want explainability (may be memory intensive).")
        st.exception(e)
    else:
        with st.spinner("Computing SHAP explanations (may take time)..."):
            try:
                explainer = shap.Explainer(ann_model, X_enc)
                shap_values = explainer(X_enc)
            except Exception as e:
                st.error("Failed to compute SHAP explanations (possible memory/CPU limits).")
                st.exception(e)
            else:
                fig, ax = plt.subplots(figsize=(10,6))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig)

                # feature names from preprocessor
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except Exception:
                    feature_names = [f"f{i}" for i in range(len(shap_values.values[0]))]

                shap_df = pd.DataFrame({
                    "feature": feature_names,
                    "shap_value": shap_values.values[0]
                }).sort_values(by="shap_value", key=abs, ascending=False)

                top_positive = shap_df[shap_df["shap_value"] > 0].head(2)
                top_negative = shap_df[shap_df["shap_value"] < 0].head(2)

                st.markdown("### 📝 Dynamic SHAP Summary")
                for _, row in top_positive.iterrows():
                    st.write(f"✅ **{row['feature']}** increased price by **{row['shap_value']:.2f}**")
                for _, row in top_negative.iterrows():
                    st.write(f"❌ **{row['feature']}** decreased price by **{abs(row['shap_value']):.2f}**")

                st.markdown("""
                **Business Insight:**  
                - Features like distance and surge multiplier usually dominate → fares rise 🚀.  
                - Weather and time-related features add smaller adjustments.  
                """)

with tab4:
    st.subheader("📑 Generate Report")
    if st.button("Create PDF Report"):
        class PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 14)
                self.cell(0, 10, "Dynamic Pricing Engine Report", align="C", ln=True)
                self.ln(5)
            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.cell(0, 10, f"Generated on {datetime.date.today()} | Page {self.page_no()}", align="C")

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "", 12)

        # Prediction summary
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Prediction Results", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 8,
            f"Predicted Price (ANN): ${predicted_price:.2f}\n"
            f"Optimal Price (RL): ${optimal_price:.2f}\n"
            f"Estimated Profit: ${profit:.2f}"
        )

        # Scenario
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Scenario Simulation", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 8,
            f"With Surge={sim_surge} and Rain={sim_rain}, predicted price = ${sim_price:.2f}."
        )

        # SHAP summary best-effort
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Explainable AI Summary", ln=True)
        pdf.set_font("Helvetica", "", 11)
        try:
            for _, row in top_positive.iterrows():
                pdf.multi_cell(0, 8, f"{row['feature']} increased price by {row['shap_value']:.2f}")
            for _, row in top_negative.iterrows():
                pdf.multi_cell(0, 8, f"{row['feature']} decreased price by {abs(row['shap_value']):.2f}")
        except Exception:
            pdf.multi_cell(0, 8, "SHAP summary not available.")

        # Chart images
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(price_range, [predicted_price for _ in price_range], marker="o")
        ax[0].set_title("Price vs Demand")
        ax[1].plot(price_range, Q_values, marker="o", color="orange")
        ax[1].set_title("Price vs Revenue")
        img_path = "chart.png"
        plt.savefig(img_path, format="png", bbox_inches="tight")
        plt.close(fig)
        try:
            pdf.image(img_path, x=15, y=120, w=180)
        except Exception:
            pass

        pdf_bytes = pdf.output(dest="S").encode("latin-1", errors="ignore")
        st.download_button(
            label="📥 Download Report",
            data=pdf_bytes,
            file_name="dynamic_pricing_report.pdf",
            mime="application/pdf"
        )

# ---------------------------- End of file ----------------------------
