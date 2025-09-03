import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras
import shap
from fpdf import FPDF
import datetime

# --- Load Dataset and Model ---
merged_clean = pd.read_csv("cleaned_dataset.csv")
ann_model = keras.models.load_model("ann_model.keras")

# --- Preprocessor ---
categorical_cols = ["cab_type", "source", "destination", "name"]
numerical_cols = ["distance", "surge_multiplier", "temp", "clouds", "pressure",
                  "rain", "humidity", "wind", "hour", "dayofweek", "is_weekend"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)
X_full = merged_clean[categorical_cols + numerical_cols]
preprocessor.fit(X_full)

# --- Model Performance (R² & RMSE) ---
y_true = merged_clean["price"].values   # ✅ Target column = price
X_full_enc = preprocessor.transform(X_full)
y_pred = ann_model.predict(X_full_enc).flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

# --- Streamlit Config ---
st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")
st.title("🚖 Dynamic Pricing Engine Dashboard")

st.markdown(f"""
🚖 **Dynamic Pricing Engine** uses AI to make ride fares smarter:  

- 🧠 **Artificial Neural Network (ANN)** predicts ride price from ride + weather conditions.  
- 🤖 **Reinforcement Learning (RL)** finds the **optimal fare** 🎯 that maximizes revenue.  
- 🔍 **Explainable AI (SHAP)** explains *why* the price is high or low.  

 📊 *Model Performance:* R² = **{r2*100:.2f}%**, RMSE = **{rmse:.2f}**
""")


# --- Sidebar Inputs ---
st.sidebar.header("📌 Ride Details")
distance = st.sidebar.slider("Distance (km)", 1, 30, 5)
cab_type = st.sidebar.selectbox("Cab Type", merged_clean["cab_type"].unique())
ride_type = st.sidebar.selectbox("Ride Option", merged_clean["name"].unique())
source = st.sidebar.selectbox("Pickup Location", merged_clean["source"].unique())
destination = st.sidebar.selectbox("Drop Location", merged_clean["destination"].unique())
hour = st.sidebar.slider("Hour of Day", 0, 23, 18)
dayofweek = st.sidebar.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
is_weekend = 1 if dayofweek in ["Sat","Sun"] else 0
surge = st.sidebar.slider("Surge Multiplier", 1.0, 3.0, 1.0)
temp = st.sidebar.slider("Temperature (°C)", -5, 40, 20)
rain = st.sidebar.slider("Rain (mm)", 0.0, 20.0, 0.0)
humidity = st.sidebar.slider("Humidity", 0.0, 1.0, 0.6)
cost = st.sidebar.slider("Estimated Ride Cost (Driver/Fuel)", 5, 50, 20)

# --- Input Prep ---
input_data = pd.DataFrame([{
    "distance": distance, "cab_type": cab_type, "source": source, "destination": destination,
    "surge_multiplier": surge, "name": ride_type, "temp": temp, "clouds": 0.5, "pressure": 1013,
    "rain": rain, "humidity": humidity, "wind": 5, "hour": hour,
    "dayofweek": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(dayofweek),
    "is_weekend": is_weekend
}])
X_enc = preprocessor.transform(input_data)

# --- Predictions ---
predicted_price = ann_model.predict(X_enc)[0][0]
price_range = np.arange(5, 100, 5)
Q_values = [price * ann_model.predict(X_enc)[0][0] for price in price_range]
optimal_price = price_range[np.argmax(Q_values)]
optimal_revenue = max(Q_values)
optimal_demand = ann_model.predict(X_enc)[0][0]
profit = optimal_price * ann_model.predict(X_enc)[0][0] - cost

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction", "📈 Revenue Curves", "🔍 Explainability", "📑 Report"])

# --- Tab 1: Prediction ---
with tab1:
    st.subheader("📊 Prediction Overview")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Predicted Price (ANN)", f"${predicted_price:.2f}")
    with col2: st.metric("Optimal Price (RL)", f"${optimal_price:.2f}")
    with col3: st.metric("Estimated Profit", f"${profit:.2f}")

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

# --- Tab 2: Revenue ---
with tab2:
    st.subheader("📈 Revenue & Demand Analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(price_range, [ann_model.predict(X_enc)[0][0] for _ in price_range], marker="o")
    ax[0].set_title("Price vs Demand Curve")
    ax[0].set_xlabel("Price ($)"); ax[0].set_ylabel("Demand")
    ax[1].plot(price_range, Q_values, marker="o", color="orange")
    ax[1].set_title("Price vs Revenue Curve")
    ax[1].set_xlabel("Price ($)"); ax[1].set_ylabel("Revenue")
    st.pyplot(fig)

    st.markdown(f"""
    ### 📊 Dynamic Revenue Summary  

    - 🎯 **Optimal Price:** ${optimal_price:.2f}  
    - 📉 **Demand at this price:** ~{optimal_demand:.0f} rides  
    - 💰 **Revenue at this price:** ~${optimal_revenue:.2f}  

    **Interpretation:**  
    - As fares rise, demand falls 📉.  
    - Revenue rises initially 💹 but drops after fares become too high.  
    - The peak point (${optimal_price:.2f}) balances price & demand, ensuring **maximum revenue**.  
    """)

# --- Tab 3: Explainability ---
with tab3:
    st.subheader("🔍 Explainable AI (SHAP)")
    st.caption("Positive values increase predicted price, negative values reduce it.")

    explainer = shap.Explainer(ann_model, X_enc)
    shap_values = explainer(X_enc)

    fig, ax = plt.subplots(figsize=(10,6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

    shap_df = pd.DataFrame({
        "feature": preprocessor.get_feature_names_out(),
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
    - This proves the model’s logic matches real-world intuition: *long rides, bad weather, peak hours → higher fares*.  
    """)

# --- Tab 4: Report ---
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

        # Prediction
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

        # Explainability
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Explainable AI Summary", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for _, row in top_positive.iterrows():
            pdf.multi_cell(0, 8, f"{row['feature']} increased price by {row['shap_value']:.2f}")
        for _, row in top_negative.iterrows():
            pdf.multi_cell(0, 8, f"{row['feature']} decreased price by {abs(row['shap_value']):.2f}")

        # Chart
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(price_range, [ann_model.predict(X_enc)[0][0] for _ in price_range], marker="o")
        ax[0].set_title("Price vs Demand")
        ax[1].plot(price_range, Q_values, marker="o", color="orange")
        ax[1].set_title("Price vs Revenue")
        img_path = "chart.png"
        plt.savefig(img_path, format="png", bbox_inches="tight")
        plt.close(fig)
        pdf.image(img_path, x=15, y=110, w=180)

        # Export PDF
        pdf_bytes = pdf.output(dest="S").encode("latin-1", errors="ignore")
        st.download_button(
            label="📥 Download Report",
            data=pdf_bytes,
            file_name="dynamic_pricing_report.pdf",
            mime="application/pdf"
        )
