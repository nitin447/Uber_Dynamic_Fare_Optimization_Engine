# 🚖 Uber Dynamic Fare Optimization Engine

AI-powered **Dynamic Pricing Engine** inspired by Uber's surge pricing model.  
This project predicts cab fares using **Artificial Neural Networks (ANN)**, finds the **optimal fare** using **Reinforcement Learning (RL)**, and explains decisions with **Explainable AI (SHAP)**.  
It also generates a **one-click professional PDF report** for business insights.  

---

## ✨ Features

- 🧠 **ANN-based Price Prediction** — predicts ride fares based on distance, weather, surge, and time.  
- 🤖 **Reinforcement Learning (RL)** — finds the **optimal price** that maximizes revenue.  
- 🔍 **Explainability with SHAP** — shows why the model predicts higher/lower fares.  
- 📈 **Revenue & Demand Curves** — visualize how pricing impacts demand and revenue.  
- 📑 **One-Click PDF Report** — generates a detailed business report with results & charts.  

---

## ⚙️ Tech Stack

- **Python 3.9+**  
- **Streamlit** → Interactive dashboard  
- **TensorFlow / Keras** → ANN model for fare prediction  
- **Scikit-learn** → Preprocessing, scaling, encoding  
- **SHAP** → Explainability (feature importance)  
- **FPDF** → Automated PDF report generation  
- **Matplotlib** → Visualization  

---

## 📊 Model Performance

- **R² = 92%** (Model explains 92% variance in cab fares)  
- **RMSE = 3.5** (Average prediction error ~3.5 currency units)  

---

## 🚀 How to Run Locally

```bash
# Clone this repository
git clone https://github.com/<your-username>/uber-dynamic-fare-optimization-engine.git
cd uber-dynamic-fare-optimization-engine

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app3.py
## 🌐 Live Demo
Try the live demo here:  
👉 https://uberdynamicfareoptimizationengine.streamlit.app/

