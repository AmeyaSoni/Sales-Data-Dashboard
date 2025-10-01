import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# Load data
df = pd.read_csv("Walmart_Sales.csv")

# Load trained model (Random Forest Regressor for forecasting)
models = joblib.load("model.pkl")
model = models['regressor']

st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("📊 Sales Data Trends & Forecasting Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
filter_cols = [col for col in df.columns if col != "Weekly_Sales" and col != "Weekly_Sales_log"]
filters = {}
for col in filter_cols:
    if df[col].dtype == 'object':
        filters[col] = st.sidebar.multiselect(f"Select {col}", options=df[col].unique(), default=list(df[col].unique()))
    else:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        filters[col] = st.sidebar.slider(f"Range for {col}", min_val, max_val, (min_val, max_val))

# Apply filters
filtered_df = df.copy()
for col, val in filters.items():
    if isinstance(val, list):
        filtered_df = filtered_df[filtered_df[col].isin(val)]
    else:
        filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1])]

# Show filtered data
st.subheader("📂 Filtered Data")
st.dataframe(filtered_df.head(20))

# Trend visualization
st.subheader("📈 Trend Analysis")
num_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
if len(num_cols) >= 2:
    x_axis = st.selectbox("X-axis", options=num_cols)
    y_axis = st.selectbox("Y-axis", options=num_cols)
    fig = px.line(filtered_df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
    st.plotly_chart(fig, use_container_width=True)

# Weekly Sales Forecasting
st.subheader("🔮 Weekly Sales Forecasting")
input_data = {}
for col in filter_cols:
    if df[col].dtype == 'object':
        input_data[col] = st.selectbox(f"Select {col}", options=df[col].unique())
    else:
        input_data[col] = st.number_input(f"Enter value for {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([input_data])

if st.button("Predict Weekly Sales"):
    # Ensure input features match model training
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = pd.factorize(df[col])[0][df[col] == input_df[col][0]][0] if input_df[col][0] in df[col].values else 0
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Weekly Sales: {prediction:.2f}")

    # Show forecast graph
    forecast_df = filtered_df.copy()
    forecast_df = forecast_df.sort_values(by="Weekly_Sales_log")
    fig2 = px.line(forecast_df, x=forecast_df.index, y="Weekly_Sales_log", title="Weekly Sales Forecast (Log Transformed)")
    st.plotly_chart(fig2, use_container_width=True)