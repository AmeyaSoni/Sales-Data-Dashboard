import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib

# ---- Page Setup & CSS ----
def set_page_config():
    st.set_page_config(
        page_title="Sales Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #f7f9fb;
        }
        .stHeader, .css-1d391kg {
            background-color: #4f5d75 !important;
        }
        .stSidebar {
            background-color: #e9ecf5 !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2d3142;
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
        }
        .stPlotlyChart {
            background-color: #fff !important;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(44,62,80,0.10);
            padding: 16px;
        }
        .block-container {
            padding-top: 32px;
        }
        .css-1v0mbdj, .css-1c7y2kd, .css-1lcbmhc {
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(44,62,80,0.08);
            margin-bottom: 24px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6a89cc 0%, #b8a9c9 100%);
            color: #fff;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(44,62,80,0.08);
            transition: background 0.2s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #4f5d75 0%, #6a89cc 100%);
        }
        .stDownloadButton>button {
            background: #4f5d75;
            color: #fff;
            border-radius: 8px;
            font-weight: 600;
        }
        .stDownloadButton>button:hover {
            background: #6a89cc;
        }
        .stMetric {
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(44,62,80,0.08);
            padding: 16px;
            margin-bottom: 16px;
            border: 2px solid #b8a9c9;
        }
        </style>
    """, unsafe_allow_html=True)

def load_css():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    body, .css-18e3th9 {
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
        background-color: #f7f9fb;
    }
    /* KPI card styling */
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4f5d75;
    }
    .kpi-delta-positive {
        color: #2ecc71; /* green */
        font-weight: 600;
    }
    .kpi-delta-negative {
        color: #e74c3c; /* red */
        font-weight: 600;
    }
    .kpi-label {
        font-size: 1.1rem;
        color: #6a89cc;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-card {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(44,62,80,0.08);
        padding: 20px 24px;
        margin-bottom: 16px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ---- Data Loading & Caching ----
@st.cache_data(show_spinner=True)
def load_data():
    try:
        df = pd.read_csv("Walmart_Sales.csv", parse_dates=["Date"])
        df.sort_values('Date', inplace=True)
        return df
    except FileNotFoundError:
        st.error("⚠️ Data file 'Walmart_Sales.csv' not found. Please upload it.")
        return None

@st.cache_resource
def load_model():
    try:
        models = joblib.load("model.pkl")
        return models.get('regressor', None)
    except FileNotFoundError:
        st.warning("⚠️ Model file 'model.pkl' not found. Forecasting disabled.")
        return None

# ---- Helper Functions ----
def calculate_period_sums(df, date_col="Date", value_col="Weekly_Sales", freq="M"):
    """
    Calculate sums for current and previous period and percentage delta.
    Returns (current_sum, previous_sum, delta_percent)
    """
    df_ = df.copy()
    df_["Period"] = df_[date_col].dt.to_period(freq)
    sums = df_.groupby("Period")[value_col].sum().sort_index()
    if len(sums) < 2:
        return 0, 0, 0
    current = sums.iloc[-1]
    previous = sums.iloc[-2]
    delta = ((current - previous) / previous * 100) if previous != 0 else 0
    return current, previous, delta

def calculate_yoy_growth(df, date_col="Date", value_col="Weekly_Sales", freq="M"):
    """
    Calculate Year-over-Year growth for the latest period.
    """
    df_ = df.copy()
    df_["Period"] = df_[date_col].dt.to_period(freq)
    sums = df_.groupby("Period")[value_col].sum().sort_index()
    if len(sums) < 13:
        return 0, 0, 0
    current = sums.iloc[-1]
    previous = sums.iloc[-13]  # same month last year
    delta = ((current - previous) / previous * 100) if previous != 0 else 0
    return current, previous, delta

def display_kpi(label, current, delta, format="$,.0f", delta_label="MoM"):
    delta_color = "kpi-delta-positive" if delta >= 0 else "kpi-delta-negative"
    delta_sign = "▲" if delta >= 0 else "▼"
    
    if format:
        try:
            value_str = f"{current:{format}}"
        except (ValueError, TypeError):
                value_str = f"{current:,.2f}"
    else:
        value_str = f"{current:,.2f}"
    delta_str = f"<div style='font-size: 16px; color: #888;'>Δ {delta_label}: {delta:+,.0f}</div>" if delta is not None else ''
    kpi_html = f"""
    <div style='font-size: 24px; font-weight: bold; color:#22223b;'>{label}</div>
    <div style='font-size: 32px; color: #22223b;'>{value_str}</div>
    {delta_str}
    <hr style='margin: 8px 0;'>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)
    
def filter_dataframe(df):
    st.sidebar.header("Filters")

    # Convert pandas Timestamp to Python date
    # Convert to Python date objects to avoid Streamlit error
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
        # Ensure min_date is not greater than max_date
    if min_date > max_date:
        min_date, max_date = max_date, min_date

    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    # Filter df by selected date range
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

    # Dynamic categorical filters
    filter_cols = {
        'Store': "Region / Store",
        'Product_Category': "Product Category",
        'Product_Name': "Product",
        'Type': "Store Type",
        'IsHoliday': "Holiday Week",
        # Add 'Salesperson' if available
    }

    for col, label in filter_cols.items():
        if col in df.columns:
            options = st.sidebar.multiselect(
                f"{label} Filter",
                options=sorted(df[col].unique()),
                default=sorted(df[col].unique())
            )
            if options:
                df = df[df[col].isin(options)]

    return df


def encode_categorical(df, ref_df):
    """
    Encode categorical columns in df based on ref_df unique values.
    """
    for col in df.select_dtypes(include=['object']).columns:
        mapping = {v: i for i, v in enumerate(ref_df[col].unique())}
        df[col] = df[col].map(mapping).fillna(0)
    return df

# ---- Main Dashboard ----
def main():
    set_page_config()
    load_css()

    df = load_data()
    model = load_model()
    if df is None:
        return

    st.markdown("<h1 style='color:#22223b; font-weight:700; text-align:center;'>Sales Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#22223b; font-weight:600;'>Key Performance Indicators</h2>", unsafe_allow_html=True)

    filtered_df = filter_dataframe(df)
    kpi_cols = st.columns(6)

    # Total Sales MoM and YoY
    total_sales, prev_sales, delta_mom = calculate_period_sums(filtered_df, freq="M")
    _, _, delta_yoy = calculate_yoy_growth(filtered_df, freq="M")
    with kpi_cols[0]:
        display_kpi("Total Sales", total_sales, delta_mom, format="$,.0f", delta_label="MoM")
    with kpi_cols[1]:
        display_kpi("Total Sales YoY", total_sales, delta_yoy, format="$,.0f", delta_label="YoY")

    # Total Orders MoM and YoY
    total_orders = filtered_df.shape[0]
    prev_orders = df[(df['Date'] >= (filtered_df['Date'].min() - pd.DateOffset(months=1))) & (df['Date'] < filtered_df['Date'].min())].shape[0]
    delta_orders_mom = ((total_orders - prev_orders) / prev_orders * 100) if prev_orders != 0 else 0
    prev_orders_yoy = df[(df['Date'] >= (filtered_df['Date'].min() - pd.DateOffset(years=1))) & (df['Date'] < (filtered_df['Date'].min() - pd.DateOffset(years=1) + pd.DateOffset(months=1)))].shape[0]
    delta_orders_yoy = ((total_orders - prev_orders_yoy) / prev_orders_yoy * 100) if prev_orders_yoy != 0 else 0
    with kpi_cols[2]:
        display_kpi("Total Orders", total_orders, delta_orders_mom, format=",", delta_label="MoM")
    with kpi_cols[3]:
        display_kpi("Total Orders YoY", total_orders, delta_orders_yoy, format=",", delta_label="YoY")

    # Average Order Value MoM and YoY
    avg_order_val = filtered_df['Weekly_Sales'].mean() if 'Weekly_Sales' in filtered_df.columns else 0
    prev_avg_order_val = df[(df['Date'] >= (filtered_df['Date'].min() - pd.DateOffset(months=1))) & (df['Date'] < filtered_df['Date'].min())]['Weekly_Sales'].mean()
    delta_avg_order_mom = ((avg_order_val - prev_avg_order_val) / prev_avg_order_val * 100) if prev_avg_order_val and prev_avg_order_val != 0 else 0
    prev_avg_order_val_yoy = df[(df['Date'] >= (filtered_df['Date'].min() - pd.DateOffset(years=1))) & (df['Date'] < (filtered_df['Date'].min() - pd.DateOffset(years=1) + pd.DateOffset(months=1)))]['Weekly_Sales'].mean()
    delta_avg_order_yoy = ((avg_order_val - prev_avg_order_val_yoy) / prev_avg_order_val_yoy * 100) if prev_avg_order_val_yoy and prev_avg_order_val_yoy != 0 else 0
    with kpi_cols[4]:
        display_kpi("Avg Order Value", avg_order_val, delta_avg_order_mom, format="$,.2f", delta_label="MoM")
    with kpi_cols[5]:
        display_kpi("Avg Order Value YoY", avg_order_val, delta_avg_order_yoy, format="$,.2f", delta_label="YoY")

    # Next row for Top Product and Top Region
    st.markdown("---")
    st.markdown("<h2 style='color:#222; font-weight:700; margin-bottom:0;'>Top Performers</h2>", unsafe_allow_html=True)
    top_cols = st.columns(2)

    # Top Product by Sales
    if 'Product_Name' in filtered_df.columns and 'Weekly_Sales' in filtered_df.columns:
        top_product_df = filtered_df.groupby('Product_Name')['Weekly_Sales'].sum()
        top_product = top_product_df.idxmax()
        top_product_val = top_product_df.max()
        with top_cols[0]:
            st.markdown(f"<div class='metric-performer' style='color:#222;'><span style='font-size:18px;font-weight:600;color:#222;'>Top Product</span><br><span style='font-size:24px;font-weight:700;color:#222;'>{top_product}</span><br><span style='font-size:16px;color:#222;'>Sales: ${top_product_val:,.0f}</span></div>", unsafe_allow_html=True)
    else:
        with top_cols[0]:
            st.markdown("<div class='metric-na' style='color:#222;'><span style='font-size:18px;font-weight:600;color:#222;'>Top Product</span><br><span style='font-size:24px;font-weight:700;color:#222;'>N/A</span></div>", unsafe_allow_html=True)

    # Top Region by Sales (Store as proxy)
    if 'Store' in filtered_df.columns and 'Weekly_Sales' in filtered_df.columns:
        top_region_df = filtered_df.groupby('Store')['Weekly_Sales'].sum()
        top_region = top_region_df.idxmax()
        top_region_val = top_region_df.max()
        with top_cols[1]:
            st.markdown(f"<div class='metric-performer' style='color:#222;'><span style='font-size:18px;font-weight:600;color:#222;'>Top Region</span><br><span style='font-size:24px;font-weight:700;color:#222;'>{top_region}</span><br><span style='font-size:16px;color:#222;'>Sales: ${top_region_val:,.0f}</span></div>", unsafe_allow_html=True)
    else:
        with top_cols[1]:
            st.markdown("<div class='metric-na' style='color:#222;'><span style='font-size:18px;font-weight:600;color:#222;'>Top Region</span><br><span style='font-size:24px;font-weight:700;color:#222;'>N/A</span></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Sales Trend Analysis
    st.markdown("<h2 style='color:#222; font-weight:700; margin-bottom:0;'>Sales Trend Analysis</h2>", unsafe_allow_html=True)
    time_agg_map = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y"
    }
    time_agg_label = st.selectbox(
        "Select Time Aggregation",
        options=list(time_agg_map.keys()),
        index=2,
        help="Choose how to aggregate sales data by time period."
    )
    time_agg = time_agg_map[time_agg_label]

    # Additional filters for trend chart
    trend_filter_cols = {
        'Store': "Region / Store",
        'Product_Category': "Product Category",
        'Product_Name': "Product",
        # Add 'Salesperson' if available
    }
    trend_filters = {}
    for col, label in trend_filter_cols.items():
        if col in filtered_df.columns:
            options = st.multiselect(f"Filter {label} (Trend Chart)", options=sorted(filtered_df[col].unique()), default=sorted(filtered_df[col].unique()))
            trend_filters[col] = options

    trend_df = filtered_df.copy()
    for col, selected in trend_filters.items():
        if selected:
            trend_df = trend_df[trend_df[col].isin(selected)]

    trend_df['Period'] = trend_df['Date'].dt.to_period(time_agg)
    sales_trend = trend_df.groupby('Period')['Weekly_Sales'].sum().reset_index()
    sales_trend['Period'] = sales_trend['Period'].dt.to_timestamp()

    fig_trend = px.line(
        sales_trend,
        x='Period',
        y='Weekly_Sales',
        title=f"Sales Trend ({time_agg} aggregation)",
        labels={'Period': 'Period', 'Weekly_Sales': 'Sales ($)'},
        markers=True,
        color_discrete_sequence=['#6a89cc']
    )
    fig_trend.update_layout(
        hovermode="x unified",
        plot_bgcolor="#f7f9fb",
        paper_bgcolor="#f7f9fb",
        font=dict(family="Inter, Segoe UI, Arial", color="#22223b"),
        title_font=dict(size=22, color="#22223b", family="Inter, Segoe UI, Arial"),
        xaxis=dict(
            color="#22223b",
            title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"),
            tickfont=dict(size=14, color="#22223b"),
            automargin=True
        ),
        yaxis=dict(
            color="#22223b",
            title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"),
            tickfont=dict(size=14, color="#22223b"),
            automargin=True
        ),
        legend=dict(
            font=dict(size=14, color="#22223b"),
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.2
        )
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Sales by Product Category Treemap
    if 'Product_Category' in filtered_df.columns:
        st.header("Sales by Product Category")
        prod_cat_df = filtered_df.groupby('Product_Category')['Weekly_Sales'].sum().reset_index()
        prod_cat_df = prod_cat_df.sort_values('Weekly_Sales', ascending=False)
        fig_prod_cat = px.treemap(
            prod_cat_df,
            path=['Product_Category'],
            values='Weekly_Sales',
            title="Sales by Product Category",
            hover_data={'Weekly_Sales': ':.2f'},
            color='Weekly_Sales',
            color_continuous_scale=['#b8a9c9', '#6a89cc', '#4f5d75']
        )
        fig_prod_cat.update_layout(
            plot_bgcolor="#f7f9fb",
            paper_bgcolor="#f7f9fb",
            font=dict(family="Inter, Segoe UI, Arial", color="#22223b"),
            title_font=dict(size=22, color="#22223b", family="Inter, Segoe UI, Arial"),
            legend=dict(
                font=dict(size=14, color="#22223b"),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.2
            )
        )
        st.plotly_chart(fig_prod_cat, use_container_width=True)

    # Top Regions by Sales Bar Chart
    st.header("Top Regions by Sales")
    if 'Store' in filtered_df.columns:
        top_regions = filtered_df.groupby('Store')['Weekly_Sales'].sum().nlargest(10).reset_index()
        fig_top_regions = px.bar(
            top_regions,
            x='Weekly_Sales',
            y='Store',
            orientation='h',
            title="Top 10 Stores by Sales",
            labels={'Store': 'Store', 'Weekly_Sales': 'Sales ($)'},
            text='Weekly_Sales',
            color='Weekly_Sales',
            color_continuous_scale=['#6a89cc', '#b8a9c9', '#4f5d75']
        )
        fig_top_regions.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_top_regions.update_layout(
            yaxis={'categoryorder':'total ascending', 'title_font':dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), 'tickfont':dict(size=14, color="#22223b")},
            xaxis={'title_font':dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), 'tickfont':dict(size=14, color="#22223b")},
            margin=dict(l=100),
            plot_bgcolor="#f7f9fb",
            paper_bgcolor="#f7f9fb",
            font=dict(family="Inter, Segoe UI, Arial", color="#22223b"),
            title_font=dict(size=22, color="#22223b", family="Inter, Segoe UI, Arial"),
            legend=dict(
                font=dict(size=14, color="#22223b"),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.2
            )
        )
        st.plotly_chart(fig_top_regions, use_container_width=True)

    # Order Value Distribution
    st.header("Order Value Distribution")
    col1, col2 = st.columns(2)
    with col1:
        if 'Weekly_Sales' in filtered_df.columns:
            fig_hist = px.histogram(
                filtered_df,
                x='Weekly_Sales',
                nbins=30,
                title="Weekly Sales Histogram",
                labels={'Weekly_Sales': 'Weekly Sales ($)'},
                color_discrete_sequence=['#6a89cc']
            )
            fig_hist.update_layout(
                plot_bgcolor="#f7f9fb",
                paper_bgcolor="#f7f9fb",
                font=dict(family="Inter, Segoe UI, Arial", color="#22223b"),
                title_font=dict(size=22, color="#22223b", family="Inter, Segoe UI, Arial"),
                xaxis=dict(title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), tickfont=dict(size=14, color="#22223b")),
                yaxis=dict(title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), tickfont=dict(size=14, color="#22223b")),
                legend=dict(
                    font=dict(size=14, color="#22223b"),
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    y=-0.2
                )
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        if 'Weekly_Sales' in filtered_df.columns:
            fig_box = px.box(
                filtered_df,
                y='Weekly_Sales',
                title="Weekly Sales Box Plot",
                labels={'Weekly_Sales': 'Weekly Sales ($)'},
                color_discrete_sequence=['#b8a9c9']
            )
            fig_box.update_layout(
                plot_bgcolor="#f7f9fb",
                paper_bgcolor="#f7f9fb",
                font=dict(family="Inter, Segoe UI, Arial", color="#22223b"),
                title_font=dict(size=22, color="#22223b", family="Inter, Segoe UI, Arial"),
                xaxis=dict(title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), tickfont=dict(size=14, color="#22223b")),
                yaxis=dict(title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), tickfont=dict(size=14, color="#22223b")),
                legend=dict(
                    font=dict(size=14, color="#22223b"),
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    y=-0.2
                )
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # Correlation Heatmap
    st.header("Correlation Heatmap")
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        corr = filtered_df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale=['#b8a9c9', '#6a89cc', '#4f5d75'],
            title='Correlation Matrix',
            aspect="auto"
        )
        fig_corr.update_layout(
            plot_bgcolor="#f7f9fb",
            paper_bgcolor="#f7f9fb",
            font=dict(family="Inter, Segoe UI, Arial", color="#22223b"),
            title_font=dict(size=22, color="#22223b", family="Inter, Segoe UI, Arial"),
            xaxis=dict(title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), tickfont=dict(size=14, color="#22223b")),
            yaxis=dict(title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), tickfont=dict(size=14, color="#22223b")),
            legend=dict(
                font=dict(size=14, color="#22223b"),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.2
            )
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Cumulative Sales Over Time (stacked by Region or Category if available)
    st.header("Cumulative Sales Over Time")
    cumsum_df = trend_df.copy()
    cumsum_df['Period'] = cumsum_df['Date'].dt.to_period(time_agg).dt.to_timestamp()

    # Choose stacking dimension
    stack_by = None
    if 'Store' in cumsum_df.columns:
        stack_by = st.selectbox("Stack cumulative sales by:", options=["None", "Store", "Product_Category"], index=0)
    else:
        stack_by = "None"

    if stack_by and stack_by != "None" and stack_by in cumsum_df.columns:
        cumsum_grouped = cumsum_df.groupby(['Period', stack_by])['Weekly_Sales'].sum().reset_index()
        fig_cum = px.area(
            cumsum_grouped,
            x='Period',
            y='Weekly_Sales',
            color=stack_by,
            title=f"Cumulative Sales Over Time (stacked by {stack_by})",
            color_discrete_sequence=['#6a89cc', '#b8a9c9', '#4f5d75', '#a3cef1', '#b8c0ff']
        )
        fig_cum.update_layout(
            hovermode="x unified",
            plot_bgcolor="#f7f9fb",
            paper_bgcolor="#f7f9fb",
            font=dict(family="Inter, Segoe UI, Arial", color="#22223b"),
            title_font=dict(size=22, color="#22223b", family="Inter, Segoe UI, Arial"),
            xaxis=dict(title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), tickfont=dict(size=14, color="#22223b")),
            yaxis=dict(title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), tickfont=dict(size=14, color="#22223b")),
            legend=dict(
                font=dict(size=14, color="#22223b"),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.2
            )
        )
    else:
        cumsum_total = cumsum_df.groupby('Period')['Weekly_Sales'].sum().reset_index()
        cumsum_total['Cumulative_Sales'] = cumsum_total['Weekly_Sales'].cumsum()
        fig_cum = px.area(
            cumsum_total,
            x='Period',
            y='Cumulative_Sales',
            title="Cumulative Sales Over Time",
            color_discrete_sequence=['#6a89cc']
        )
        fig_cum.update_layout(
            hovermode="x unified",
            plot_bgcolor="#f7f9fb",
            paper_bgcolor="#f7f9fb",
            font=dict(family="Inter, Segoe UI, Arial", color="#22223b"),
            title_font=dict(size=22, color="#22223b", family="Inter, Segoe UI, Arial"),
            xaxis=dict(title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), tickfont=dict(size=14, color="#22223b")),
            yaxis=dict(title_font=dict(size=16, color="#22223b", family="Inter, Segoe UI, Arial"), tickfont=dict(size=14, color="#22223b")),
            legend=dict(
                font=dict(size=14, color="#22223b"),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.2
            )
        )

    st.plotly_chart(fig_cum, use_container_width=True)

    # Real-Time Monitoring Placeholder
    st.header("Real-Time Sales Monitoring (Placeholder)")
    st.info("This section can be extended to show live updating charts or tables for current day/week sales.")

    # Dataset Download
    st.markdown("---")
    st.header("Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name='filtered_sales_data.csv', mime='text/csv')

    # Sales Forecasting Section
    with st.expander("<span style='color:#22223b;'>Sales Forecasting</span>", unsafe_allow_html=True):
        if model is None:
            st.warning("Sales forecasting model not loaded.")
        else:
            st.markdown("<span style='color:#22223b;'>Configure prediction parameters:</span>", unsafe_allow_html=True)
            input_data = {}
            # Interactive inputs based on dataframe columns (excluding target and date)
            for col in filtered_df.columns:
                if col == 'Weekly_Sales':
                    continue
                if filtered_df[col].dtype == 'object':
                    options = sorted(filtered_df[col].unique())
                    input_data[col] = st.selectbox(f"{col}", options)
                elif np.issubdtype(filtered_df[col].dtype, np.number):
                    min_val = float(filtered_df[col].min())
                    max_val = float(filtered_df[col].max())
                    mean_val = float(filtered_df[col].mean())
                    input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

            # Ensure required columns for prediction
            for required_col in ['Date', 'Sales_Class']:
                if required_col not in input_data:
                    if required_col in filtered_df.columns:
                        if filtered_df[required_col].dtype == 'object':
                            options = sorted(filtered_df[required_col].unique())
                            input_data[required_col] = st.selectbox(f"{required_col}", options)
                        elif np.issubdtype(filtered_df[required_col].dtype, np.number):
                            min_val = float(filtered_df[required_col].min())
                            max_val = float(filtered_df[required_col].max())
                            mean_val = float(filtered_df[required_col].mean())
                            input_data[required_col] = st.number_input(f"{required_col}", min_value=min_val, max_value=max_val, value=mean_val)
                        else:
                            input_data[required_col] = filtered_df[required_col].iloc[0]
                    else:
                        input_data[required_col] = None

            if st.button("Generate Sales Forecast"):
                input_df = pd.DataFrame([input_data])
                input_df = encode_categorical(input_df, df)
                # Predict using the loaded model
                try:
                    forecast = model.predict(input_df)[0]
                    st.success(f"<span style='color:#22223b;'>Predicted Weekly Sales: ${forecast:,.2f}</span>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"<span style='color:#e74c3c;'>Error during forecasting: {e}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
