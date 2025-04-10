

import streamlit as st

st.set_page_config(page_title="AI Business Intelligence Suite", layout="wide")

st.title("🧠 AI-Powered Business Performance Intelligence Suite")
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Go to", [
    "KPI Overview",
    "Revenue Forecasting",
    "Customer Churn Prediction",
    "Anomaly Detection",
    "Customer Segmentation",
    "AI Recommendations"
])



import pandas as pd
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Saurav Kumar\python files\UPWORK PROJECTS\Buisiness inteligense suite\Data\Superstore.csv",encoding='ISO-8859-1', parse_dates=['Order Date'])
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df.rename(columns={'order_date': 'order_date'}, inplace=True)
    return df

df = load_data()
df['year_month'] = df['order_date'].dt.to_period('M').astype(str)

# PAGE 1: KPI Overview
if page == "KPI Overview":
    st.header("📊 Business KPIs Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"${df['sales'].sum():,.0f}")
    col2.metric("📈 Total Profit", f"${df['profit'].sum():,.0f}")
    col3.metric("📦 Total Orders", df['order_id'].nunique())
    col4.metric("👥 Unique Customers", df['customer_id'].nunique())

    st.markdown("### 📅 Sales Over Time")
    sales_over_time = df.groupby('year_month')['sales'].sum().reset_index()
    fig1 = px.line(sales_over_time, x='year_month', y='sales', title='Monthly Sales Trend', markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 📂 Profit by Category")
    profit_by_cat = df.groupby(['category', 'sub-category'])['profit'].sum().reset_index()
    fig2 = px.bar(profit_by_cat, x='sub-category', y='profit', color='category',
                  title='Profit by Category & Sub-Category',
                  labels={'profit': 'Total Profit'}, height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🌍 Regional Sales Distribution")
    region_sales = df.groupby('region')['sales'].sum().reset_index()
    fig3 = px.pie(region_sales, names='region', values='sales', title='Sales by Region')
    st.plotly_chart(fig3, use_container_width=True)
from sklearn.linear_model import LinearRegression
import numpy as np

if page == "Revenue Forecasting":
    st.header("📈 Revenue Forecasting")

    # Prepare data
    monthly = df.groupby('year_month')['sales'].sum().reset_index()
    monthly['year_month'] = pd.to_datetime(monthly['year_month'])
    monthly['month_num'] = np.arange(len(monthly))

    # Train simple model
    X = monthly[['month_num']]
    y = monthly['sales']
    model = LinearRegression().fit(X, y)

    # Forecast future months
    n_future = st.slider("🔮 Months to Forecast", 3, 12, 6)
    future_months = pd.date_range(start=monthly['year_month'].max() + pd.offsets.MonthBegin(),
                                  periods=n_future, freq='MS')
    future_nums = np.arange(len(monthly), len(monthly) + n_future).reshape(-1, 1)
    future_preds = model.predict(future_nums)

    forecast_df = pd.DataFrame({
        'year_month': future_months,
        'sales': future_preds
    })

    # Combine for plotting
    full_df = pd.concat([monthly[['year_month', 'sales']], forecast_df])

    fig_forecast = px.line(full_df, x='year_month', y='sales', title="📈 Forecasted Revenue",
                           labels={'year_month': 'Month', 'sales': 'Revenue'},
                           markers=True)
    fig_forecast.add_scatter(x=forecast_df['year_month'], y=forecast_df['sales'],
                             mode='lines+markers', name='Forecast', line=dict(dash='dot'))

    st.plotly_chart(fig_forecast, use_container_width=True)

if page == "Customer Churn Prediction":
    st.header("🚨 Customer Churn Prediction")

    st.markdown("""
    This model predicts the **probability of churn** for each customer using behavioral, sales, and frequency features.
    """)

    # Load preprocessed churn data (from your earlier notebook step)
    churn_data = pd.read_csv(r"C:\Users\Saurav Kumar\python files\UPWORK PROJECTS\churn_predictions.csv",encoding='ISO-8859-1')  # Save this from notebook
    churn_data['churn_risk'] = churn_data['churn_probability'].apply(lambda x: 'High' if x > 0.5 else 'Low')
    
    # Churn probability histogram
    fig = px.histogram(churn_data, x='churn_probability', nbins=20,
                       title="🔁 Churn Probability Distribution",
                       labels={'churn_probability': 'Predicted Churn Probability'})
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot: Risk vs Customer CLTV
    st.markdown("### 🎯 Churn Risk vs Customer Lifetime Value (CLTV)")
    fig2 = px.scatter(
        churn_data,
        x='cltv',
        y='churn_probability',
        color='churn_risk',
        hover_data=['customer_id'],
        title="Customer Risk Positioning (CLTV vs Churn Probability)"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Table of High-Risk customers
    st.markdown("### 🔎 High-Risk Customers (Churn Probability > 0.5)")
    st.dataframe(churn_data[churn_data['churn_probability'] > 0.5][
    ['customer_id', 'cltv', 'avg_order_value', 'days_since_last_order', 'churn_probability']
])
if page == "Anomaly Detection":
    st.header("🚨 Anomaly Detection in Business Metrics")

    st.markdown("""
    In this section, we detect anomalies in monthly **Revenue** and **Profit** using the **IQR (Interquartile Range)** method.
    These anomalies can highlight significant drops or spikes in performance.
    """)

    # Monthly Revenue Calculation
    monthly_revenue = df.groupby('year_month')['sales'].sum().reset_index()
    monthly_revenue['year_month'] = pd.to_datetime(monthly_revenue['year_month'])

    # IQR for Revenue
    Q1_revenue = monthly_revenue['sales'].quantile(0.25)
    Q3_revenue = monthly_revenue['sales'].quantile(0.75)
    IQR_revenue = Q3_revenue - Q1_revenue
    lower_bound_revenue = Q1_revenue - 1.5 * IQR_revenue
    upper_bound_revenue = Q3_revenue + 1.5 * IQR_revenue
    monthly_revenue['anomaly'] = monthly_revenue['sales'].apply(
        lambda x: 'Anomaly' if x < lower_bound_revenue or x > upper_bound_revenue else 'Normal')

    # Monthly Profit Calculation
    monthly_profit = df.groupby('year_month')['profit'].sum().reset_index()
    monthly_profit['year_month'] = pd.to_datetime(monthly_profit['year_month'])

    # IQR for Profit
    Q1_profit = monthly_profit['profit'].quantile(0.25)
    Q3_profit = monthly_profit['profit'].quantile(0.75)
    IQR_profit = Q3_profit - Q1_profit
    lower_bound_profit = Q1_profit - 1.5 * IQR_profit
    upper_bound_profit = Q3_profit + 1.5 * IQR_profit
    monthly_profit['anomaly'] = monthly_profit['profit'].apply(
        lambda x: 'Anomaly' if x < lower_bound_profit or x > upper_bound_profit else 'Normal')

    # Revenue Anomalies Visualization
    st.markdown("### 📉 Revenue Anomalies")
    fig_revenue = px.line(monthly_revenue, x='year_month', y='sales', color='anomaly', title="Monthly Revenue Anomalies")
    st.plotly_chart(fig_revenue, use_container_width=True)

    # Profit Anomalies Visualization
    st.markdown("### 📈 Profit Anomalies")
    fig_profit = px.line(monthly_profit, x='year_month', y='profit', color='anomaly', title="Monthly Profit Anomalies")
    st.plotly_chart(fig_profit, use_container_width=True)
if page == "Customer Segmentation":
    st.header("👥 Customer Segmentation with K-Means Clustering")

    st.markdown("""
    We clustered customers using **K-Means** on behavior-based features like:
    - Average Order Value
    - Order Frequency
    - Days Since Last Order
    - Total Orders & Total Sales
    This helps identify valuable segments for targeting and retention.
    """)

    # Load customer segments from file
    segments_df = pd.read_csv(r"C:\Users\Saurav Kumar\python files\UPWORK PROJECTS\customer_segmentation.csv",encoding='ISO-8859-1')
    segments_df['order_frequency'] = segments_df['order_count'] / (segments_df['days_since_last_order'] + 1)

    
    
    # Pie Chart of Segment Distribution
    st.markdown("### 🔹 Segment Distribution")
    seg_dist = segments_df['segment'].value_counts().reset_index()
    seg_dist.columns = ['Segment', 'Count']
    fig1 = px.pie(seg_dist, names='Segment', values='Count', title="Customer Segments (K-Means Clustering)",color_discrete_sequence=px.colors.sequential.Plasma)
    st.plotly_chart(fig1, use_container_width=True)

    # 2D Scatterplot - Days Since Last Order vs Order Frequency
    st.markdown("### 🔹 Customer Behavior Mapping")
    fig2 = px.scatter(
        segments_df,
        x='days_since_last_order',
        y='order_frequency',
        color='segment',
        hover_data=['customer_id', 'avg_order_value', 'total_sales'],
        title="Customer Segments by Recency & Frequency"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Filter Table
    st.markdown("### 🔍 Explore Customers by Segment")
    selected_segment = st.selectbox("Select Segment", segments_df['segment'].unique())
    filtered = segments_df[segments_df['segment'] == selected_segment]
    st.dataframe(filtered)
if page == "AI Recommendations":
    st.header("💡 AI-Powered Business Recommendations")

    st.markdown("""
    This section provides **data-driven business strategies** derived from:
    - Churn Prediction & Risk Assessment
    - Customer Segmentation via K-Means
    - Sales & Profit KPIs
    - Forecasting & Anomaly Detection
    """)

    # Load necessary datasets
    segments_df = pd.read_csv(r"C:\Users\Saurav Kumar\python files\UPWORK PROJECTS\customer_segmentation.csv",encoding='ISO-8859-1')
    segments_df['order_frequency'] = segments_df['order_count'] / (segments_df['days_since_last_order'] + 1)
    churn_data = pd.read_csv(r"C:\Users\Saurav Kumar\python files\UPWORK PROJECTS\churn_predictions.csv",encoding='ISO-8859-1')  # Save this from notebook
    churn_data['churn_risk'] = churn_data['churn_probability'].apply(lambda x: 'High' if x > 0.5 else 'Low')
    df=pd.read_csv(r"C:\Users\Saurav Kumar\python files\UPWORK PROJECTS\cleaned_superstore_data.csv",encoding='ISO-8859-1')

    # -------------------------------
    # 🔹 Section 1: Customer Retention Strategy
    st.markdown("### 🛡️ Customer Retention Strategy")
    
    st.write(segments_df['segment_strategy'].value_counts())

    high_risk = churn_data[churn_data['churn_risk'] == 'High']
    st.info(f"⚠️ {len(high_risk)} customers identified as **High Churn Risk**.")
    st.write("**Recommendation:** Offer personalized loyalty programs, special discounts, or proactive customer service to retain them.")

    # -------------------------------
    # 🔹 Section 2: Marketing Optimization
    st.markdown("### 📣 Marketing Optimization")
    growth_segment = segments_df[segments_df['segment'] == 'Medium Value']
    st.success(f"📊 {len(growth_segment)} customers belong to the **Growth Opportunity** segment.")
    st.write("**Recommendation:** Focus marketing campaigns on this segment using product bundles, upsells, and seasonal offers.")

    # -------------------------------
    # 🔹 Section 3: Sales Growth Suggestions
    st.markdown("### 📈 Sales Growth Suggestions")
    region_sales = df.groupby('region')['sales'].sum().sort_values(ascending=False)
    top_region = region_sales.idxmax()
    st.write(f"**Top Performing Region:** `{top_region}` with ${region_sales[top_region]:,.2f} in sales.")
    st.write("**Recommendation:** Increase product depth, local promotions, and regional influencer tie-ups in this area.")

    # -------------------------------
    # 🔹 Section 4: Cost Optimization Tips
    st.markdown("### ⚙️ Operational Cost Optimization")
    negative_profit = df[df['profit'] < 0]
    st.warning(f"🚨 {len(negative_profit)} transactions recorded **negative profit**.")
    st.write("**Recommendation:** Review these SKUs, evaluate pricing or vendor issues, and consider phasing out low-margin products.")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1522202176988-66273c2fd55f"); /* Stylish ecommerce image */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.6);  /* Adds semi-transparent overlay for readability */
        padding: 2rem;
        border-radius: 12px;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
