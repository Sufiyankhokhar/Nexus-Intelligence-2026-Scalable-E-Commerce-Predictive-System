import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Nexus Intelligence 2026",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\featured_ecommerce.csv"
)

# ==========================
# TITLE
# ==========================

st.title("🚀 Nexus Intelligence 2026")
st.markdown("### Scalable E-Commerce Predictive Analytics Platform")

# ==========================
# SIDEBAR FILTERS
# ==========================

st.sidebar.header("Filters")

country = st.sidebar.selectbox(
    "Select Country",
    ["All"] + sorted(list(df["country"].dropna().unique()))
)

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(list(df["category"].dropna().unique()))
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["All"] + sorted(list(df["payment_method"].dropna().unique()))
)

# ==========================
# APPLY FILTERS
# ==========================

filtered_df = df.copy()

if country != "All":
    filtered_df = filtered_df[
        filtered_df["country"] == country
    ]

if category != "All":
    filtered_df = filtered_df[
        filtered_df["category"] == category
    ]

if payment_method != "All":
    filtered_df = filtered_df[
        filtered_df["payment_method"] == payment_method
    ]

# ==========================
# KPI CARDS
# ==========================

st.subheader("📊 Business KPIs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Revenue",
        f"${filtered_df['total_price_usd'].sum():,.2f}"
    )

with col2:
    st.metric(
        "Profit",
        f"${filtered_df['profit_usd'].sum():,.2f}"
    )

with col3:
    st.metric(
        "Orders",
        filtered_df["order_id"].nunique()
    )

with col4:
    st.metric(
        "Customers",
        filtered_df["customer_id"].nunique()
    )

# ==========================
# DATA PREVIEW
# ==========================

st.subheader("📄 Dataset Preview")

st.dataframe(
    filtered_df.head(20)
)

# ==========================
# REVENUE BY COUNTRY
# ==========================

st.subheader("🌍 Revenue by Country")

country_sales = (
    filtered_df
    .groupby("country")["total_price_usd"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10,5))

country_sales.plot(
    kind="bar",
    ax=ax
)

plt.xticks(rotation=45)

st.pyplot(fig)

# ==========================
# CATEGORY REVENUE
# ==========================

st.subheader("🛒 Revenue by Category")

category_sales = (
    filtered_df
    .groupby("category")["total_price_usd"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(category_sales)

# ==========================
# FRAUD ANALYSIS
# ==========================

st.subheader("🚨 Fraud Analysis")

st.metric(
    "Average Fraud Risk Score",
    round(
        filtered_df["fraud_risk_score"].mean(),
        2
    )
)

fig, ax = plt.subplots(figsize=(8,4))

filtered_df["fraud_risk_score"].hist(
    bins=20,
    ax=ax
)

st.pyplot(fig)

# ==========================
# CUSTOMER SEGMENTATION
# ==========================

if "customer_cluster" in filtered_df.columns:

    st.subheader("👥 Customer Segmentation")

    cluster_counts = (
        filtered_df["customer_cluster"]
        .value_counts()
    )

    st.bar_chart(cluster_counts)

# ==========================
# SALES FORECAST SECTION
# ==========================

st.subheader("📈 Sales Forecasting Model")

st.info("""
Model Used: Random Forest Regressor

MAE : 220

R² Score : 0.26
""")

# ==========================
# EXECUTIVE SUMMARY
# ==========================

st.subheader("🏆 Executive Summary")

top_country = (
    filtered_df
    .groupby("country")["total_price_usd"]
    .sum()
    .idxmax()
)

top_category = (
    filtered_df
    .groupby("category")["total_price_usd"]
    .sum()
    .idxmax()
)

st.success(
    f"Top Revenue Country : {top_country}"
)

st.success(
    f"Top Revenue Category : {top_category}"
)

# ==========================
# FOOTER
# ==========================

st.markdown("---")

st.markdown(
    "Built with Python, Pandas, Scikit-Learn, Streamlit and Machine Learning"
)