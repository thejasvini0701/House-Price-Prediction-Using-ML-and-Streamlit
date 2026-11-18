
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PropertyAI | House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# ==========================================
# 2. LOAD DATA AND MODEL
# ==========================================

df = pd.read_csv("data.csv")
model = joblib.load("house_price_model.pkl")

features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "yr_built"
]

clean_df = df.dropna(subset=features + ["price"])
clean_df = clean_df[clean_df["price"] > 0]

# ==========================================
# 3. CUSTOM DARK THEME
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stHeader"] {
    background-color: rgba(11, 17, 32, 0.9);
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p, label, span {
    color: #CBD5E1;
}

div[data-testid="stMetric"] {
    background-color: #172337;
    border: 1px solid #26364D;
    padding: 18px;
    border-radius: 12px;
}

.stButton > button {
    background-color: #0D9488;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 20px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background-color: #0F766E;
    color: white;
    border: 1px solid #5EEAD4;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #26364D;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🏡 PropertyAI")
st.sidebar.caption("House Price Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Data Analysis",
        "Price Prediction",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Powered by Machine Learning\n\nBuilt with Python & Streamlit"
)

# ==========================================
# 5. MAIN HEADER
# ==========================================

st.title("🏡 PropertyAI")
st.caption("Smart Property Analytics | Machine Learning Powered")
st.divider()

# ==========================================
# 6. DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    st.write(
        "Welcome to your House Price Intelligence Dashboard!"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Properties",
            f"{len(clean_df):,}"
        )

    with col2:
        st.metric(
            "Average House Price",
            f"{clean_df['price'].mean():,.2f}"
        )

    with col3:
        st.metric(
            "Average Living Area",
            f"{clean_df['sqft_living'].mean():,.0f} sqft"
        )

    st.divider()

    st.subheader("🏠 Recent Dataset Records")

    st.dataframe(
        clean_df.head(10),
        use_container_width=True
    )

    # House price distribution

    st.divider()
    st.subheader("📈 House Price Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.histplot(
        clean_df["price"],
        bins=40,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("House Price")
    ax.set_ylabel("Number of Properties")
    ax.set_title("Distribution of House Prices")

    fig.patch.set_facecolor("#172337")
    ax.set_facecolor("#172337")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    st.pyplot(fig)
    plt.close(fig)

    # Average price by bedrooms

    st.divider()
    st.subheader("📊 Average Price by Bedrooms")

    bedroom_prices = (
        clean_df.groupby("bedrooms")["price"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.barplot(
        data=bedroom_prices,
        x="bedrooms",
        y="price",
        ax=ax
    )

    ax.set_xlabel("Number of Bedrooms")
    ax.set_ylabel("Average House Price")
    ax.set_title("Average Price by Bedrooms")

    fig.patch.set_facecolor("#172337")
    ax.set_facecolor("#172337")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    st.pyplot(fig)
    plt.close(fig)

    # Average price by floors

    st.divider()
    st.subheader("🏢 Average Price by Floors")

    floor_prices = (
        clean_df.groupby("floors")["price"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.barplot(
        data=floor_prices,
        x="floors",
        y="price",
        ax=ax
    )

    ax.set_xlabel("Number of Floors")
    ax.set_ylabel("Average House Price")
    ax.set_title("Average Price by Floors")

    fig.patch.set_facecolor("#172337")
    ax.set_facecolor("#172337")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    st.pyplot(fig)
    plt.close(fig)

# ==========================================
# 7. DATA ANALYSIS PAGE
# ==========================================

elif page == "Data Analysis":

    st.header("📊 Data Analysis")

    st.write(
        "Explore your dataset and understand "
        "relationships between house features."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Dataset Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Valid Records",
            f"{len(clean_df):,}"
        )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        clean_df.head(10),
        use_container_width=True
    )

    st.divider()

    st.subheader("Feature Correlation Heatmap")

    corr = clean_df[features + ["price"]].corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Feature Correlation", color="white")
    fig.patch.set_facecolor("#172337")
    ax.set_facecolor("#172337")

    st.pyplot(fig)
    plt.close(fig)

# ==========================================
# 8. PRICE PREDICTION PAGE
# ==========================================

elif page == "Price Prediction":

    st.header("🏡 House Price Prediction")

    st.write(
        "Enter the property details below "
        "to estimate its price."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        bedrooms = st.number_input(
            "🛏️ Number of Bedrooms",
            min_value=0,
            max_value=20,
            value=3
        )

        bathrooms = st.number_input(
            "🛁 Number of Bathrooms",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        )

        sqft_living = st.number_input(
            "🏠 Living Area (sqft)",
            min_value=1,
            value=1800
        )

    with col2:

        sqft_lot = st.number_input(
            "📐 Lot Area (sqft)",
            min_value=1,
            value=5000
        )

        floors = st.number_input(
            "🏢 Number of Floors",
            min_value=1.0,
            max_value=10.0,
            value=1.0,
            step=0.5
        )

        yr_built = st.number_input(
            "📅 Year Built",
            min_value=1800,
            max_value=2026,
            value=2005
        )

    st.divider()

    if st.button("✨ Predict House Price"):

        new_house = pd.DataFrame([{
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "yr_built": yr_built
        }])

        predicted_price = model.predict(new_house)[0]

        if predicted_price > 0:

            st.success("Prediction Completed!")

            st.metric(
                "Estimated House Price",
                f"{predicted_price:,.2f}"
            )

            st.subheader("Property Details")

            st.dataframe(
                new_house,
                use_container_width=True
            )

            st.caption(
                "This is an ML-based estimate, "
                "not a guaranteed property valuation."
            )

        else:
            st.warning(
                "The model returned a non-positive estimate. "
                "Try different property details."
            )

# ==========================================
# 9. MODEL PERFORMANCE PAGE
# ==========================================

elif page == "Model Performance":

    st.header("🤖 Model Performance")

    st.write(
        "Evaluation results of our trained "
        "Linear Regression model."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Mean Absolute Error (MAE)",
            "164,083.68"
        )

    with col2:

        st.metric(
            "R² Score",
            "0.569"
        )

    st.divider()

    st.subheader("Understanding the Results")

    st.markdown("""
    **Mean Absolute Error (MAE)**

    The average absolute difference between
    actual house prices and predicted prices.

    **R² Score**

    Indicates how much variation in house prices
    is explained by the model on the test dataset.
    """)

    st.info(
        "These metrics are from our earlier test-set "
        "evaluation. They are baseline results, not "
        "a guarantee of prediction accuracy."
    )

    st.divider()

    st.subheader("Model Evaluation Summary")

    performance_data = pd.DataFrame({
        "Metric": ["MAE", "R² Score"],
        "Value": [164083.68, 0.569]
    })

    st.dataframe(
        performance_data,
        use_container_width=True
    )

# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "PropertyAI | Built using Python, Scikit-learn "
    "and Streamlit"
)
