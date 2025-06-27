import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import os
from datetime import datetime

# Page config
st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv("housing_extended.csv")
df.dropna(inplace=True)
df["price_per_sqft"] = (df["price"] * 100000) / df["area"]

# Add dummy lat/lon if not present
if 'latitude' not in df.columns or 'longitude' not in df.columns:
    np.random.seed(42)
    df['latitude'] = np.random.uniform(19.07, 19.15, len(df))
    df['longitude'] = np.random.uniform(72.85, 72.95, len(df))

feature_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Sidebar filters
st.sidebar.header("🔍 Filter Dataset")
bedroom_filter = st.sidebar.multiselect("Bedrooms", sorted(df['bedrooms'].unique()), default=list(sorted(df['bedrooms'].unique())))
story_filter = st.sidebar.multiselect("Floors", sorted(df['stories'].unique()), default=list(sorted(df['stories'].unique())))
parking_filter = st.sidebar.multiselect("Parking Spaces", sorted(df['parking'].unique()), default=list(sorted(df['parking'].unique())))

filtered_df = df[
    (df['bedrooms'].isin(bedroom_filter)) &
    (df['stories'].isin(story_filter)) &
    (df['parking'].isin(parking_filter))
]

if filtered_df.empty:
    st.warning("⚠️ No data matches your filter. Showing full dataset instead.")
    filtered_df = df.copy()

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Visualizations", "🧾 Insights & History"])

# === TAB 1: Prediction ===
with tab1:
    st.header("📌 Enter Property Details")

    area = st.slider("Area (sq ft)", 500, 3000, 1500, step=100)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
    stories = st.selectbox("Floors", [1, 2, 3])
    parking = st.selectbox("Parking Spaces", [0, 1, 2, 3])
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])

    if st.button("💡 Predict Price"):
        input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
        prediction = model.predict(input_data)[0]
        st.success(f"🏷️ Estimated House Price: ₹ {round(prediction, 2)} Lakhs")

        # Save prediction history
        history_row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'predicted_price': round(prediction, 2)
        }
        if os.path.exists("prediction_history.csv"):
            history_df = pd.read_csv("prediction_history.csv")
            history_df = pd.concat([history_df, pd.DataFrame([history_row])], ignore_index=True)
        else:
            history_df = pd.DataFrame([history_row])
        history_df.to_csv("prediction_history.csv", index=False)

        # SHAP plots
        explainer = shap.Explainer(model, df[feature_cols])
        shap_values = explainer(pd.DataFrame(input_data, columns=feature_cols))

        st.subheader("🧠 Feature Impact (Bar)")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

        st.subheader("🔍 SHAP Waterfall Plot")
        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig2)

        st.subheader("🏗️ Estimated Construction Materials")
        materials = {
            'Bricks (units)': int(area * 8),
            'Cement (bags)': round(area * 0.4),
            'Steel (kg)': round(area * 3.5),
            'Sand (cu.ft)': round(area * 1.2),
            'Tiles (sq.ft)': round(area * 1.05),
            'Paint (litres)': round(area * 0.1),
            'Plumbing & Wiring (₹)': int(area * 180)
        }
        mat_df = pd.DataFrame.from_dict(materials, orient='index', columns=['Estimated Quantity / Cost'])
        st.table(mat_df)

# === TAB 2: Visualizations ===
with tab2:
    st.header("📊 Data Visualizations")

    st.subheader("🗺️ Property Map View")
    st.map(filtered_df[['latitude', 'longitude']])

    st.subheader("📈 Area vs Price")
    fig = px.scatter(filtered_df, x="area", y="price", color="bedrooms", size="bathrooms", hover_data=['stories', 'parking'])
    st.plotly_chart(fig)

    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(filtered_df[feature_cols + ['price']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# === TAB 3: Insights & History ===
with tab3:
    st.header("🧾 Market Insights")

    col1, col2 = st.columns(2)
    with col1:
        avg_price = filtered_df.groupby("bedrooms")["price"].mean().reset_index()
        fig = px.bar(avg_price, x="bedrooms", y="price", color="bedrooms", title="Avg Price by Bedrooms")
        st.plotly_chart(fig)

    with col2:
        parking_data = filtered_df["parking"].value_counts().reset_index()
        parking_data.columns = ["parking", "count"]
        fig = px.pie(parking_data, values='count', names='parking', title='Parking Distribution')
        st.plotly_chart(fig)

    st.subheader("📜 Prediction History")
    if os.path.exists("prediction_history.csv"):
        history_df = pd.read_csv("prediction_history.csv")
        st.dataframe(history_df)
        st.download_button("⬇️ Download History", data=history_df.to_csv(index=False), file_name="prediction_history.csv", mime='text/csv')
    else:
        st.info("No prediction history available yet.")



# Footer
st.markdown("---")
st.markdown("<center>© 2025 House Price Predictor | Developed by Pradip Rathod</center>", unsafe_allow_html=True)
# Social Links
st.markdown("---")
st.markdown("### 📬 Connect with Me")
st.markdown("""
<div style='text-align: center;'>
    <a href='https://www.linkedin.com/in/pradipgrathod/' target='_blank'>
        <img src='https://img.icons8.com/color/48/linkedin.png' alt='LinkedIn' style='margin-right: 10px;'/>
    </a>
    <a href='https://github.com/rtdpradip-07' target='_blank'>
        <img src='https://img.icons8.com/nolan/48/github.png' alt='GitHub' style='margin-right: 10px;'/>
    </a>
    <a href='https://www.instagram.com/rtd_pradip_07?igsh=MWV6NG56c3o4ajluZQ==' target='_blank'>
        <img src='https://img.icons8.com/color/48/instagram-new.png' alt='Instagram'/>
    </a>
</div>
""", unsafe_allow_html=True)