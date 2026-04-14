[zomato_clean (1).csv](https://github.com/user-attachments/files/26700917/zomato_clean.1.csv)
[prediksi.py](https://github.com/user-attachments/files/26700934/prediksi.py)import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Customer Rating Prediction", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = pickle.load(open('best_model_xgboost (2).pkl', 'rb'))
    scaler = pickle.load(open('scaler (1).pkl', 'rb'))
    with open('feature_columns (1).pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("zomato_clean (1).csv")

    # rapihin nama kolom (ANTI ERROR)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("(", "")
    df.columns = df.columns.str.replace(")", "")

    return df

df = load_data()

# =========================
# TITLE
# =========================
st.title("📦 Customer Rating Prediction App")

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["📊 EDA & Overview", "🤖 Prediction"])

# =========================================================
# TAB 1 → EDA
# =========================================================
with tab1:
    st.header("📌 Project Overview")

    st.write("""
    Project ini bertujuan untuk menganalisis faktor yang mempengaruhi rating pelanggan 
    pada layanan delivery dan membangun model prediksi rating menggunakan XGBoost.
    """)

    st.subheader("📊 Dataset")
    st.dataframe(df.head())

    # =====================
    # DISTRIBUSI HARI
    # =====================
    st.subheader("📅 Distribusi Order per Hari")

    fig1, ax1 = plt.subplots()
    sns.countplot(x='day_of_week', data=df, order=df['day_of_week'].value_counts().index, ax=ax1)

    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height())}',
                     (p.get_x()+p.get_width()/2, p.get_height()),
                     ha='center')

    st.pyplot(fig1)

    # =====================
    # TIME CATEGORY
    # =====================
    st.subheader("⏰ Distribusi Waktu Order")

    fig2, ax2 = plt.subplots()
    sns.countplot(x='time_category', data=df, order=df['time_category'].value_counts().index, ax=ax2)
    st.pyplot(fig2)

    # =====================
    # RATING VS TIME
    # =====================
    st.subheader("⭐ Rating vs Waktu")

    fig3, ax3 = plt.subplots()
    sns.barplot(data=df, x='time_category', y='Delivery_person_Ratings', ax=ax3)
    st.pyplot(fig3)

    # =====================
    # DELIVERY TIME
    # =====================
    st.subheader("🚀 Delivery Time vs Rating")

    fig4, ax4 = plt.subplots()
    sns.regplot(data=df, x='Time_taken_min', y='Delivery_person_Ratings',
                scatter_kws={'alpha':0.2}, ax=ax4)
    st.pyplot(fig4)

    # =====================
    # TRAFFIC
    # =====================
    st.subheader("🚦 Traffic vs Rating")

    fig5, ax5 = plt.subplots()
    sns.boxplot(data=df, x='Road_traffic_density', y='Delivery_person_Ratings', ax=ax5)
    st.pyplot(fig5)

    # =====================
    # WEATHER
    # =====================
    st.subheader("🌦️ Weather vs Rating")

    fig6, ax6 = plt.subplots()
    sns.boxplot(data=df, x='Weather_conditions', y='Delivery_person_Ratings', ax=ax6)
    plt.xticks(rotation=45)
    st.pyplot(fig6)

    # =====================
    # INSIGHT
    # =====================
    st.subheader("📌 Key Insight")

    st.success("""
    - Semakin lama delivery → rating menurun  
    - Traffic tinggi & cuaca buruk → rating lebih rendah  
    - Waktu order tertentu memiliki volume tinggi (peak hour)
    """)

# =========================================================
# TAB 2 → PREDICTION
# =========================================================
with tab2:
    st.header("🤖 Predict Customer Rating")

    # =========================
    # INPUT USER
    # =========================
    age = st.slider("Delivery Person Age", 18, 60, 30)
    distance = st.slider("Distance (km)", 1, 30, 10)
    time_taken = st.slider("Delivery Time (minutes)", 10, 60, 30)
    order_prepare_time = st.slider("Order Prepare Time (minutes)", 5, 30, 15)

    traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Stormy", "Sandstorms", "Windy"])
    vehicle = st.selectbox("Vehicle Type", ["motorcycle", "scooter"])
    city = st.selectbox("City", ["Urban", "Semi-Urban", "Metropolitan"])
    order_type = st.selectbox("Order Type", ["Snack", "Meal", "Drinks", "Buffet"])
    time_category = st.selectbox("Time Category", ["Morning", "Afternoon", "Evening", "Night"])

    festival = st.selectbox("Festival", ["No", "Yes"])
    multiple_deliveries = st.slider("Multiple Deliveries", 0, 5, 1)
    vehicle_condition = st.slider("Vehicle Condition", 0, 2, 1)

    day_of_week = st.selectbox("Day of Week",
                              ["Monday", "Tuesday", "Wednesday",
                               "Thursday", "Friday", "Saturday", "Sunday"])

    # =========================
    # ENCODING
    # =========================
    festival_val = 1 if festival == "Yes" else 0

    traffic_map = {"Low":1, "Medium":2, "High":3, "Jam":4}
    traffic_val = traffic_map[traffic]

    day_map = {
        "Monday":1, "Tuesday":2, "Wednesday":3,
        "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7
    }
    day_val = day_map[day_of_week]

    # =========================
    # DATAFRAME
    # =========================
    input_data = pd.DataFrame({
        'Delivery_person_Age': [age],
        'distance': [distance],
        'Time_taken (min)': [time_taken],
        'order_prepare_time (min)': [order_prepare_time],
        'Road_traffic_density': [traffic_val],
        'Festival': [festival_val],
        'multiple_deliveries': [multiple_deliveries],
        'Vehicle_condition': [vehicle_condition],
        'day_of_week': [day_val],
        'Weather_conditions': [weather],
        'Type_of_vehicle': [vehicle],
        'City': [city],
        'Type_of_order': [order_type],
        'time_category': [time_category]
    })

    # =========================
    # ONE HOT
    # =========================
    input_data = pd.get_dummies(input_data)

    # =========================
    # SAMAKAN FITUR
    # =========================
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    # =========================
    # SCALING
    # =========================
    scaling_cols = [
        'Delivery_person_Age',
        'distance',
        'Time_taken (min)',
        'order_prepare_time (min)'
    ]

    input_data[scaling_cols] = scaler.transform(input_data[scaling_cols])

    # =========================
    # PREDICT
    # =========================
    if st.button("🚀 Predict Rating"):
        prediction = model.predict(input_data)[0]

        st.success(f"⭐ Predicted Rating: {round(prediction, 2)}")

        if prediction < 4:
            st.warning("⚠️ Risiko rating rendah! Perbaiki waktu pengantaran.")
        else:
            st.info("✅ Performa delivery sudah baik!")

