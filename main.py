import streamlit as st
import numpy as np
import joblib

# Load model & encoder
rf_model = joblib.load('rf_classifier.joblib')
le_category = joblib.load('category_encoder.joblib')
le_type = joblib.load('type_encoder.joblib')

gbr_model = joblib.load('gbr_regressor.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Prediksi Kategori & Total Transaksi Vending Machine")
st.header("Input Data Transaksi")

rprice = st.number_input("Harga Produk (RPrice)", min_value=0.0)
rqty = st.number_input("Jumlah Produk (RQty)", min_value=1)
linetotal = st.number_input("Line Total", min_value=0.0)

type_input = st.selectbox("Tipe Pembayaran", ["Cash", "Credit"])
type_encoded = le_type.transform([type_input])[0]

# --- KLASIFIKASI ---
X_class = np.array([[rprice, rqty, linetotal, type_encoded]])

if st.button("Prediksi Kategori Produk"):
    pred_class = rf_model.predict(X_class)
    category = le_category.inverse_transform(pred_class)
    st.success(f"Kategori Produk: {category[0]}")

# --- REGRESI ---
X_reg = np.array([[rprice, rqty, linetotal]])
X_reg_scaled = scaler.transform(X_reg)

if st.button("Prediksi Total Transaksi"):
    pred_total = gbr_model.predict(X_reg_scaled)
    st.success(f"Estimasi Total Transaksi: ${pred_total[0]:.2f}")
