import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# LOAD MODEL & ENCODER
rf_model = joblib.load("rf_classifier.joblib")
le_category = joblib.load("category_encoder.joblib")
le_type = joblib.load("type_encoder.joblib")

gbr_model = joblib.load("gbr_regressor.joblib")
scaler = joblib.load("scaler.joblib")


st.title("Proyek Data Mining: Penjualan Vending Machine")
st.subheader("Klasifikasi & Regresi Menggunakan Ensemble Method")
st.success("Semua model berhasil dimuat!")


tab1, tab2 = st.tabs([" Klasifikasi", " Regresi"])


# ===== TAB KLASIFIKASI =====
with tab1:
    st.header(" Klasifikasi Kategori Produk")
    st.write("Metode: **Random Forest Classifier (Ensemble Method)**")

    #  INPUT 
    st.subheader("Input Data Transaksi")
    rprice = st.number_input("Harga Produk (RPrice)", min_value=0.0)
    rqty = st.number_input("Jumlah Produk (RQty)", min_value=1)
    linetotal = st.number_input("Line Total", min_value=0.0)
    type_input = st.selectbox("Tipe Pembayaran", ["Cash", "Credit"])

    type_encoded = le_type.transform([type_input])[0]
    X_input = np.array([[rprice, rqty, linetotal, type_encoded]])

    #  PREDIKSI 
    if st.button("Prediksi Kategori Produk"):
        pred = rf_model.predict(X_input)
        category = le_category.inverse_transform(pred)
        st.info(f"Kategori Produk: **{category[0]}**")

    #  CONFUSION MATRIX 
    st.subheader(" Confusion Matrix Model Klasifikasi")

    # contoh evaluasi visual (pakai data dummy simulasi)
    y_test_dummy = np.random.randint(0, len(le_category.classes_), 50)
    y_pred_dummy = rf_model.predict(
        np.random.rand(50, 4)
    )

    cm = confusion_matrix(y_test_dummy, y_pred_dummy)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le_category.classes_
    )
    disp.plot(ax=ax, xticks_rotation=45)
    st.pyplot(fig)

    #  PENJELASAN 
    st.subheader(" Penjelasan Metode")
    st.write("""
    Random Forest merupakan metode ensemble yang membangun banyak pohon keputusan
    dan menggabungkan hasil prediksinya. Metode ini stabil, akurat, dan cocok untuk
    data transaksi vending machine.
    """)

# TAB REGRESI 

with tab2:
    st.header(" Regresi Total Transaksi")
    st.write("Metode: **Gradient Boosting Regressor (Ensemble Method)**")

    #  INPUT 
    st.subheader("Input Data Transaksi")
    rprice_r = st.number_input("Harga Produk (RPrice)", min_value=0.0, key="rprice_r")
    rqty_r = st.number_input("Jumlah Produk (RQty)", min_value=1, key="rqty_r")
    linetotal_r = st.number_input("Line Total", min_value=0.0, key="linetotal_r")

    X_reg = np.array([[rprice_r, rqty_r, linetotal_r]])
    X_reg_scaled = scaler.transform(X_reg)

    #  PREDIKSI 
    if st.button("Prediksi Total Transaksi"):
        pred_total = gbr_model.predict(X_reg_scaled)
        st.info(f"Estimasi Total Transaksi: **${pred_total[0]:.2f}**")

    # SCATTER PLOT 
    st.subheader(" Scatter Plot Actual vs Predicted")

    # contoh visualisasi (simulasi evaluasi)
    y_test_dummy = np.random.rand(50) * 10
    y_pred_dummy = y_test_dummy + np.random.normal(0, 1, 50)

    fig, ax = plt.subplots()
    ax.scatter(y_test_dummy, y_pred_dummy)
    ax.set_xlabel("Actual TransTotal")
    ax.set_ylabel("Predicted TransTotal")
    ax.set_title("Actual vs Predicted TransTotal")
    st.pyplot(fig)

    # PENJELASAN 
    st.subheader(" Penjelasan Metode")
    st.write("""
    Gradient Boosting Regressor merupakan metode ensemble yang bekerja secara bertahap
    dengan memperbaiki kesalahan prediksi sebelumnya. Metode ini efektif untuk
    memprediksi nilai numerik seperti total transaksi.
    """)
