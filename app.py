import streamlit as st
import numpy as np
import joblib


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


# KLASIFIKASI

with tab1:
    st.header("Klasifikasi Kategori Produk")
    st.write("Metode: **Random Forest Classifier (Ensemble Method)**")

    # INPUT
    rprice = st.number_input("Harga Produk (RPrice)", min_value=0.0)
    rqty = st.number_input("Jumlah Produk (RQty)", min_value=1)
    linetotal = st.number_input("Line Total", min_value=0.0)
    type_input = st.selectbox("Tipe Pembayaran", ["Cash", "Credit"])

    type_encoded = le_type.transform([type_input])[0]
    X_input = np.array([[rprice, rqty, linetotal, type_encoded]])

    # PREDIKSI
    if st.button("Prediksi Kategori Produk"):
        pred = rf_model.predict(X_input)
        category = le_category.inverse_transform(pred)
        st.info(f"Kategori Produk: **{category[0]}**")

    # CONFUSION MATRIX (PNG)
    st.subheader("Confusion Matrix Model")
    st.image(
        "confusion_matrix.png",
        caption="Confusion Matrix Random Forest Classifier",
        use_container_width=True
    )

    # PENJELASAN
    st.write("""
    Confusion Matrix menunjukkan perbandingan antara kategori aktual dan hasil prediksi.
    Nilai diagonal menandakan prediksi yang benar, sedangkan nilai di luar diagonal
    menunjukkan kesalahan prediksi. Berdasarkan confusion matrix yang ditampilkan, 
    terlihat bahwa kategori Food dan Carbonated memiliki jumlah prediksi benar yang paling tinggi, 
    ditunjukkan oleh nilai besar pada bagian diagonal. Hal ini menunjukkan bahwa model klasifikasi 
    mampu mengenali kategori produk yang paling sering muncul dalam data dengan baik.
    Namun, masih terdapat beberapa kesalahan prediksi, terutama pada kategori dengan jumlah data 
    lebih sedikit seperti Water, yang sering diprediksi sebagai kategori lain. Hal ini menunjukkan 
    bahwa ketidakseimbangan jumlah data antar kategori memengaruhi performa model.
    """)

#  REGRESI

with tab2:
    st.header("Regresi Total Transaksi")
    st.write("Metode: **Gradient Boosting Regressor (Ensemble Method)**")

    # INPUT
    rprice_r = st.number_input("Harga Produk (RPrice)", min_value=0.0, key="rprice_r")
    rqty_r = st.number_input("Jumlah Produk (RQty)", min_value=1, key="rqty_r")
    linetotal_r = st.number_input("Line Total", min_value=0.0, key="linetotal_r")

    X_reg = np.array([[rprice_r, rqty_r, linetotal_r]])
    X_reg_scaled = scaler.transform(X_reg)

    # PREDIKSI
    if st.button("Prediksi Total Transaksi"):
        pred_total = gbr_model.predict(X_reg_scaled)
        st.info(f"Estimasi Total Transaksi: **${pred_total[0]:.2f}**")

    # SCATTER PLOT (PNG)
    st.subheader("Scatter Plot Actual vs Predicted")
    st.image(
        "scatter_plot.png",
        caption="Actual vs Predicted TransTotal",
        use_container_width=True
    )

    # PENJELASAN
    st.write("""
    Pada scatter plot regresi, sebagian besar titik berada di sekitar garis diagonal, 
    yang menandakan bahwa nilai prediksi total transaksi cukup mendekati nilai aktual. 
    Meskipun terdapat beberapa titik yang menyimpang, pola keseluruhan menunjukkan 
    bahwa model regresi sudah mampu menangkap hubungan antara harga, jumlah produk, 
    dan total transaksi dengan cukup baik.
    """)
