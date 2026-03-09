import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib # type: ignore

# =========================
# LOAD MODEL
# =========================

model = joblib.load("diamond_modelR.pkl")

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Diamond Price Estimator",
    page_icon="💎",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("💎 Diamond Price Estimator")
st.caption("Masukkan spesifikasi diamond, lalu klik Predict untuk melihat estimasi harga.")

st.divider()

# =========================
# INPUT FORM
# =========================

st.subheader("Spesifikasi Diamond")

col1, col2, col3 = st.columns(3)

with col1:
    carat = st.number_input("Carat", min_value=0.2, max_value=5.0, value=0.9, step=0.05)
    x     = st.number_input("Length x (mm)", min_value=3.0, max_value=11.0, value=6.0, step=0.1)
    depth = st.number_input("Depth %", min_value=43.0, max_value=79.0, value=61.5, step=0.1)

with col2:
    cut   = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    y     = st.number_input("Width y (mm)", min_value=3.0, max_value=11.0, value=6.0, step=0.1)
    table = st.number_input("Table %", min_value=43.0, max_value=95.0, value=57.0, step=0.5)

with col3:
    color   = st.selectbox("Color (D=terbaik)", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity (IF=terbaik)", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
    z       = st.number_input("Height z (mm)", min_value=2.0, max_value=7.0, value=3.8, step=0.1)

# =========================
# FEATURE ENGINEERING
# =========================

volume  = x * y * z
density = carat / volume if volume > 0 else 0.0

input_df = pd.DataFrame({
    "carat":   [carat],
    "depth":   [depth],
    "table":   [table],
    "x":       [x],
    "y":       [y],
    "z":       [z],
    "cut":     [cut],
    "color":   [color],
    "clarity": [clarity],
    "volume":  [round(volume, 4)],
    "density": [round(density, 6)]
})

st.divider()

# =========================
# PREDICT BUTTON
# =========================

if st.button("🔍 Predict Price", use_container_width=True, type="primary"):

    prediction = model.predict(input_df)[0]
    prediction = max(0, prediction)

    st.success(f"### Estimasi Harga: **${prediction:,.0f}**")

    # Price tier
    if prediction < 1000:
        tier = "Entry (< $1,000)"
    elif prediction < 5000:
        tier = "Mid-range ($1,000 – $5,000)"
    elif prediction < 12000:
        tier = "Premium ($5,000 – $12,000)"
    else:
        tier = "Luxury (> $12,000)"

    st.info(f"Kategori Harga: **{tier}**")

    with st.expander("Lihat detail input"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# =========================
# FOOTER
# =========================

st.divider()
st.caption("Model: RandomForest Regressor · R² Score: 97.48% · Split 80:20 · Dataset: 53,775 diamonds")