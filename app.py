import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fuzzy_model import create_fuzzy_system
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.markdown("## 🎓 Student Performance Prediction Dashboard")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🎛️ Controls")

study = st.sidebar.slider("Study Time", 0, 4, 2)
absences = st.sidebar.slider("Absences", 0, 30, 5)
g1 = st.sidebar.slider("G1 Marks", 0, 20, 10)
g2 = st.sidebar.slider("G2 Marks", 0, 20, 10)

predict_btn = st.sidebar.button("🚀 Predict")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("student-por.csv")
df.columns = df.columns.str.strip()

for col in ['G1','G2','G3','absences','studytime']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# -------------------------
# LAYOUT
# -------------------------
col1, col2 = st.columns(2)

# -------------------------
# VISUALS
# -------------------------
with col1:
    st.subheader("📊 Grade Distribution")
    fig = plt.figure()
    plt.hist(df['G3'])
    plt.title("Final Grades")
    st.pyplot(fig)

with col2:
    st.subheader("📈 G1 vs G3")
    fig = plt.figure()
    plt.scatter(df['G1'], df['G3'])
    plt.xlabel("G1")
    plt.ylabel("G3")
    st.pyplot(fig)

# -------------------------
# HEATMAP
# -------------------------
st.subheader("🔥 Correlation Heatmap")

corr = df[['G1','G2','G3','studytime','absences']].corr()

fig = plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
st.pyplot(fig)

# -------------------------
# FUZZY PREDICTION
# -------------------------
fuzzy_preds = []

for i in range(len(df)):
    sim = create_fuzzy_system()

    try:
        sim.input['study'] = df.iloc[i]['studytime'] / 4
        sim.input['absences'] = df.iloc[i]['absences'] / 30
        sim.input['g1'] = df.iloc[i]['G1'] / 20
        sim.input['g2'] = df.iloc[i]['G2'] / 20

        sim.compute()

        if 'g3' in sim.output:
            fuzzy_preds.append(sim.output['g3'] * 20)
        else:
            fuzzy_preds.append(0)

    except:
        fuzzy_preds.append(0)

fuzzy_preds = np.array(fuzzy_preds)
actual = df['G3'].values

# -------------------------
# ML MODEL
# -------------------------
X = df[['studytime','absences','G1','G2']]
y = df['G3']

model = LinearRegression()
model.fit(X, y)
ml_preds = model.predict(X)

# -------------------------
# METRICS (CARDS STYLE)
# -------------------------
st.markdown("## 📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

fuzzy_mae = mean_absolute_error(actual, fuzzy_preds)
ml_mae = mean_absolute_error(actual, ml_preds)
fuzzy_rmse = np.sqrt(mean_squared_error(actual, fuzzy_preds))
ml_rmse = np.sqrt(mean_squared_error(actual, ml_preds))

col1.metric("Fuzzy MAE", f"{fuzzy_mae:.2f}")
col2.metric("ML MAE", f"{ml_mae:.2f}")
col3.metric("Fuzzy RMSE", f"{fuzzy_rmse:.2f}")
col4.metric("ML RMSE", f"{ml_rmse:.2f}")

# -------------------------
# ACCURACY
# -------------------------
actual_class = (actual >= 10).astype(int)
fuzzy_class = (fuzzy_preds >= 10).astype(int)
ml_class = (ml_preds >= 10).astype(int)

col1, col2 = st.columns(2)
col1.metric("Fuzzy Accuracy", f"{accuracy_score(actual_class, fuzzy_class)*100:.2f}%")
col2.metric("ML Accuracy", f"{accuracy_score(actual_class, ml_class)*100:.2f}%")

# -------------------------
# CONFUSION MATRIX
# -------------------------
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(actual_class, fuzzy_class)

fig = plt.figure()
plt.imshow(cm)
plt.colorbar()
plt.xticks([0,1], ["Fail","Pass"])
plt.yticks([0,1], ["Fail","Pass"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# -------------------------
# PREDICTION RESULT
# -------------------------
if predict_btn:
    sim = create_fuzzy_system()

    sim.input['study'] = study / 4
    sim.input['absences'] = absences / 30
    sim.input['g1'] = g1 / 20
    sim.input['g2'] = g2 / 20

    sim.compute()
    result = sim.output['g3'] * 20

    st.markdown("## 🎯 Prediction Result")

    if result >= 10:
        st.success(f"✅ PASS | Predicted Score: {result:.2f}")
    else:
        st.error(f"❌ FAIL | Predicted Score: {result:.2f}")

# -------------------------
# 🎯 INTERACTIVE PREDICTION PANEL
# -------------------------
st.markdown("---")
st.markdown("## 🎯 Student Performance Predictor")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Enter Student Details")

    study_input = st.slider("Study Time (0–4)", 0, 4, 2)
    absences_input = st.slider("Absences (0–30)", 0, 30, 5)
    g1_input = st.slider("G1 Marks (0–20)", 0, 20, 10)
    g2_input = st.slider("G2 Marks (0–20)", 0, 20, 10)

    predict = st.button("🚀 Predict Now")

with col2:
    st.markdown("### 📊 Prediction Output")

    if predict:
        sim = create_fuzzy_system()

        sim.input['study'] = study_input / 4
        sim.input['absences'] = absences_input / 30
        sim.input['g1'] = g1_input / 20
        sim.input['g2'] = g2_input / 20

        sim.compute()
        result = sim.output['g3'] * 20

        # 🔥 BIG RESULT DISPLAY
        st.markdown(f"## 🎯 Score: {result:.2f}")

        if result >= 10:
            st.success("✅ STATUS: PASS")
        else:
            st.error("❌ STATUS: FAIL")

        # 🔥 EXTRA INSIGHT
        st.markdown("### 📌 Insight")
        if g2_input > 15:
            st.write("High G2 → Strong final performance")
        elif absences_input > 20:
            st.write("High absences → Risk of failure")
        else:
            st.write("Balanced performance factors")
# -------------------------
# DOWNLOAD
# -------------------------
result_df = pd.DataFrame({
    "Actual": actual,
    "Fuzzy": fuzzy_preds,
    "ML": ml_preds
})

st.download_button("📥 Download Results", result_df.to_csv(index=False), "results.csv")
