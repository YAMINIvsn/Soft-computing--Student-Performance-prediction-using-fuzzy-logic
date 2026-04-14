import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fuzzy_model import create_fuzzy_system
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Student Dashboard", layout="wide")

# -------------------------
# 🌈 MODERN UI STYLE
# -------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.02);
}
.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    background: -webkit-linear-gradient(#00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.result-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎓 Student Performance Dashboard</div>', unsafe_allow_html=True)

# -------------------------
# LOAD DATA (FAST ⚡)
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("student-por.csv")
    df.columns = df.columns.str.strip()

    for col in ['G1','G2','G3','absences','studytime']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna()

df = load_data()
df = df.sample(150)

# -------------------------
# FUZZY + ML (FIXED ORDER 🔥)
# -------------------------
@st.cache_data
def compute_fuzzy(df):
    preds = []
    for i in range(len(df)):
        sim = create_fuzzy_system()
        try:
            sim.input['study'] = df.iloc[i]['studytime'] / 4
            sim.input['absences'] = df.iloc[i]['absences'] / 30
            sim.input['g1'] = df.iloc[i]['G1'] / 20
            sim.input['g2'] = df.iloc[i]['G2'] / 20
            sim.compute()
            preds.append(sim.output['g3'] * 20)
        except:
            preds.append(0)
    return np.array(preds)

fuzzy_preds = compute_fuzzy(df)
actual = df['G3'].values

X = df[['studytime','absences','G1','G2']]
y = df['G3']

model = LinearRegression()
model.fit(X, y)
ml_preds = model.predict(X)

# -------------------------
# BASIC VISUALS
# -------------------------
st.markdown("## 📊 Insights")

col1, col2 = st.columns(2)

with col1:
    fig = plt.figure()
    plt.hist(df['G3'])
    plt.title("Grade Distribution")
    st.pyplot(fig)

with col2:
    fig = plt.figure()
    plt.scatter(df['G1'], df['G3'])
    plt.xlabel("G1")
    plt.ylabel("G3")
    st.pyplot(fig)

# -------------------------
# HEATMAP
# -------------------------
st.markdown("## 🔥 Correlation Heatmap")

corr = df[['G1','G2','G3','studytime','absences']].corr()

fig = plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
st.pyplot(fig)

# -------------------------
# ADVANCED VISUALS 🔥
# -------------------------

# Pairplot style
st.markdown("## 🔍 Multi Feature Analysis")

cols = ['G1','G2','G3','studytime','absences']
fig = plt.figure(figsize=(10,8))

for i in range(len(cols)):
    for j in range(len(cols)):
        plt.subplot(len(cols), len(cols), i*len(cols)+j+1)
        if i == j:
            plt.hist(df[cols[i]])
        else:
            plt.scatter(df[cols[j]], df[cols[i]], s=5)
        plt.xticks([])
        plt.yticks([])

plt.tight_layout()
st.pyplot(fig)

# Feature importance
st.markdown("## 📊 Feature Importance")

fig = plt.figure()
plt.bar(['study','absences','G1','G2'], model.coef_)
st.pyplot(fig)

# Error distribution
st.markdown("## ⚠️ Error Distribution")

fig = plt.figure()
plt.hist(actual - fuzzy_preds)
st.pyplot(fig)

# Predicted vs Actual
st.markdown("## 🎯 Predicted vs Actual")

fig = plt.figure()
plt.scatter(actual, fuzzy_preds)
st.pyplot(fig)

# Model comparison
st.markdown("## 📈 Model Comparison")

fig = plt.figure()
plt.plot(actual[:50], label="Actual")
plt.plot(fuzzy_preds[:50], label="Fuzzy")
plt.plot(ml_preds[:50], label="ML")
plt.legend()
st.pyplot(fig)

# Pie chart
st.markdown("## 🧩 Pass vs Fail")

fig = plt.figure()
plt.pie((df['G3']>=10).value_counts(), labels=["Pass","Fail"], autopct='%1.1f%%')
st.pyplot(fig)

# 3D plot
st.markdown("## 🧠 3D Visualization")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['G1'], df['G2'], df['G3'])
st.pyplot(fig)

# Residuals
st.markdown("## 🧠 Residual Analysis")

fig = plt.figure()
plt.scatter(ml_preds, actual-ml_preds)
plt.axhline(0)
st.pyplot(fig)

# Cumulative error
st.markdown("## 📈 Cumulative Error")

fig = plt.figure()
plt.plot(np.cumsum(np.abs(actual - fuzzy_preds)))
st.pyplot(fig)

# -------------------------
# METRICS
# -------------------------
# -------------------------
# 📊 PERFORMANCE METRICS (ADVANCED)
# -------------------------
st.markdown("## 📊 Model Performance Dashboard")

# ---- Regression Metrics ----
fuzzy_mae = mean_absolute_error(actual, fuzzy_preds)
ml_mae = mean_absolute_error(actual, ml_preds)

fuzzy_rmse = np.sqrt(mean_squared_error(actual, fuzzy_preds))
ml_rmse = np.sqrt(mean_squared_error(actual, ml_preds))

fuzzy_r2 = r2_score(actual, fuzzy_preds)
ml_r2 = r2_score(actual, ml_preds)

# ---- Classification Metrics ----
actual_class = (actual >= 10).astype(int)
fuzzy_class = (fuzzy_preds >= 10).astype(int)
ml_class = (ml_preds >= 10).astype(int)

fuzzy_acc = accuracy_score(actual_class, fuzzy_class)
ml_acc = accuracy_score(actual_class, ml_class)

# -------------------------
# DISPLAY METRICS (CARDS)
# -------------------------
st.markdown("### 🔍 Regression Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Fuzzy MAE", f"{fuzzy_mae:.2f}")
col2.metric("Fuzzy RMSE", f"{fuzzy_rmse:.2f}")
col3.metric("Fuzzy R²", f"{fuzzy_r2:.3f}")

col1, col2, col3 = st.columns(3)

col1.metric("ML MAE", f"{ml_mae:.2f}")
col2.metric("ML RMSE", f"{ml_rmse:.2f}")
col3.metric("ML R²", f"{ml_r2:.3f}")

# -------------------------
# 🎯 ACCURACY
# -------------------------
st.markdown("### 🎯 Classification Accuracy")

col1, col2 = st.columns(2)

col1.metric("Fuzzy Accuracy", f"{fuzzy_acc*100:.2f}%")
col2.metric("ML Accuracy", f"{ml_acc*100:.2f}%")

# -------------------------
# 🎯 LIVE PREDICTOR
# -------------------------
st.markdown("## 🎯 Live Prediction")

col1, col2 = st.columns(2)

with col1:
    study = st.slider("Study Time", 0, 4, 2)
    absences = st.slider("Absences", 0, 30, 5)
    g1 = st.slider("G1", 0, 20, 10)
    g2 = st.slider("G2", 0, 20, 10)
    predict = st.button("Predict")

with col2:
    if predict:
        sim = create_fuzzy_system()
        sim.input['study'] = study/4
        sim.input['absences'] = absences/30
        sim.input['g1'] = g1/20
        sim.input['g2'] = g2/20
        sim.compute()
        result = sim.output['g3'] * 20

        st.markdown(f'<div class="result-box">Score: {result:.2f}</div>', unsafe_allow_html=True)

        if result >= 10:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

# -------------------------
# DOWNLOAD
# -------------------------
result_df = pd.DataFrame({
    "Actual": actual,
    "Fuzzy": fuzzy_preds,
    "ML": ml_preds
})

st.download_button("📥 Download Results", result_df.to_csv(index=False), "results.csv")