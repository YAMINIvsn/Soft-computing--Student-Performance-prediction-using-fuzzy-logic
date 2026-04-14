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
st.set_page_config(page_title="Student Performance Analytics", layout="wide")

# -------------------------
# INDUSTRY-LEVEL UI THEME (CLEAN + ANIMATED)
# -------------------------
st.markdown("""
<style>

/* Animated dark gradient background */
body {
    background: linear-gradient(-45deg, #0a0f1c, #111827, #0f172a, #1e293b);
    background-size: 400% 400%;
    animation: gradientBG 14s ease infinite;
    color: #e5e7eb;
}

/* animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Main container spacing */
.block-container {
    padding: 2.5rem 3rem;
}

/* Title styling */
.title {
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #60a5fa, #3b82f6, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}

/* Subheaders */
h2, h3 {
    color: #e5e7eb;
}

/* Clean metric cards */
div[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 10px;
    backdrop-filter: blur(8px);
}

/* Plot container */
.css-1v0mbdj img {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Student Performance Analytics Dashboard</div>', unsafe_allow_html=True)

# -------------------------
# LOAD DATA
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
# FUZZY MODEL
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

# -------------------------
# LINEAR MODEL
# -------------------------
X = df[['studytime','absences','G1','G2']]
y = df['G3']

model = LinearRegression()
model.fit(X, y)
ml_preds = model.predict(X)

# -------------------------
# INSIGHTS
# -------------------------
st.markdown("## Insights")

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
st.markdown("## Correlation Heatmap")

corr = df[['G1','G2','G3','studytime','absences']].corr()

fig = plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
st.pyplot(fig)

# -------------------------
# MULTI FEATURE ANALYSIS
# -------------------------
st.markdown("## Multi Feature Analysis")

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

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
st.markdown("## Feature Importance")

fig = plt.figure()
plt.bar(['study','absences','G1','G2'], model.coef_)
st.pyplot(fig)

# -------------------------
# ERROR DISTRIBUTION
# -------------------------
st.markdown("## Error Distribution")

fig = plt.figure()
plt.hist(actual - fuzzy_preds)
st.pyplot(fig)

# -------------------------
# PREDICTED VS ACTUAL
# -------------------------
st.markdown("## Predicted vs Actual")

fig = plt.figure()
plt.scatter(actual, fuzzy_preds)
st.pyplot(fig)

# -------------------------
# MODEL COMPARISON
# -------------------------
st.markdown("## Model Comparison")

fig = plt.figure()
plt.plot(actual[:50], label="Actual")
plt.plot(fuzzy_preds[:50], label="Fuzzy")
plt.plot(ml_preds[:50], label="ML")
plt.legend()
st.pyplot(fig)

# -------------------------
# PASS FAIL
# -------------------------
st.markdown("## Pass vs Fail")

fig = plt.figure()
plt.pie((df['G3']>=10).value_counts(), labels=["Pass","Fail"], autopct='%1.1f%%')
st.pyplot(fig)

# -------------------------
# 3D VISUALIZATION
# -------------------------
st.markdown("## 3D Visualization")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['G1'], df['G2'], df['G3'])
st.pyplot(fig)

# -------------------------
# RESIDUALS
# -------------------------
st.markdown("## Residual Analysis")

fig = plt.figure()
plt.scatter(ml_preds, actual-ml_preds)
plt.axhline(0)
st.pyplot(fig)

# -------------------------
# CUMULATIVE ERROR
# -------------------------
st.markdown("## Cumulative Error")

fig = plt.figure()
plt.plot(np.cumsum(np.abs(actual - fuzzy_preds)))
st.pyplot(fig)

# -------------------------
# METRICS
# -------------------------
st.markdown("## Model Performance Dashboard")

fuzzy_mae = mean_absolute_error(actual, fuzzy_preds)
ml_mae = mean_absolute_error(actual, ml_preds)

fuzzy_rmse = np.sqrt(mean_squared_error(actual, fuzzy_preds))
ml_rmse = np.sqrt(mean_squared_error(actual, ml_preds))

fuzzy_r2 = r2_score(actual, fuzzy_preds)
ml_r2 = r2_score(actual, ml_preds)

actual_class = (actual >= 10).astype(int)
fuzzy_class = (fuzzy_preds >= 10).astype(int)
ml_class = (ml_preds >= 10).astype(int)

fuzzy_acc = accuracy_score(actual_class, fuzzy_class)
ml_acc = accuracy_score(actual_class, ml_class)

col1, col2, col3 = st.columns(3)
col1.metric("Fuzzy MAE", f"{fuzzy_mae:.2f}")
col2.metric("Fuzzy RMSE", f"{fuzzy_rmse:.2f}")
col3.metric("Fuzzy R²", f"{fuzzy_r2:.3f}")

col1, col2, col3 = st.columns(3)
col1.metric("ML MAE", f"{ml_mae:.2f}")
col2.metric("ML RMSE", f"{ml_rmse:.2f}")
col3.metric("ML R²", f"{ml_r2:.3f}")

col1, col2 = st.columns(2)
col1.metric("Fuzzy Accuracy", f"{fuzzy_acc*100:.2f}%")
col2.metric("ML Accuracy", f"{ml_acc*100:.2f}%")

# -------------------------
# LIVE PREDICTOR
# -------------------------
st.markdown("## Live Prediction")

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

        st.markdown(f"### Predicted Score: {result:.2f}")

        if result >= 10:
            st.success("PASS")
        else:
            st.error("FAIL")

# -------------------------
# DOWNLOAD
# -------------------------
result_df = pd.DataFrame({
    "Actual": actual,
    "Fuzzy": fuzzy_preds,
    "ML": ml_preds
})

st.download_button("Download Results", result_df.to_csv(index=False), "results.csv")