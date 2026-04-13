import pandas as pd
from fuzzy_model import create_fuzzy_system
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("student-por.csv", sep=';')
df = df[['studytime','absences','G1','G2','G3']]
df.columns = ['study','absences','g1','g2','g3']

# 🔹 Classification (Pass/Fail)
df['pass'] = df['g3'].apply(lambda x: 1 if x >= 10 else 0)

# Fuzzy predictions
fuzzy_preds = []

for i in range(len(df)):
    sim = create_fuzzy_system()

    sim.input['study'] = df.iloc[i]['study']
    sim.input['absences'] = min(df.iloc[i]['absences'], 30)
    sim.input['g1'] = df.iloc[i]['g1']
    sim.input['g2'] = df.iloc[i]['g2']

    sim.compute()
    fuzzy_preds.append(sim.output['g3'])

# ML model
X = df[['study','absences','g1','g2']]
y = df['g3']

model = LinearRegression()
model.fit(X, y)
ml_preds = model.predict(X)

print("Fuzzy MAE:", mean_absolute_error(y, fuzzy_preds))
print("ML MAE:", mean_absolute_error(y, ml_preds))