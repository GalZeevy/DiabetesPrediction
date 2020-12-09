import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("diabetes.csv")

df["Glucose"] = df["Glucose"].replace(0, np.nan)
df["BloodPressure"] = df["BloodPressure"].replace(0, np.nan)
df["SkinThickness"] = df["SkinThickness"].replace(0, np.nan)
df["Insulin"] = df["Insulin"].replace(0, np.nan)
df["BMI"] = df["BMI"].replace(0, np.nan)

#To each column create a distribution graph that contains only not nan values, and shows diabetic and non diabetic groups distribution
for col in df:
    if col == "Outcome":
        continue
    else:
        diabetic = df[df["Outcome"] == 1]
        nonDiabetic = df[df["Outcome"] == 0]
        diabeticCol = diabetic[col].dropna()
        nonDiabeticCol = nonDiabetic[col].dropna()
        plt.hist(diabeticCol, bins=100, alpha=0.5, label="Diabetic")
        plt.hist(nonDiabeticCol, bins=100, alpha=0.5, label="Non-Diabetic")
        plt.xlabel(col, size=14)
        plt.title(col)
        plt.legend(loc='upper right')
        plt.show()
