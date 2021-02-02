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

print("Null Values in the database (%)")
print(round(df.isnull().sum(axis = 0) * 100 / len(df),2))

dfCompData = pd.read_csv("STCompletionCheck.csv")

#Descriptive Statistics
print(df.describe().T)
print(dfCompData.describe().T)

#Show columns' distributions
def paramsDistribution(df):
    loc = 1
    for col in df:
     if col == "Outcome":
           continue
     else:
        plt.subplot(3, 3, loc)
        plt.hist(df[col], bins=100, alpha=0.5)
        plt.title(col)
        loc = loc + 1
    plt.show()

#Show columns' distributions in diabetic and non diabetic groups distribution
def compareParamsDistribution(df):
    loc = 1
    for col in df:
     if col == "Outcome":
           continue
     else:
        diabetic = df[df["Outcome"] == 1]
        nonDiabetic = df[df["Outcome"] == 0]
        diabeticCol = diabetic[col].dropna()
        nonDiabeticCol = nonDiabetic[col].dropna()
        plt.subplot(3,3,loc)
        plt.hist(diabeticCol, bins=100, alpha=0.5, label="Diabetic")
        plt.hist(nonDiabeticCol, bins=100, alpha=0.5, label="Non-Diabetic")
        plt.title(col)
        plt.legend(loc='upper right')
        loc = loc +1

    plt.show()

def showCorrelation(df):
    corr = df.corr()
    graph = sns.heatmap(corr, annot=True, cmap="Blues")
    plt.setp(graph.get_xticklabels(), rotation=15)
    plt.show()

#Show columns' distributions before and after data completion
paramsDistribution(df)
paramsDistribution(dfCompData)
compareParamsDistribution(df)
compareParamsDistribution(dfCompData)
showCorrelation(df)
showCorrelation(dfCompData)


