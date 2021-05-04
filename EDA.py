import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import consts
import sys

def readFile(file):
    try:
        df = pd.read_csv(file)
        return df
    except FileNotFoundError:
        print("File not found")
        sys.exit()

#Show columns' distributions
def paramsDistribution(df):
    loc = 1
    for col in df:
     if col == consts.outcome:
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
     if col == consts.outcome:
           continue
     else:
        diabetic = df[df[consts.outcome] == 1]
        nonDiabetic = df[df[consts.outcome] == 0]
        diabeticCol = diabetic[col].dropna()
        nonDiabeticCol = nonDiabetic[col].dropna()
        plt.subplot(3,3,loc)
        plt.hist(diabeticCol, bins=100, alpha=0.5, label=consts.diabeticLable)
        plt.hist(nonDiabeticCol, bins=100, alpha=0.5, label=consts.nonDiabeticLable)
        plt.title(col)
        plt.legend(loc='upper right')
        loc = loc +1

    plt.show()

def showCorrelation(df):
    corr = df.corr()
    graph = sns.heatmap(corr, annot=True, cmap="Blues")
    plt.setp(graph.get_xticklabels(), rotation=15)
    plt.show()


#Read file before data completion
df = readFile(consts.diabetesFile)
#Read file after data completion
dfCompData = readFile(consts.dataCompletionFile)

#Replace zero with null value
df[consts.glucose] = df[consts.glucose].replace(0, np.nan)
df[consts.bloodPressure] = df[consts.bloodPressure].replace(0, np.nan)
df[consts.skinThickness] = df[consts.skinThickness].replace(0, np.nan)
df[consts.insulin] = df[consts.insulin].replace(0, np.nan)
df[consts.bmi] = df[consts.bmi].replace(0, np.nan)

#Show percentage of missing values in data columns
print("Null Values in the database (%)")
print(round(df.isnull().sum(axis = 0) * 100 / len(df),2))

#Descriptive Statistics
print(df.describe().T)
print(dfCompData.describe().T)

#Show columns' distributions before and after data completion
paramsDistribution(df)
paramsDistribution(dfCompData)
compareParamsDistribution(df)
compareParamsDistribution(dfCompData)
showCorrelation(df)
showCorrelation(dfCompData)


