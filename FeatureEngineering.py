import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

df = pd.read_csv("diabetes.csv")

#consts

under_weight_upper_limit = 18.5
normal_weight_upper_limit = 24.9
over_weight_upper_limit = 29.9

normal_glucose_upper_limit = 140
prediabetes_glucose_limit = 199

normal_insulin_upper_limit = 150

normal_BP_upper_limit = 80
level1_high_BP_upper_limit = 90
level2_high_BP_upper_limit = 120

#Divide continuous variables to categories by medical terms
def BMICats(x):
    if x < under_weight_upper_limit:
        return 1
    elif under_weight_upper_limit <= x <= normal_weight_upper_limit:
        return 2
    elif normal_weight_upper_limit <= x < over_weight_upper_limit:
        return 3
    else:
        return 4

def GlucoseCats(x):
    if x < normal_glucose_upper_limit:
        return 1
    elif normal_glucose_upper_limit <= x <= prediabetes_glucose_limit:
        return 2
    elif prediabetes_glucose_limit < x:
        return 3

def BloodPressureCats(x):
    if x < normal_BP_upper_limit:
        return 1
    elif normal_BP_upper_limit <= x < level1_high_BP_upper_limit:
        return 2
    elif level1_high_BP_upper_limit <= x < level2_high_BP_upper_limit:
        return 3
    else:
        return 4

def insulinCats(x):
    if x < normal_insulin_upper_limit:
        return 1
    else:
        return 2

def pieChart(df, column, lables):
    diabetic = df[df["Outcome"] == 1]
    nonDiabetic = df[df["Outcome"] == 0]
    plt.subplot(3, 3, 1)
    s = df[column].map(lables).value_counts()
    plt.pie(s, labels=s.index, autopct='%1.1f%%')
    plt.subplot(3, 3, 2)
    s = diabetic[column].map(lables).value_counts()
    plt.pie(s, labels=s.index, autopct='%1.1f%%')
    plt.subplot(3, 3, 3)
    s = nonDiabetic[column].map(lables).value_counts()
    plt.pie(s, labels=s.index, autopct='%1.1f%%')

    plt.show()

df['BMICategory'] = df['BMI'].apply(BMICats)
df['GlucoseCategory'] = df['Glucose'].apply(GlucoseCats)
df['BloodPressureCategory'] = df['BloodPressure'].apply(BloodPressureCats)
df['InsulinCategory'] = df['Insulin'].apply(insulinCats)

dfcats = df[['BMICategory','BloodPressureCategory', 'GlucoseCategory', 'InsulinCategory','Outcome']]

#Check correlation of new categories with the outcome
corr = dfcats.corr(method = 'kendall')
print(corr)
graph = sns.heatmap(corr, annot = True, cmap= "Blues")
plt.setp(graph.get_xticklabels(), rotation=45)
plt.show()

BMI_lables = {1:'Underweight',2:'Normalweight',3:'Overweight', 4:'Obese'}
Glucose_lables = {1: 'Normal', 2:'Prediabetis', 3: 'Diabetis'}
BloodPressure_lables = {1:'Normal', 2: 'High Level 1', 3:'High Level 2', 4:'Highest BP'}
Insulin_lables = {1:'Normal', 2:"High"}

pieChart(df,"BMICategory",BMI_lables)
pieChart(df,"GlucoseCategory",Glucose_lables)
pieChart(df,"BloodPressureCategory",BloodPressure_lables)
pieChart(df,"InsulinCategory",Insulin_lables)









