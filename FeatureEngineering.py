import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import consts
import GeneralFunctions

#Divide continuous variables to categories by medical terms
def BMICats(x):
    if x < consts.under_weight_upper_limit:
        return 1
    elif consts.under_weight_upper_limit <= x <= consts.normal_weight_upper_limit:
        return 2
    elif consts.normal_weight_upper_limit <= x < consts.over_weight_upper_limit:
        return 3
    else:
        return 4

def GlucoseCats(x):
    if x < consts.normal_glucose_upper_limit:
        return 1
    elif consts.normal_glucose_upper_limit <= x <= consts.prediabetes_glucose_limit:
        return 2
    elif consts.prediabetes_glucose_limit < x:
        return 3

def BloodPressureCats(x):
    if x < consts.normal_BP_upper_limit:
        return 1
    elif consts.normal_BP_upper_limit <= x < consts.level1_high_BP_upper_limit:
        return 2
    elif consts.level1_high_BP_upper_limit <= x < consts.level2_high_BP_upper_limit:
        return 3
    else:
        return 4

def insulinCats(x):
    if x < consts.normal_insulin_upper_limit:
        return 1
    else:
        return 2

def createCategoriesColumns(df):
    df[consts.bmiCategory] = df[consts.bmi].apply(BMICats)
    df[consts.glucoseCategory] = df[consts.glucose].apply(GlucoseCats)
    df[consts.bloodPressureCategory] = df[consts.bloodPressure].apply(BloodPressureCats)
    df[consts.insulinCategory] = df[consts.insulin].apply(insulinCats)

    dfcats = df[[consts.bmiCategory, consts.bloodPressureCategory, consts.glucoseCategory, consts.insulinCategory,
                 consts.outcome]]
    return df, dfcats

def pieChart(df, column, lables):
    diabetic = df[df[consts.outcome] == 1]
    nonDiabetic = df[df[consts.outcome] == 0]
    plt.subplot(3, 3, 1)
    s = df[column].map(lables).value_counts()
    plt.pie(s, labels=s.index, autopct='%1.1f%%')
    plt.title(column + " - All")
    plt.subplot(3, 3, 2)
    s = diabetic[column].map(lables).value_counts()
    plt.pie(s, labels=s.index, autopct='%1.1f%%')
    plt.title(column + consts.diabeticLable)
    plt.subplot(3, 3, 3)
    s = nonDiabetic[column].map(lables).value_counts()
    plt.pie(s, labels=s.index, autopct='%1.1f%%')
    plt.title(column + consts.nonDiabeticLable)

    plt.show()

df = GeneralFunctions.readFile(consts.diabetesFile)
df, dfcats = createCategoriesColumns(df)
GeneralFunctions.showCorrelation(dfcats,"Correlation of categories",'kendall')

BMI_lables = {1:'Underweight',2:'Normalweight',3:'Overweight', 4:'Obese'}
Glucose_lables = {1: 'Normal', 2:'Prediabetis', 3: 'Diabetis'}
BloodPressure_lables = {1:'Normal', 2: 'High Level 1', 3:'High Level 2', 4:'Highest BP'}
Insulin_lables = {1:'Normal', 2:"High"}

pieChart(df,consts.bmiCategory,BMI_lables)
pieChart(df,consts.glucoseCategory,Glucose_lables)
pieChart(df,consts.bloodPressureCategory,BloodPressure_lables)
pieChart(df,consts.insulinCategory,Insulin_lables)









