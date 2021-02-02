import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

def showCorrelation(df):
    corr = df.corr()
    graph = sns.heatmap(corr, annot=True, cmap="Blues")
    plt.setp(graph.get_xticklabels(), rotation=15)
    plt.show()

df = pd.read_csv("diabetes.csv")

#Zero's represent missing values, I replaced them with nans
df["Glucose"] = df["Glucose"].replace(0, np.nan)
df["BloodPressure"] = df["BloodPressure"].replace(0, np.nan)
df["BMI"] = df["BMI"].replace(0, np.nan)

#Filling the missing values in columns Glucose, BloodPressure and BMI
glucose = df["Glucose"]
bloodPressure = df["BloodPressure"]
BMI = df["BMI"]

#Check columns descriptive statisics (To see if there are significant differences between mean, median and mode of each one)
print(glucose.describe())
print(bloodPressure.describe())
print(BMI.describe())

#In the above columns mean and median are almost the same. I will prefer to use the mean, which includes all the data.
df["Glucose"] = df["Glucose"].fillna(round(glucose.mean()))
df["BloodPressure"] = df["BloodPressure"].fillna(round(bloodPressure.mean()))
df["BMI"] = df["BMI"].fillna(round(BMI.mean(),1))

#Check correlation of dataframe variables with column (only data that is filled)

def checkExistingValuesCorr (col):
    dfFullColVals = df[df[col] != 0]
    showCorrelation(dfFullColVals)

#Compare xCol distributions when yCol has values and when it doesn't have
def emptyAndFullColValuesDistributions(xCol, yCol):
    bins = 50
    dfyColFull = df[df[yCol] != 0]
    dfyColEmpty = df[df[yCol] == 0]
    plt.hist(dfyColFull[xCol], bins, alpha=0.5, label= xCol + 'Full')
    plt.hist(dfyColEmpty[xCol], bins, alpha=0.5, label= xCol + 'Empty')
    plt.xlabel(xCol)
    plt.legend(loc='upper right')
    plt.show()

def minKForKNN(X_train, y_train, X_test, y_test):
    rmse_val = []
    minK = 0
    minRmse = 1000
    for K in range(1, 20):
        knn = KNeighborsRegressor(n_neighbors=K)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmse_val.append(rmse) #store rmse values
        print('RMSE value for k= ' , K , 'is:', rmse)
        if rmse < minRmse:
            minRmse = rmse
            minK = K

    return minK, rmse_val, minRmse, pred

def knnModel(df,xCol, yCol):
    dfyColFull = df[df[yCol] != 0]
    xColFull = dfyColFull[xCol]
    yColFull = dfyColFull[yCol]
    xColFull = xColFull.values.reshape(-1,1)
    MMScaler = MinMaxScaler(feature_range=(0, 1))
    scaledxColFull = MMScaler.fit_transform(xColFull)
    scaledxColFullDF = pd.DataFrame(scaledxColFull)

    X_train, X_test, y_train, y_test = train_test_split(scaledxColFullDF,yColFull,test_size=0.2, random_state = 21)

    #train and test
    minK, rmse_val, minRmse, pred = minKForKNN(X_train, y_train, X_test, y_test)

    curve = pd.DataFrame(rmse_val) #elbow curve
    curve.plot(xlabel = 'K', ylabel = 'rmse')
    plt.show()

    print("minRmse:", minRmse)
    print("minK:", minK)
    print("y_train mean", np.mean(y_train))
    print("y_train std", np.std(y_train))
    print("y_test mean", np.mean(y_test))
    print("y_test std", np.std(y_test))
    print("pred mean", np.mean(pred))
    print("pred std", np.std(pred))

    #predict (write a function for duplicate code)
    knn = KNeighborsRegressor(n_neighbors=minK)
    knn.fit(X_train, y_train)
    dfyColEmpty = df[df[yCol] == 0]
    xColEmpty = dfyColEmpty[xCol].values.reshape(-1,1)
    MMScaler = MinMaxScaler(feature_range=(0, 1))
    scaledxColEmpty = MMScaler.fit_transform(xColEmpty)
    scaledxColEmptyDF = pd.DataFrame(scaledxColEmpty)
    pred = knn.predict(scaledxColEmptyDF)
    print("KNN Prediction:",pred)

    print("Pred_Mean:",np.mean(pred))
    print("Pred_STD:",np.std(pred))

    predDF = pd.DataFrame(pred)
    predDF = predDF.rename(columns={0: yCol})
    yColEmptyXCol = dfyColEmpty[xCol].reset_index(drop=True)
    predDF[xCol] = yColEmptyXCol

    #fill predicted values in dataframe
    for index, row in df.iterrows():
        if row[yCol] == 0:
            predSTrow = predDF.loc[predDF[xCol] == row[xCol]]
            yColDf = predSTrow[yCol].head(1)
            print(yColDf.values[0])
            df.loc[index, yCol] = yColDf.values[0]
    return df


checkExistingValuesCorr("SkinThickness")
checkExistingValuesCorr("Insulin")
emptyAndFullColValuesDistributions("BMI", "SkinThickness")
emptyAndFullColValuesDistributions("Glucose", "Insulin")
df = knnModel(df, "BMI", "SkinThickness")
df = knnModel(df, "Glucose", "Insulin")

showCorrelation(df)

df.to_csv("DataCompletionCheck.csv", index = False)