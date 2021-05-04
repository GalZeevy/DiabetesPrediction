import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import consts
import GeneralFunctions

#definitions 1. result column - column in df which its data is completed with knn.
# 2. reference column - column in df which has the highest correlation with the result column.
# ref column is used as reference for the completion of the data in the result column by knn algorithm

#Check correlation of dataframe columns with a specific column existing values (only data that is not empty)
def CorrOnColExistingValues(df, col):
    dfWithColExistValues = df[df[col] != 0]
    title = "Correlation of " + col + " existing values with dataframe columns"
    GeneralFunctions.showCorrelation(dfWithColExistValues, title, 'pearson')


#Compare reference column distributions when result column has only empty values and when it has only full values
def compareDistributions(df, refCol, resCol):
    bins = 50
    dfResColFull = df[df[resCol] != 0]
    dfResColEmpty = df[df[resCol] == 0]
    plt.hist(dfResColFull[refCol], bins, alpha=0.5, label= refCol + 'Full')
    plt.hist(dfResColEmpty[refCol], bins, alpha=0.5, label= refCol + 'Empty')
    plt.title('Compare distributions of ' + refCol + " when " +resCol + " is empty and full of values")
    plt.xlabel(refCol)
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

def knnModel(df,refCol, resCol):
    dfResColFull = df[df[resCol] != 0]
    refColFull = dfResColFull[refCol]
    resColFull = dfResColFull[resCol]
    refColFull = refColFull.values.reshape(-1,1)
    MMScaler = MinMaxScaler(feature_range=(0, 1))
    scaledRefColFull = MMScaler.fit_transform(refColFull)
    scaledRefColFullDF = pd.DataFrame(scaledRefColFull)

    X_train, X_test, y_train, y_test = train_test_split(scaledRefColFullDF,resColFull,test_size=0.2, random_state = 21)

    #train and test
    minK, rmse_val, minRmse, pred = minKForKNN(X_train, y_train, X_test, y_test)

    curve = pd.DataFrame(rmse_val) #elbow curve
    curve.plot(xlabel = 'K', ylabel = 'rmse', legend = False)
    plt.title("KNN Elbow Curve")
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
    dfResColEmpty = df[df[resCol] == 0]
    refColEmpty = dfResColEmpty[refCol].values.reshape(-1,1)
    MMScaler = MinMaxScaler(feature_range=(0, 1))
    scaledRefColEmpty = MMScaler.fit_transform(refColEmpty)
    scaledRefColEmptyDF = pd.DataFrame(scaledRefColEmpty)
    pred = knn.predict(scaledRefColEmptyDF)
    print("KNN Prediction:",pred)

    print("Pred_Mean:",np.mean(pred))
    print("Pred_STD:",np.std(pred))

    predDF = pd.DataFrame(pred)
    predDF = predDF.rename(columns={0: resCol})
    resColEmptyRefCol = dfResColEmpty[refCol].reset_index(drop=True)
    predDF[refCol] = resColEmptyRefCol

    #fill predicted values in dataframe
    for index, row in df.iterrows():
        if row[resCol] == 0:
            predSTrow = predDF.loc[predDF[refCol] == row[refCol]]
            resColDf = predSTrow[resCol].head(1)
            print(resColDf.values[0])
            df.loc[index, resCol] = resColDf.values[0]
    return df

def fillMissingValuesInCol(df, colName):
    # Zero's represent missing values, I replaced them with nans
    df[colName] = df[colName].replace(0,np.nan)
    print(df[colName].describe())
    # In the above column mean and median are almost the same, so mean was used because it includes all the data.
    if colName != consts.bmi:
        df[colName] = df[colName].fillna(round(df[colName].mean()))
    else:
        df[colName] = df[colName].fillna(round(df[colName].mean(),1))
    return df

df = GeneralFunctions.readFile(consts.diabetesFile)
df = fillMissingValuesInCol(df,consts.glucose)
df = fillMissingValuesInCol(df, consts.bloodPressure)
df = fillMissingValuesInCol(df, consts.bmi)
CorrOnColExistingValues(df,consts.skinThickness)
CorrOnColExistingValues(df,consts.insulin)
compareDistributions(df,consts.bmi, consts.skinThickness)
compareDistributions(df,consts.glucose, consts.insulin)
df = knnModel(df, consts.bmi, consts.skinThickness)
df = knnModel(df, consts.glucose, consts.insulin)
GeneralFunctions.showCorrelation(df, "Correlation test after data completion",'pearson')

df.to_csv(consts.dataCompletionFile, index = False)