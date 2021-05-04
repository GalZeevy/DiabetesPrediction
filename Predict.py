import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import consts
import GeneralFunctions

def trainTestData(df):
    x = df.drop([consts.outcome], 1, inplace=False)
    y = df[consts.outcome]
    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

    #Check outcome distribution in train and test sets
    print(y_train.value_counts())
    print(y_test.value_counts())
    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize = True))

    return x_train, x_test,y_train,y_test

def buildModels():
    estimators = []
    #Logistic Regression
    logit = LogisticRegression(solver = "newton-cg")
    estimators.append(('Logistic Regression', logit))
    #Random Forest
    param_grid = {
        'max_features': [2, 3],
        'n_estimators': [100, 200, 300, 1000]
    }
    RFClassifier = RandomForestClassifier()
    RFClassifierGS = GridSearchCV(estimator = RFClassifier, param_grid = param_grid, refit=True)
    estimators.append(('Random Forest', RFClassifierGS))
    #Naive Bayes
    NB = GaussianNB()
    estimators.append(('Naive Bayes', NB))
    #Support Vectors Machine
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    SVM = GridSearchCV(SVC(), param_grid, refit=True)
    estimators.append(('Support Vector Classifier', SVM))
    #Gradient Boosting Trees
    param_grid = {
                  "max_features":["log2","sqrt"],
                  'n_estimators': [5,10,15,20]
                  }
    GBT = GridSearchCV(GradientBoostingClassifier(), param_grid = param_grid, refit = True)
    estimators.append(('Gradient Boosting Trees', GBT))
    return estimators

def fitModels(estimators, x_train, y_train):
    for name, model in estimators:
        model.fit(x_train, y_train)
    return estimators

def plotModelsCF(estimators, x_train, y_train, x_test,y_test):
    i = 1
    results = pd.DataFrame()
    fitModels(estimators, x_train, y_train)
    plt.title("Confustion matrix of each model")
    for name, model in estimators:
        y_pred = model.predict(x_test)

        confusionMatrix = metrics.confusion_matrix(y_test, y_pred)
        plt.subplot(3,2,i)
        sb.heatmap(pd.DataFrame(confusionMatrix), annot=True, cmap="Blues", fmt='g')
        plt.title(name)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

        results.loc[i, "Model"] = name
        results.loc[i, "Accuracy"] = metrics.accuracy_score(y_test, y_pred)
        results.loc[i, "Precision"] = metrics.precision_score(y_test, y_pred)
        results.loc[i, "Recall"] = metrics.recall_score(y_test, y_pred)

        i=i+1

    print(results)
    plt.subplots_adjust(hspace = 1)
    plt.show()

def showConfusionMatrix(y_test,y_pred, modelName):
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusionMatrix)

    sb.heatmap(pd.DataFrame(confusionMatrix), annot=True, cmap="Blues" ,fmt='g')
    plt.title(modelName  + ' - Confusion Matrix ')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def ensembleCF(estimators):
    voting = VotingClassifier(estimators=estimators)
    voting.fit(x_train, y_train)
    y_pred = voting.predict(x_test)
    showConfusionMatrix(y_test, y_pred, 'Ensemble')
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.precision_score(y_test, y_pred))
    print(metrics.recall_score(y_test, y_pred))

def featureImportance(modelObj):
    RFClassifierGS = modelObj[1]
    importances = RFClassifierGS.best_estimator_.feature_importances_
    plt.title(modelObj[0] + " Feature Importance")
    plt.bar(range(len(importances)), importances, width=0.3, color = "cornflowerblue")
    plt.xticks(range(len(importances)), x_train.columns, rotation = 15)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

df = GeneralFunctions.readFile(consts.dataCompletionFile)
x_train, x_test,y_train,y_test = trainTestData(df)
estimators = buildModels()
estimators = fitModels(estimators,x_train,y_train)
plotModelsCF(estimators, x_train, y_train, x_test,y_test)
ensembleCF(estimators)
featureImportance(estimators[1])














