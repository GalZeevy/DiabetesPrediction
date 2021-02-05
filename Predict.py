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

def showConfusionMatrix(y_test,y_pred, modelName):
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusionMatrix)

    sb.heatmap(pd.DataFrame(confusionMatrix), annot=True, cmap="Blues" ,fmt='g')
    plt.title(modelName  + ' - Confusion Matrix ')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


df = pd.read_csv("STCompletionCheck.csv")

x = df.drop(["Outcome"],1, inplace = False)
y = df["Outcome"]

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#Check outcome distribution in train and test sets
print(y_train.value_counts())
print(y_test.value_counts())
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize = True))

#Initialize all models
estimators = []
columns = ["Model", "Accuracy", "Presicion", "Recall"]
results = pd.DataFrame()
logit = LogisticRegression(solver = "newton-cg")
estimators.append(('Logistic Regression', logit))
#Choose suitable parameters 200 is one of the estimators
param_grid = {
    'max_features': [2, 3],
    'n_estimators': [100, 200, 300, 1000]
}
#RFClassifier = RandomForestClassifier(n_estimators=200, random_state = 0)
RFClassifier = RandomForestClassifier()
RFClassifierGS = GridSearchCV(estimator = RFClassifier, param_grid = param_grid, refit=True)
#RFClassifier = RandomForestClassifier(n_estimators=200, random_state = 0)
estimators.append(('Random Forest', RFClassifierGS))
NB = GaussianNB()
estimators.append(('Naive Bayes', NB))

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

SVM = GridSearchCV(SVC(), param_grid, refit=True)
estimators.append(('Support Vector Classifier', SVM))
param_grid = {
              "max_features":["log2","sqrt"],
              'n_estimators': [5,10,15,20]
              }
GBT = GridSearchCV(GradientBoostingClassifier(), param_grid = param_grid, refit = True)
estimators.append(('Gradient Boosting Trees', GBT))
voting = VotingClassifier(estimators = estimators)

#All models results
i = 1
for name, model in estimators:
     model.fit(x_train,y_train)
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

voting.fit(x_train,y_train)
y_pred = voting.predict(x_test)
showConfusionMatrix(y_test, y_pred, 'Ensemble')
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.precision_score(y_test, y_pred))
print(metrics.recall_score(y_test, y_pred))




