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

def showConfusionMatrix(y_test,y_pred, modelName):
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusionMatrix)

    sb.heatmap(pd.DataFrame(confusionMatrix), annot=True, cmap="Blues" ,fmt='g')
    plt.title(modelName  + ' - Confusion Matrix ')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def showRFFeatImoprtances(RFclassifier):
    importances = RFClassifier.feature_importances_
    print(importances)
    std = np.std([tree.feature_importances_ for tree in RFClassifier.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    features_names = df.columns
    feature_names = [features_names[i] for i in indices]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x_train.shape[1]), importances[indices],
        color="b", align="center")
    plt.xticks(range(x_train.shape[1]), feature_names)

    plt.xlabel("Feature", size = '10')
    plt.ylabel("Importance", size = '10')
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

#One model confustion matrix (For example)
logit = LogisticRegression(solver = "newton-cg")
logit.fit(x_train, y_train)
y_pred = logit.predict(x_test)

showConfusionMatrix(y_test,y_pred, "Logistic Regression")

#Initialize all models
estimators = []
columns = ["Model", "Accuracy", "Presicion", "Recall"]
results = pd.DataFrame()
logit = LogisticRegression(solver = "newton-cg")
estimators.append(('Logistic Regression', logit))
RFClassifier = RandomForestClassifier(n_estimators=200, random_state = 0)
estimators.append(('Random Forest', RFClassifier))
NB = GaussianNB()
estimators.append(('Naive Bayes', NB))
SVM = SVC(random_state=0)
estimators.append(('Support Vector Classifier', SVM))
GBT = GradientBoostingClassifier(random_state=0)
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

showRFFeatImoprtances(RFClassifier)
#ROC Curve - to complete