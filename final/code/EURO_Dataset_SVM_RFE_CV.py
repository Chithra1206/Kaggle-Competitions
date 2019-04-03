#Author - Keerthana Jayaprakash
# Support Vector Machine - Machine Learning Project

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def Standardize(df):
    for i in df.columns:
        Mean = np.mean(df[i])
        StdDev = np.std(df[i])
        if StdDev == 0:
            df.drop(i,axis = 1, inplace = True)
        else:
            df[i] = (df[i] - Mean)/StdDev
    return df

np.random.seed(123)
EURO_Data = pd.read_csv('EURO_Dataset.csv')
EURO_Data.drop(['ccf','smoke', 'cigs','years','famhist','dummy','ca',
                'restckm','exerckm','restef','restwm','exeref','exerwm',
               'thalsev','thalpul','earlobe', 'lmt','ladprox',
               'laddist','diag','cxmain','ramus','om1','om2','rcaprox',
               'rcadist','lvx1', 'lvx2','lvx3','lvx4',
               'lvf','cathef','junk'], axis = 1,inplace = True)
condition = EURO_Data['num'] == 0
EURO_Data['target'] = np.where(condition, 0 , 1)
data_EURO_target = EURO_Data['target']
EURO_Data.drop(['num','id','target'],axis = 1, inplace = True)
EURO_Data = pd.get_dummies(EURO_Data, columns= ['cp','restecg','slope','thal','loc'])
data_std = Standardize(EURO_Data)
data_std['target'] = data_EURO_target
print("Data preprocessed...")
data = data_std.as_matrix()
train_x, test_x, train_y, test_y = train_test_split(data[:, 0:-1], data[:,-1],train_size=0.75)
names = list(EURO_Data.columns.values)
print("Executing Recursive Feature Elimination in SVM...")
svc = SVC(kernel="linear", C=25)
rfecv  = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10),
              scoring='accuracy')
rfecv.fit(train_x, train_y)
rfecv.score(train_x, train_y)
predicted= rfecv.predict(test_x)
accuracy = accuracy_score(test_y, predicted)
print("The support array \n",rfecv.support_)
print("The ranking array \n",rfecv.ranking_)
type(rfecv.grid_scores_)
print(sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), names)))
print("Test Accuracy is ", accuracy)
print("The Cross-validation score :" ,max(rfecv.grid_scores_))
print("Optimal number of features : 26")

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

