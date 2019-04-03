#Author - Keerthana Jayaprakash
# Support Vector Machine - Machine Learning Project

from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
data_USA = pd.read_csv('USA_Dataset.csv')
data_USA.drop(['ccf','painloc','painexer','relrest','pncaden',
               'smoke', 'dm','thaltime','dummy','rldv5','ca',
                'restckm','exerckm','restef','restwm','exeref','exerwm',
               'thalsev','thalpul','earlobe', 'lmt','ladprox',
               'laddist','diag','cxmain','ramus','om1','om2','rcaprox',
               'rcadist','lvx1', 'lvx2','lvx3','lvx4',
               'lvf','cathef','junk'], axis = 1,inplace = True)
condition = data_USA['num'] == 0
data_USA['target'] = np.where(condition, 0 , 1)
data_USA_target = data_USA['target']
data_USA.drop(['num','id','target'],axis = 1, inplace = True)
data_USA = pd.get_dummies(data_USA, columns= ['cp','restecg','slope','thal','loc'])
data_std = Standardize(data_USA)
data_std['target'] = data_USA_target
print("Data preprocessed...")
data = data_std.as_matrix()
train_x, test_x, train_y, test_y = train_test_split(data[:, 0:-1], data[:,-1],train_size=0.75)
param_grid = {'kernel': ['linear'],
               'C': [5]}
print("Implementing SVM...")
model = svm.SVC()
clf = GridSearchCV(model, param_grid, cv = 10)
clf.fit(train_x, train_y)
Training_score = clf.score(train_x, train_y,)
predicted= clf.predict(test_x)
accuracy = accuracy_score(test_y, predicted)
print("Training Accuracy is ", Training_score)
print("The Test Accuracy is ", accuracy)
results = clf.best_params_
print(results)
print("Building confusion matrix...")
cm = pd.DataFrame(confusion_matrix(test_y, predicted), columns = [0 ,1], index = [0,1])
sensitivity = cm[0][0]/(cm[0][0]+cm[0][1])
print('Sensitivity : ', sensitivity)
specificity = cm[1][1]/(cm[1][0]+cm[1][1])
print('Specificity : ', specificity)
print("The best parameters are :", results)
sns.heatmap(cm, annot=True)
plt.show()
