import numpy as np
import pandas as pdata

import urllib
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split

from sklearn import metrics
from sklearn.metrics import accuracy_score

Euro_dataset = pdata.read_csv('C:/Users/ujji/Desktop/ML/Euro_Dataset.csv')

unused_cols = ['loc','ccf','smoke','cigs','years',
                'famhist','dummy','ca','restckm','exerckm',
                'restef','restwm','exeref','exerwm','thalsev',
                'thalpul','earlobe','lmt','ladprox','laddist',
                'diag',	'cxmain','ramus','om1','om2','rcaprox',
                'rcadist','lvx1','lvx2','lvx3','lvx4','lvf','cathef','junk']

Euro_dataset.drop(unused_cols, axis = 1,inplace = True)
print(Euro_dataset.head())

Euro_data = Euro_dataset.ix[:,0:40].values
yall = Euro_dataset.ix[:,41].values

#heart_disease = 0 means absence of Heart Disease and 1 means presence of Heart Disease
#Hence replacing all values other than 0 to 1
Euro_dataset['num'].replace(to_replace=[1,2,3,4],value=1,inplace=True)
Euro_dataset = Euro_dataset.rename(columns = {"num":"heart_disease"})
print(Euro_dataset.tail(10))

X_train, X_test, y_train, y_test = train_test_split(Euro_data, yall, test_size=0.25, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

BernNB = BernoulliNB(binarize = True)
BernNB.fit(X_train, y_train)
print(BernNB)

y_expect = y_test
y_pred = BernNB.predict(X_test)
print(accuracy_score(y_expect, y_pred))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = BernoulliNB()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
nb_roc_auc = roc_auc_score(y_test, BernNB.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, BernNB.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Naive Bayes (area = %0.2f)' % nb_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
#######GRIDSEARCH######################
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
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
X_train, X_test, y_train, y_test = train_test_split(Euro_data, yall, test_size=0.25, random_state=0)
parameters = {'alpha' :[1.0,2.0,5.0,10.0]}
model = BernoulliNB()
clf = GridSearchCV(model, parameters, cv=10)
clf.fit(X_train, y_train)
Training_score = clf.score(X_train, y_train,)
predicted= clf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print("Training Accuracy is ", Training_score)
print("The Test Accuracy is ", accuracy)
results = clf.best_params_
print("The best parameters are :", results)
