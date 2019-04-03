import pandas as pdata
import numpy as np

from scipy.stats import spearmanr

import seaborn as sb
import matplotlib.pyplot as plt
from pylab import rcParams
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
%matplotlib inline
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

Euro_dataset = pdata.read_csv('C:/Users/ujji/Desktop/ML/Euro_Dataset.csv')

unused_cols = ['loc','ccf','smoke','cigs','years',
                'famhist','dummy','ca','restckm','exerckm',
                'restef','restwm','exeref','exerwm','thalsev',
                'thalpul','earlobe','lmt','ladprox','laddist',
                'diag','cxmain','ramus','om1','om2','rcaprox',
                'rcadist','lvx1','lvx2','lvx3','lvx4','lvf','cathef','junk']

Euro_dataset.drop(unused_cols, axis = 1,inplace = True)
print(Euro_dataset.head())

#Checking for missing values
Euro_dataset.isnull().sum()

#Checking if predictor variable is binary or ordinal
sb.countplot(x='num', data = Euro_dataset, palette='hls')

#Checking that the dataset size is sufficient
Euro_dataset.info()

#heart_disease = 0 means absence of Heart Disease and 1 means presence of Heart Disease
#Hence replacing all values other than 0 to 1
Euro_dataset['num'].replace(to_replace=[1,2,3,4],value=1,inplace=True)
Euro_dataset = Euro_dataset.rename(columns = {"num":"heart_disease"})
print(Euro_dataset.tail(10))

print (" Number of patients in dataframe: %i\n Number of patients with disease: %i\n Number of patients without disease: %i\n"
      % (len(Euro_dataset.index),len(Euro_dataset[Euro_dataset.heart_disease==1].index),
         len(Euro_dataset[Euro_dataset.heart_disease==0].index)))
print (Euro_dataset.head())
print (Euro_dataset.describe())

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

#Xall = scale(data)
#Xall = Euro_dataset.ix[3,:].values
#Euro_data = Euro_dataset.ix[:,(1,3)].values
Euro_data = Euro_dataset.ix[:,0:40].values
yall = Euro_dataset.ix[:,41].values
#Xall = Euro_dataset[new_columns_2[1:]].values

LogReg = LogisticRegression()
model = LogReg.fit(Euro_data,yall)
print("Logistic Regression score on full data set:%f" %model.score(Euro_data,yall))

y_pred = LogReg.predict(Euro_data)
from sklearn.metrics import classification_report
print("Classification Report on the full dataset:")
print(classification_report(yall, y_pred))

print("Confusion matrix:")
print(metrics.confusion_matrix(yall,y_pred))


####################################################################
X_train, X_test, y_train, y_test = train_test_split(Euro_data, yall, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

y_pred1 = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred1)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred1))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

Xsel0 = [patient for (patient,status) in zip(Euro_data,yall) if status==0] # No-disease cases
Xsel1 = [patient for (patient,status) in zip(Euro_data,yall) if status==1] # Disease cases
print('\nNumber of disease cases: %i, no-disease cases: %i' %(len(Xsel1),len(Xsel0)))
Xsel0_Prob = [p1 for (p0,p1) in model.predict_proba(Xsel0)] # Predicted prob. of heart disease for no-disease cases
Xsel1_Prob = [p1 for (p0,p1) in model.predict_proba(Xsel1)]
fig, axes = plt.subplots( nrows=1, ncols=1, figsize=(6,6) )
plt.subplots_adjust( top=0.92 )
plt.suptitle("Euro Data Set", fontsize=20)
axes.hist(Xsel0_Prob,color=["chartreuse"],histtype="step",label="no-disease cases (208)")
axes.hist(Xsel1_Prob,color=["crimson"],histtype="step",label="disease cases (274)")
axes.set_xlabel("Predicted Probability of Disease",fontsize=15)
axes.set_ylabel("Number of Patients",fontsize=15)
axes.set_ylim( 0.0, 90.0 )
axes.legend(prop={'size': 15},loc="upper right")
plt.show()
##############################GRID-SEARCH#######################
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
parameters = {'penalty' :['l1','l2'], 
              'C': [1,10,100]}
model = LogisticRegression()
clf = GridSearchCV(model, parameters, cv=10)
clf.fit(X_train, y_train)
Training_score = clf.score(X_train, y_train,)
predicted= clf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print("Training Accuracy is ", Training_score)
print("The Test Accuracy is ", accuracy)
results = clf.best_params_
print("The best parameters are :", results)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))