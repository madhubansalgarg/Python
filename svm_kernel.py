#Classfication#

#It is a supervised machine-learning approach. SVM  is the problem of 
#identifying to which of a set of categories (sub-populations) a new observation belongs,
#on the basis of a training set of data containing observations (or instances) 
#whose category membership is known

#Data#

#For the purpose of demonstration We shall use the  impact of  social advertizement on purchase of car.


#Variables in data and description#
#Input variables:
#User data:
 # User ID (numeric)
 # Gender : male/female(categorical)
 # Age : numeric
 # EstimatedSalary : education (categorical)

#Output variable (desired target):
  #Has the user bought the car? 0,1


import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt


# read file

df=pd.read_csv('Social_Network_Ads.csv')

x=df.iloc[:,[2,3]].values
y= df.iloc[:,4].values

# divide into test and train data

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

# Step 3 : Execute classification

from sklearn.svm import SVC

#linear classifier

classifier_linear=SVC(kernel='linear',random_state=0)
classifier_linear.fit(x_train,y_train)
y_pred=classifier_linear.predict(x_test)

from sklearn.metrics import confusion_matrix
cm_linear=confusion_matrix(y_test,y_pred)

#k-fold validation
from sklearn.model_selection import cross_val_score
accuracies_linear = cross_val_score(estimator=classifier_linear, X=x_train, y=y_train, cv=10 )
accuracies_linear.mean()
accuracies_linear.std()

# Visualising the Training set results
from matplotlib.colors import ListedColormap

x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_linear.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results

x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_linear.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Gaussian kernel

classifier_rbf=SVC(kernel='rbf',random_state=0)
classifier_rbf.fit(x_train,y_train)
y_pred=classifier_rbf.predict(x_test)
cm_rbf=confusion_matrix(y_test,y_pred)

accuracies_rbf = cross_val_score(estimator=classifier_rbf, X=x_train, y=y_train, cv=10 )
accuracies_rbf.mean()
accuracies_rbf.std()
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_rbf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results

x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_rbf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#polynomial kernel
classifier_poly=SVC(kernel='poly',degree = 2,random_state=0)
classifier_poly.fit(x_train,y_train)
y_pred=classifier_poly.predict(x_test)
cm_poly=confusion_matrix(y_test,y_pred)
accuracies_poly = cross_val_score(estimator=classifier_poly, X=x_train, y=y_train, cv=10 )
accuracies_poly.mean()
accuracies_poly.std()

# With current information  SVM-Gaussin Kernel is better

#using gridsearch to find best model and best best param

parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
              {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.001,0.001]}]

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=classifier_rbf, param_grid=parameters,scoring='accuracy', cv=10,n_jobs=-1)
grid_search=grid_search.fit(x_train,y_train)
best_accuracy=grid_search.best_score_
best_params = grid_search.best_params_

#best parameters in the current set is 0.5. Taking it atmore granule level

parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
              {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid_search = GridSearchCV(estimator=classifier_rbf, param_grid=parameters,scoring='accuracy', cv=10,n_jobs=-1)
grid_search=grid_search.fit(x_train,y_train)
best_accuracy1=grid_search.best_score_
best_params1 = grid_search.best_params_

#best model for  maximum accuracy is gausssian kernel with gamma parameters as  0.7. 
