import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm
from time import time
from sklearn.model_selection import GridSearchCV

# values for grid search
C_values = [.5, 1, 5, 10, 50, 100]
eps_values = [0.0005, 0.001, 0.005, 0.01, 0.1, 0.25, 0.5]
nu_values =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# set to True for SVR
# set to False for NuSVR
svr=True

if svr:
	Classifier = svm.SVR
	cname = "SVR"
	param_grid = {'C': C_values,
              'epsilon': eps_values }
else:
	Classifier = svm.NuSVR
	cname = "NuSVR"
	param_grid = {'C': C_values,
              'nu': nu_values }

# read the data file
df = pd.read_csv("data.csv", header=None)

#
# break the data file up into components as numpy arrays
#
#ftest = df.iloc[:,0].to_numpy()
#ftrain = df.iloc[:,1].to_numpy()
Xtest = df.iloc[:,2:21].to_numpy().T
Xtrain = df.iloc[:,21:40].to_numpy().T
ytest = df.iloc[:,40].to_numpy()
ytrain = df.iloc[:,41].to_numpy()

# fit the classifier to training data
clf = GridSearchCV(Classifier(gamma='auto'), param_grid, cv=5, iid=False)
clf = clf.fit(Xtrain.T, ytrain)

# extract scores and best parameters
scores = clf.cv_results_['mean_test_score'];
C = clf.best_estimator_.get_params()["C"]
if svr:
	scores = scores.reshape(len(C_values),len(eps_values))
	epsilon = clf.best_estimator_.get_params()["epsilon"]
else:
	nu = clf.best_estimator_.get_params()["nu"]
	scores = scores.reshape(len(C_values),len(nu_values))

#plt.figure(figsize=(8, 6))
#plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.ylabel('C')

if svr:
	plt.xlabel(r"$\epsilon$")
	plt.title("Best: C=%d, $\epsilon=%.3g$"%(C, epsilon))
else:
	plt.xlabel(r"$\nu$")
	plt.title(r"Best: C=%d, $\nu=%.3g$"%(C, nu))

plt.colorbar()
plt.yticks(np.arange(len(C_values)), C_values)
if svr:
	plt.xticks(np.arange(len(eps_values)), eps_values)
else:
	plt.xticks(np.arange(len(nu_values)), nu_values)

plt.suptitle('{} Grid Search Score'.format(cname))
plt.show()

yhat = clf.predict(Xtest.T)
e = ytest - yhat
e2 = np.dot(e,e)/len(ytest)

trange = list(range(0,len(yhat)))
plt.plot(trange,ytest,label="$y_{test}$")
plt.plot(trange,yhat,label="$\hat{y}$")
plt.xlabel("Time")
plt.legend()
plt.suptitle("{} Real vs. Predicted".format(cname))
if svr:
	plt.title("C=%d, $\epsilon=%.3g, e^2=%.3g$"%(C, epsilon, e2))
else:
	plt.title(r"C=%d, $\nu=%.3g, e^2=%.3g$"%(C, nu, e2))
plt.show()
