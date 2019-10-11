import numpy as np
from numpy.linalg import inv

class RidgeReg:
	def __init__(self, gamma):
		self.gamma_ = gamma

	def fit(self, X, y):
		# append 1 to end of each row for bias
		X = np.append(X,np.ones((len(X),1)),axis=1)
		# calculate the K (kernel) matrix
		K = np.matmul(X.T,X)
		p = np.matmul(X.T,y)
		# calculate the weights
		self.w_ = np.matmul(inv(K+self.gamma_*np.eye(len(K))), p)
		return self

	def predict(self, X):
		X = np.append(X,np.ones((len(X),1)),axis=1)
		y_pred = np.matmul(self.w_.T,X.T)
		return y_pred

	def score(self, X, y):
		ypred = self.predict(X)
		e = y - ypred
		# return negative error so it works with grid search
		return -np.dot(e,e)/len(y)
	
	def set_params(self, **params):
		self.gamma_ = params["gamma"]
		return self
	
	def get_params(self,deep=False):
		return {"gamma":self.gamma_}

import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# read the data file
df = pd.read_csv("data.csv", header=None)

#
# break the data file up into components as numpy arrays
#
#ftest = df.iloc[:,0].to_numpy()
#ftrain = df.iloc[:,1].to_numpy()
Xtest = df.iloc[:,2:21].to_numpy()
Xtrain = df.iloc[:,21:40].to_numpy()
ytest = df.iloc[:,40].to_numpy()
ytrain = df.iloc[:,41].to_numpy()

# vary gamma from 10^-2 to 10^1
gammas=10**np.linspace(-2,1,101)

# use grid search to find best gamam
param_grid = {'gamma': gammas}
clf = GridSearchCV(RidgeReg(gamma='auto'), param_grid, cv=5, iid=False)
clf = clf.fit(Xtrain, ytrain)

# get the scores and find the gamma with smallest error
scores = -clf.cv_results_['mean_test_score'];
be2 = np.inf
sindex = 0
for i in range(len(scores)):
	if scores[i] < be2:
		be2 = scores[i]
		sindex = i
bgamma=gammas[sindex]

# now test it against the full test set
yhat = clf.predict(Xtest)
e2 = -clf.score(Xtest, ytest)

# plot the gamma search
plt.plot(gammas,scores,label="Error")
plt.scatter([bgamma],[be2])
plt.xlabel("$\gamma$")
plt.ylabel("$e^2$")
plt.xscale(value="log")
plt.suptitle("Validation Mean Square Error vs. $\gamma$")
plt.title("min at $\gamma=%.3f, e^2=%.3g$"%(bgamma,be2),fontsize=10)
plt.show()

# plot real vs. predicted
time = list(range(0,len(yhat)))
plt.plot(time,ytest,label="$y_{test}$")
plt.plot(time,yhat,label="$\hat{y}$")
plt.xlabel("Time")
plt.legend()
plt.suptitle("Real vs. Predicted")
plt.title("$\gamma=%.3f,e^2=%.3g$"%(bgamma,e2),fontsize=10)
plt.show()
