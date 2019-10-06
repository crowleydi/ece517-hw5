import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm

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

#
# append a 1 for a bias weight
# transpose so each xi is a column
Xtrain = np.append(Xtrain,np.ones((len(ytrain),1)),axis=1).T
Xtest = np.append(Xtest,np.ones((len(ytest),1)),axis=1).T

# calculate the trace because gamma
# is a scale of the trace
#trace = np.trace(K)

iterations = 500
ngamma = 100
C = 1

# values for plotting gamma vs. error, etc.
gammas = np.zeros((ngamma+1,))
errors = np.zeros((ngamma+1,))

# calculate ranges for 80/20 split
idx=list(range(0,len(ytrain)))
split=len(idx)*8//10

for i in tqdm(range(0,iterations)):
	# shuffle the data indexes
	random.shuffle(idx)
	# split train and test samples
	# training portion
	tr=idx[0:split]
	Xtr = Xtrain[:,tr]
	ytr = ytrain[tr]
	# testing portion
	te=idx[split:]
	Xte = Xtrain[:,te]
	yte = ytrain[te]

	# vary gamma from 10^-2 to 10^1
	pmin = -2
	pmax = 1
	for j in range(0,ngamma+1):
		gamma = 10**(j*(pmax-pmin)/ngamma+pmin)
		gammas[j] = gamma

		clf = svm.SVR(gamma=gamma)
		clf.fit(Xtr.T, ytr)
		yhat = clf.predict(Xte.T)
		e = yte - yhat
		e2 = np.dot(e,e)/len(yhat)
		errors[j] = errors[j] + e2

# now find the lowest error/gamma
lowe = np.inf
errors = errors / iterations
for i in range(0,len(gammas)):
	if (errors[i] < lowe):
		lowe = errors[i]
		gamma = gammas[i]

clf = svm.SVR(gamma=gamma)
clf.fit(Xtrain.T, ytrain)
yhat = clf.predict(Xtest.T)
e = ytest - yhat
e2 = np.dot(e,e)/len(ytest)

plt.plot(gammas,errors,label="Error")
#plt.plot(gammas,losses,label="Loss")
plt.scatter([gamma],[lowe])
plt.xlabel("$\gamma$")
plt.ylabel("$e^2$")
plt.xscale(value="log")
#plt.legend()
plt.suptitle("Training Mean Square Error vs. $\gamma$")
plt.title("min at $\gamma=%.3f, e^2=%.3g$"%(gamma,lowe),fontsize=10)
plt.show()

time = list(range(0,len(yhat)))
plt.plot(time,ytest,label="$y_{test}$")
plt.plot(time,yhat,label="$\hat{y}$")
plt.xlabel("Time")
plt.legend()
plt.suptitle("Real vs. Predicted")
plt.title("$\gamma=%.3f,e^2=%.3g$"%(gamma,e2),fontsize=10)
plt.show()
