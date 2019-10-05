import numpy as np
import pandas as pd

# read the data file
df = pd.read_csv("data.csv", header=None)

# break the data file up into components, numpy arrays
ftest = df.iloc[:,0].to_numpy(dtype=float)
ftrain = df.iloc[:,1].to_numpy(dtype=float)
Xtest = df.iloc[:,2:21].to_numpy(dtype=float)
Xtrain = df.iloc[:,21:40].to_numpy(dtype=float)
ytest = df.iloc[:,40].to_numpy(dtype=float)
ytrain = df.iloc[:,41].to_numpy(dtype=float)

