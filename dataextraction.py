import numpy as np
import pandas as pd

#input file name here
filename ='heartdata.csv'
df = pd.read_csv(filename)

#replace "present" with 1 and "absent" with 0 in the famhist column
df.famhist = df.famhist.replace('Present',1)
df.famhist = df.famhist.replace('Absent',0)

raw_data = df.values

cols = range(1, 11) # Getting rid of the first column since it's column numbers

# 0: sbp, 1: tobacco, 2: ldl, 3: adiposity, 4: famhist, 5: typea
# 6: obesity, 7: alcohol, 8: age, 9: chd
X = raw_data[:, cols]

N,M = X.shape

X_standard = X - np.ones((N,1))*X.mean(axis=0)
for i in range(len(X_standard[0])):
    X_standard[:,i] /= np.std(X_standard[:,i])
X_standard[:,4] = X[:,4]
X_standard[:,9] = X[:,9]

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

#print(df)
