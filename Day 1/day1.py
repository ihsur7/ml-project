import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import os

# print(os.getcwd())

dataset = pd.read_csv('datasets/Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# print(X)
# print(Y)

print(type(X[:,1:3]))

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[ : , 1:3])

X[ : , 1:3] = imputer.transform(X[ : , 1:3])

