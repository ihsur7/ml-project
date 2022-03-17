import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing  import StandardScaler

from sklearn.compose import ColumnTransformer

import os

# print(os.getcwd())

dataset = pd.read_csv('datasets/Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# print(X)
# print(Y)

# print(type(X[:,1:3]))

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[ : , 1:3])

X[ : , 1:3] = imputer.transform(X[ : , 1:3])

labelencoder_X = LabelEncoder()

X[ : , 0] = labelencoder_X.fit_transform(X[:, 0])

ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder='passthrough')

# onehotencoder = OneHotEncoder(categorical_features = [0])

# X = onehotencoder.fit_transform(X).toarray()

X = ct.fit_transform(X)

labelencoder_Y = LabelEncoder()

Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

print(X, Y, X_train, X_test)