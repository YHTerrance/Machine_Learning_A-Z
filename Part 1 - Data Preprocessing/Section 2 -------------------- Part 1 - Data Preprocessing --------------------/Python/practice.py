# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# All columns except last one
X = dataset.iloc[:, :-1].values
# Only the last column
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
# Create an instance of the class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# One-hot encoding of country names (first column [0])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# force it to become a numpy array
X = np.array(ct.fit_transform(X))

# Transform dependent variable y
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into the Training set and Test set (8:2 split), fixing random seed
# Do this before feature scaling to avoid data leakage (feature scaling should not have access to test set)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature scaling (avoid feature being dominated by other features)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
# Use the same fitting
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)
