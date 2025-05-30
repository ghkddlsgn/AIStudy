# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset
data = pd.read_csv('titanic.csv')

# Identify the categorical data
passenger_id = data.iloc[:, 0].values
survived = data[:, 1].values
name = data.iloc[:, 3].values

# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
passenger_id = np.array(ct.fit_transform(passenger_id))

# Apply the fit_transform method on the instance of ColumnTransformer


# Convert the output into a NumPy array


# Use LabelEncoder to encode binary categorical data


# Print the updated matrix of features and the dependent variable vector
