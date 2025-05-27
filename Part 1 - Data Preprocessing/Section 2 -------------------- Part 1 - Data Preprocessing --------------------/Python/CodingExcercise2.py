# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
# Load the dataset
df = pd.read_csv('pima-indians-diabetes.csv')
x = df.iloc[:, :-1].values

# Identify missing data (assumes that missing data is represented as NaN)

# Print the number of missing entries in each column
print(df.isnull().sum())

# Configure an instance of the SimpleImputer class
imputer = SimpleImputer()
# Fit the imputer on the DataFrame
imputer.fit(x)
# Apply the transform to the DataFrame
x = imputer.transform(x)
#Print your updated matrix of features
print(df)