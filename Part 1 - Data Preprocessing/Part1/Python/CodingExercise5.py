# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Wine Quality Red dataset
data = pd.read_csv("winequality_red.csv")

# Separate features and target
features = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# Split the dataset into an 80-20 training-test set
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 1)

# Create an instance of the StandardScaler class
sc = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
sc.fit(x_train, y_train)
sc.transform(x_train, y_train)

# Apply the transform to the test set
sc.transform(x_test, y_test)

# Print the scaled training and test datasets
print(x_train)
print(x_test)