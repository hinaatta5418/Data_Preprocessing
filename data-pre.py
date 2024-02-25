#Scikit-learn, often referred to as sklearn, is an open-source machine learning library for Python. It provides simple and efficient tools for data analysis and modeling, including various machine learning algorithms, preprocessing techniques, and utilities for tasks such as classification, regression, clustering, dimensionality reduction, and more.
import pandas as pd #used for handling numbers
import numpy as np #used for handling dataset
from sklearn.impute import SimpleImputer # used for handling missing values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #LabelEncoder is used to convert categorical labels (text-based) into numerical format. OneHotEncoder, on the other hand, is used to convert categorical data into a binary matrix (0s and 1s). It creates a new binary column for each category and indicates the presence (1) or absence (0) of that category in each row.
from sklearn.model_selection import train_test_split #used for splitting trainingand testing data you can train your model on one portion (the training set) and then evaluate its performance on another portion (the testing set)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler # used for feature scaling(StandardScaler helps you achieve this by transforming your data such that it has a mean of 0 and a standard deviation of 1.)


ds = pd.read_csv('C:\\Users\\hinaa\\Documents\\Data Analyst\\DataAnalysisUsingPythonproject\\Data-Preprocessing_Lab1\\DataPreprocessing.csv')
print(ds)

# Splitting the attributes into independent and dependent attributes
X=ds.iloc[: , :-1].values # iloc stands for integer-location based indexing in pandas
#iloc[: , :-1] #selecting all rows and all columns except the last one
#.values means all the values will be appeared 
#X contain all the depedent variable 
print(X)

Y=ds.iloc[: , -1] #Y contain all the independent variable or target variable of your dataset
#[: , -1] all the rows and last column 
print(Y)


#Handling Missing Data
#handling the missing data and replace missing values with nan from numpy and replace with mean of all the other values
imputer=SimpleImputer(missing_values=np.nan , strategy='mean') # It is a class that provides basic strategies for imputing missing values. The strategy parameter allows you to choose the imputation strategy, such as mean, median, most_frequent (mode), or constant.
#fit: The fit method is used to compute the mean (or other statistics based on the chosen strategy) for each column in the specified dataset.
imputer=imputer.fit(X[: , 1:])
#Also, make sure the columns you're imputing have numerical data, as mean imputation is suitable for numeric features.
X[: , 1:]=imputer.transform(X[: , 1:])

print(X)

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Use ColumnTransformer for one-hot encoding the first column in X
preprocessor = ColumnTransformer(
    transformers=[
        ('onehotencoder', OneHotEncoder(), [0])  # Replace [0] with the actual index of the categorical column
    ],
    remainder='passthrough'  # Keep the non-transformed columns as they are
)

# Fit and transform X
X = preprocessor.fit_transform(X)

# Use LabelEncoder for encoding the target variable Y
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
random_state=0)

#feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

