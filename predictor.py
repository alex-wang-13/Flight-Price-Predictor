import pandas as pd
import datetime as dt

from sklearn.linear_model import LinearRegression

"""
A program to predict the price column of a .csv file dataframe.
"""

DEFAULT = "training-data.csv"
TRAIN_FP = input("Enter the file path to the training data: ")
TRAIN_FP = TRAIN_FP.strip()
if not TRAIN_FP:
  TRAIN_FP = DEFAULT

"""
Get the training data.
"""
train: pd.DataFrame = pd.read_csv(TRAIN_FP)
train.info()

"""
Get the correlation matrix to the price column.
"""
corr_matrix = train.corr()['price']
print(corr_matrix)