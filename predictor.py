import pandas as pd
import datetime as dt

from sklearn.linear_model import LinearRegression

TRAIN_FP = input("Enter the file path to the training data: ")

"""
Get the training data.
"""
train = pd.read_csv(TRAIN_FP)
train.info()

