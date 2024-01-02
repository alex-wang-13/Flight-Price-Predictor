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
print()

"""
Convert non-numerical features to numerical ones (or remove them).
"""
def make_numerical(df: pd.DataFrame) -> pd.DataFrame:
  """
  A function to make all the columns in the dataframe numerical.
  This function also generates some features.

  Parameters:
  - df (pd.DataFrame): A DataFrame similar to the sample provided.

  Returns:
  - pd.DataFrame: An all-numerical DataFrame.
  """

  # Convert boolean columns to numerical.
  boolean_columns = df.select_dtypes(include="bool").columns.tolist()
  for col in boolean_columns:
    df[col] = df[col].astype(int)

  if "searchDate" in df and "flightDate" in df:
    df["searchDate"] = pd.to_datetime(df["searchDate"], format="%Y-%m-%d")
    df["flightDate"] = pd.to_datetime(df["flightDate"], format="%Y-%m-%d")
    # Created an arbitrary reference date.
    reference_date = dt.datetime(2022, 1, 1)
    # A feature representing the number of days since the reference date.
    df["yearToFlightDate"] = (df["flightDate"] - reference_date).dt.days
    # A feature representing the number of days between search and flight dates.
    df["daysToFlightDate"] = (df["flightDate"] - df["searchDate"]).dt.days

  if "travelDuration" in df:
    # Converting the datetime string to the total duration of the trip in seconds.
    df["travelDuration"] = pd.to_timedelta(df["travelDuration"])
    df["travelDuration"] = df["travelDuration"].dt.total_seconds()

  return df.select_dtypes(include="number")

"""
Get the correlation matrix to the price column.
"""
train = make_numerical(train)
corr_matrix = train.corr()['price']
print(f"correlation matrix w/ respect to price:\n{corr_matrix}")

