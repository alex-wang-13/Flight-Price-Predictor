import pandas as pd
import datetime as dt

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score

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
data: pd.DataFrame = pd.read_csv(TRAIN_FP)
data.info()
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
data = make_numerical(data)
data.dropna(inplace=True)
corr_matrix = data.corr()['price']
print(f"correlation matrix w/ respect to price:\n{corr_matrix}\n")

"""
Split test and training data.
"""
X = data.drop(labels='price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

"""
Feature selection with recursive feature elimination (RFE)
"""
estimator = RandomForestRegressor(n_estimators=10)
selector = RFE(estimator, n_features_to_select=5)
selector.fit(X_train, y_train)
relevant_features = X_train.columns[selector.support_].tolist()
print(f"Most relavent features: {relevant_features}\n")

"""
Find the best parameters on the best model.
"""
rf_model = RandomForestRegressor(random_state=42)
param_grid: dict = {
  "n_estimators": [5, 10, 20, 30],
  "max_depth": [None, 10, 20, 30],
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2, 4]
}
random_search = RandomizedSearchCV(
  rf_model, param_distributions=param_grid, n_iter=5, scoring="r2", cv=5, random_state=42, verbose=1
)

"""
Search for the best model with most relevant features.
"""
random_search.fit(X_train[relevant_features], y_train)
best_model: RandomForestRegressor = random_search.best_estimator_
best_params = random_search.best_params_

"""
Use the best model to predict make predictions on test set.
"""
y_pred = best_model.predict(X_test[relevant_features])
error = r2_score(y_test, y_pred)
print(f"Best model: {best_model}\nBest parameters: {best_params}\n")
print(f"Accuracy (r^2 score): {error}")

"""
Export results.
"""
results: pd.DataFrame = X_test[relevant_features]
results.loc[:, "price"] = y_pred
results.to_csv("prediction.csv", index=False)