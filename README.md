# Flight Price Predictor

This program is designed to predict the prices of airline tickets using a machine learning model based on the RandomForestRegressor. It incorporates feature selection with Recursive Feature Elimination (RFE) and hyperparameter tuning using Randomized Search Cross-Validation (RandomizedSearchCV).

## Getting Started
### Prerequisites

Make sure you have the necessary libraries installed (see `requirements.txt`). You can also install needed libraries at a later time.

### Usage

1. Clone the repository:
``` bash
git clone https://github.com/yourusername/flight-price-predictor.git
cd flight-price-predictor
```

2. Make sure that you have all the libraries needed to run the program:
``` bash
pip install -r requirements.txt
```

3. Run the script:
``` bash
python flight_price_predictor.py
```

4. Enter the file path to your training data when prompted.
5. The program will process the data, convert non-numerical features to numerical ones, and show the correlation matrix with respect to the price column.
6. It will split the data into training and testing sets, perform feature selection using RFE, and find the best hyperparameters using Randomized Search CV.
7. Finally, it will use the best model to predict prices on the test set, calculate the accuracy (R-squared score), and export the results to a CSV file named `prediction.csv`.

## Data
Ensure that your training data is in a CSV file format with a 'price' column representing the target variable.

## Features
* Input: CSV file containing training data.
* Output: 'prediction.csv' containing predicted prices on the test set.

## Author
Alex Wang

## License
This project is licensed under the MIT License - see the LICENSE file for details.