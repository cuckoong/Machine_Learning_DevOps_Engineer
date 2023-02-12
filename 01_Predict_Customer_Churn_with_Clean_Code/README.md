# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This Project use data from churn library. 
It includes data exploration, data preprocessing, feature engineering, model training, and model evaluation. 
- You may use the customer's data (Age, dependents count, months on book, total relationship count,
months inactive, contacts count, credit limit, total revolving balance, average open to buy, total amount change from Q1 to Q4, 
total transaction count, total transaction amount, total transaction count change from Q1 to Q4, average utilisation ratio,
gender, education level, marital status, income category, card category) to predict whether the customer will churn or not.
- Model used for this project is Logistic Regression and Random Forest.
- Trained model will be saved for future use. Reports will be generated for model evaluation.
- The project also includes test files for all functions.

## Files and data description
The project structure:
```
Predict Customer Churn
├── data
│   ├── bank_data.csv
│── images
│   ├── eda
│   ├── results
│── logs
    │── churn_library.log
│── models
│   ├── logistic_model.pkl
    │── rfc_model.pkl
├── churn_library.py
├── churn_script_logging_and_test.py
├── conftest.py
│── pytest.ini
├── requirements.txt
├── README.md

```
- `data`: contains the data used in the project
- `images`: the data exploration images in ./images/eda and the results images in ./images/results
- `logs`: contains the log files
- `models`: contains the model files (.pkl)
- `churn_library.py`: python scripts that contains all functions used in the project
- `churn_script_logging_and_test.py`: python scripts that contains the logging and test functions
- `conftest.py`: python scripts that contains the test fixtures
- `pytest.ini`: python scripts that contains the pytest configuration
- `requirements.txt`: contains the required packages
- `README.md`: contains the project description

## Running Files
- step 1: install necessary environment package using ```python -m pip install -r requirements_py3.8.txt```;

- step 2: run the churn_library.py file using ```python churn_library.py```;

- step2: run test file using ``` pytest churn_script_logging_and_tests.py``` to run and test the script of 
churn_library.py, and the log files are generated in the ```./logs``` folder;

## Dependencies and Libraries
- Python 3.8
- Pandas 1.2.4
- Matplotlib 3.3.4
- Seaborn 0.11.2
- Scikit-learn 0.24.1
- shap 0.40.0
- pytest 6.2.4
- pylint 2.7.4
- autopep8 1.5.6