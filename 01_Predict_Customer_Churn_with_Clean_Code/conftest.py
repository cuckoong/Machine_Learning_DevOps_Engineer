# content of conftest.py
import pytest
from churn_library import *


@pytest.fixture(scope="session")
def pth():
    return "./data/bank_data.csv"


@pytest.fixture(scope="session")
def df():
    return import_data("./data/bank_data.csv")


@pytest.fixture(scope="session")
def category_lst():
    return ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']


@pytest.fixture(scope="session")
def response():
    return 'Churn'


@pytest.fixture(scope="session")
def encoded_df(df, category_lst, response):
    df_res = encoder_helper(df, category_lst, response)
    return df_res


@pytest.fixture(scope="session")
def dataset(encoded_df, response):
    return perform_feature_engineering(encoded_df, response)
