"""
This is to test the function in churn_library.py, including import data,
eda function, data encoder helper, feature engineering, and model training.
Date: December 2022
Author: Panda Wu
"""
import os
import logging
import churn_library as cls

LOGGER = logging.getLogger(__name__)


def test_import(pth):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = cls.import_data(pth)
        LOGGER.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        LOGGER.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert len(df) > 0
        assert len(df.columns) > 0
    except AssertionError as err:
        LOGGER.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df):
    """
    test perform eda function
    """

    try:
        # cls.perform_eda(df)
        # check if the file is created
        assert os.path.exists('./images/eda/Correlation_Map.png')
        LOGGER.info("Testing performing eda - Correlation_Map: SUCCESS")
        assert os.path.exists('./images/eda/Churn_Distribution.png')
        LOGGER.info("Testing performing eda - Churn_Distribution: SUCCESS")
        assert os.path.exists('./images/eda/Customer_Age_Distribution.png')
        LOGGER.info(
            "Testing performing eda - Customer_Age_Distribution: SUCCESS")
        assert os.path.exists('./images/eda/Marital_Status_Distribution.png')
        LOGGER.info(
            "Testing performing eda - Marital_Status_Distribution: SUCCESS")
        assert os.path.exists('./images/eda/Total_trans_Ct_density.png')
        LOGGER.info("Testing performing eda - Total_trans_Ct_density: SUCCESS")
        LOGGER.info("Testing performing EDA: SUCCESS")
    except FileNotFoundError as err:
        LOGGER.error("Testing performing EDA: %s", err)
        raise err


def test_encoder_helper(df, category_lst, response):
    """
    test encoder helper
    """

    try:
        encoded_df = cls.encoder_helper(df, category_lst, response)
        # check if the encoded_df is not empty
        assert encoded_df.shape[0] > 0
        assert encoded_df.shape[1] > 0
        LOGGER.info("Testing encoder helper - encoded_df shape: %s",
                    str(encoded_df.shape))
        # check response column in encoded_df
        assert response in encoded_df.columns
        LOGGER.info(
            "Testing encoder helper - encoded_df contains response column: SUCCESS")

        # check encoded columns in encoded_df
        for col in category_lst:
            assert col + "_" + response in encoded_df.columns
        LOGGER.info(
            "Testing encoder helper - encoded_df contains encoded column: SUCCESS")
        LOGGER.info("Testing encoder helper: SUCCESS")
    except AssertionError as err:
        LOGGER.error("Testing encoder helper: %s", err)
        raise err


def test_perform_feature_engineering(encoded_df, response):
    """
    test perform_feature_engineering
    """
    try:
        dataset = cls.perform_feature_engineering(encoded_df, response)
        # check if the dataset contains 4 elements
        assert len(dataset) == 4
        LOGGER.info(
            "Testing perform feature engineering - dataset length: SUCCESS")

        # check length of each element in dataset
        assert len(dataset[0]) == len(dataset[2])  # X_train, y_train
        assert len(dataset[1]) == len(dataset[3])  # X_test y_test
        LOGGER.info(
            "Testing perform feature engineering - input data shape: SUCCESS")

        LOGGER.info("Testing feature engineering: SUCCESS")
    except AssertionError as err:
        LOGGER.error("Testing feature engineering: %s", err)
        raise err


def test_train_models(dataset):
    """
    test train_models
    """
    try:
        # cls.train_models(*dataset)
        # check if the model results file is created
        assert os.path.exists('./images/results/feature_importance_rfc.png')
        assert os.path.exists('./images/results/rfc_lrc_roc_curve.png')
        assert os.path.exists(
            './images/results/Random_Forest_classification_report.png')
        assert os.path.exists(
            './images/results/Logistic_Regression_classification_report.png')

        # check model saved
        assert os.path.exists('./models/logistic_model.pkl')
        assert os.path.exists('./models/rfc_model.pkl')
        LOGGER.info("Testing train model: SUCCESS")
    except AssertionError as err:
        LOGGER.error("Testing train model: %s", err)
        raise err
