"""
Data preprocessing and modeling for customer churn
Date: December 2022
Author: Panda Wu
"""

# import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import constants as c

# from sklearn.preprocessing import normalize

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    df['Churn'] = df[c.churn_label_name].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    plt.figure(figsize=(12, 6))
    df['Churn'].hist()
    plt.title("Churn Distribution")
    plt.tight_layout()
    plt.savefig('images/eda/Churn_Distribution.png')

    plt.figure(figsize=(12, 6))
    df['Customer_Age'].hist()
    plt.title("Customer Age Distribution")
    plt.tight_layout()
    plt.savefig('images/eda/Customer_Age_Distribution.png')

    plt.figure(figsize=(12, 6))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Marital Status Distribution")
    plt.tight_layout()
    plt.savefig('images/eda/Marital_Status_Distribution.png')

    plt.figure(figsize=(12, 6))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title("Marital Status Distribution")
    plt.tight_layout()
    plt.savefig('images/eda/Total_Trans_Ct_density.png')

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Correlation map")
    plt.tight_layout()
    plt.savefig('images/eda/Correlation_Map.png')


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for item in category_lst:
        df[item + "_" + response] = df[item].apply(
            lambda x: df.loc[df[item] == x, response].mean())
    return df


def perform_feature_engineering(df, response):
    """
    performs feature engineering on df and returns X and y
    :param df: pandas dataframe, the data for the model to be trained on
    :param response: string of response name
    :return:
    """
    X = df[c.keep_cols]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(
        model_name,
        y_train,
        y_test,
        y_train_preds,
        y_test_preds):
    """
    produces 2 for training and testing results and stores report as image
    in images folder
    input:
            model_name: string of model name
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions from model
            y_test_preds: test predictions from model
    output:
             None
    """
    # classification_report of random forest
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{model_name} Classification Report')
    axs[0].text(
        0.5,
        0.5,
        classification_report(
            y_train,
            y_train_preds),
        ha='center')
    axs[0].axis('off')
    axs[0].set_title('Training Classification Report')

    axs[1].text(
        0.5,
        0.5,
        classification_report(
            y_test,
            y_test_preds),
        ha='center')
    axs[1].axis('off')
    axs[1].set_title('Test Classification Report')
    plt.tight_layout()
    plt.savefig(f'images/results/{model_name}_classification_report.png')


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importance in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # plots two roc in one figure
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        lrc,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8,
        c='r',
        label='Logistic Regression')
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8,
        c='b',
        label='Random Forest')
    plt.legend()
    plt.savefig('images/results/rfc_lrc_roc_curve.png')

    # classification report
    classification_report_image(
        'Random_Forest',
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf)
    classification_report_image(
        'Logistic_Regression',
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr)

    # feature importance for random forest
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        'images/results/feature_importance_rfc.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')


if __name__ == '__main__':
    # load data
    bank_df = import_data('./data/bank_data.csv')

    # explore data
    perform_eda(bank_df)

    # encode data
    encoded_bank_df = encoder_helper(
        bank_df,
        category_lst=c.category_lst,
        response='Churn')

    # feature engineering
    dataset = perform_feature_engineering(encoded_bank_df, response='Churn')

    # train model
    train_models(*dataset)
