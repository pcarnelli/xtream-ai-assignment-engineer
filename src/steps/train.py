# -*- coding: utf-8 -*-
""" Trains the model for the diamonds dataset.

This script applies a pipeline to train a model with the diamond dataset. The
pipeline consists in the following steps: 1) an ordinal encoder for the
categorical features, 2) a function that removes observations (rows) with
negative prices, null z values and/or outliers, 3) a column transformation
that drops the x and y feature columns, and 4) a scikit-learn gradient boosting
regressor.

The provided dataset should be in CSV format and contain these named columns for
features: 'carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity'.
And it should also contain a named column for the target 'price'. See README.md
in the datasets/diamonds folder for more details.

The following third-party packages are required: numpy, pandas,
scipy, scikit-learn, imbalanced-learn, and dill.

This script can be executed from the project's root directory with the following
command:

    $ python src/steps/train.py
    
It can also be imported as a module and contains the following functions:

    * robust_mahalanobis_method - detects outliers.
    * clean_data - preprocess data.
    * create_pipeline - creates a pipeline for data preprocessing and model training.
    * train - trains and evaluates the model.


@author: Patricio Carnelli
@contact: pcarnelli@gmail.com
@credit: Mark Douthwaite, Alicia Horsch
@links: https://github.com/markdouthwaite/serverless-scikit-learn-demo/tree/master,
https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-2-3a3319ec2c33
@license: MIT
@date: 24-Apr-2024
@version: 0.1
"""


import time
import json
from typing import Optional, Union, List, Any
from datetime import datetime

import dill
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler


# Timestamp format
TIMESTAMP_FMT = '%Y%m%d-%H:%M:%S'

# Names of features, target, and categorical features' categories
NUMERIC_FEATURES: List[str] = ['carat', 'depth', 'table', 'x', 'y', 'z']
CATEGORICAL_FEATURES: List[str] = ['cut', 'color', 'clarity']
TARGET: str = 'price'
CATEGORIES_CUT: List[str] = ['Fair', 'Good', 'Very Good', 'Ideal', 'Premium']
CATEGORIES_COLOR: List[str] = list(map(chr, range(ord('D'), ord('J')+1)))[::-1]
CATEGORIES_CLARITY: List[str] = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
CATEGORIES: List[List[str]] = [CATEGORIES_CUT, CATEGORIES_COLOR, CATEGORIES_CLARITY]


def robust_mahalanobis_method(df: pd.core.frame.DataFrame) -> List[int]:

    """Multivariate outlier detection by the robust Mahalanobis method.

    Args:
        df (pd.core.frame.DataFrame): Input data frame.

    Returns:
        List[int]: List of indices of the detected outlier observations.
    """
    
    # Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(df.values.T)
    X = rng.multivariate_normal(mean=np.mean(df, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_ # Robust covariance
    robust_mean = cov.location_ # Robust mean
    inv_covmat = sp.linalg.inv(mcd) # Inverse covariance
    
    # Robust Mahalanobis distance
    x_minus_mu = df - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    # Chi2-based cutoff values for outlier detection
    # Degrees of freedom (df) = number of features
    # Significance level = 0.1%
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))

    # Detect outliers
    outlier = []
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    
    return outlier


def clean_data(X: Union[np.ndarray, pd.core.frame.DataFrame],
          y: Union[np.ndarray, pd.core.frame.DataFrame],
          numeric_features: List[str],
          categorical_features: List[str],
          target: str,
          remove_price_neg: bool = True,
          remove_z_zero: bool = True,
          remove_outliers: bool = False,
) -> tuple[pd.core.frame.DataFrame, pd.core.series.Series]:
    
    """Removes from the diamonds dataset observations (rows) with negative
    prices, null z values and/or outliers (identified by function
    robust_mahalanobis_method).

    Args:
        X (Union[np.ndarray, pd.core.frame.DataFrame]): Feature values array or data frame.
        y (Union[np.ndarray, pd.core.frame.DataFrame]): Target values array or data frame.
        numeric_features (List[str]): List of names of the numeric features.
        categorical_features (List[str]): List of names of the categorical features.
        target (str): Name of the target variable.
        remove_price_neg (bool, optional): Indicates if negative price values should be removed. Defaults to True.
        remove_z_zero (bool, optional): Indicates if null z values should be removed. Defaults to True.
        remove_outliers (bool, optional): Indicates if outliers should be detected and removed. Defaults to False.

    Returns:
        tuple[pd.core.frame.DataFrame, pd.core.series.Series]: Cleansed feature and target arrays or data frames.
    """

    features = numeric_features + categorical_features
    target = target

    # Convert input to dataframe
    X = pd.DataFrame(X, columns=features)
    y = pd.DataFrame(y, columns=[target])
    df: pd.core.frame.DataFrame = X.join(y)
    df_pre = df.copy()

    # Remove observations with price <= 0
    if remove_price_neg:
        df_pre = df_pre[df_pre[target]>0]

    # Remove observations with z = 0
    if remove_z_zero:
        df_pre = df_pre[df_pre['z']!=0]

    # Remove outliers detected by function robust_mahalanobis_method
    if remove_outliers:
        outliers = robust_mahalanobis_method(df_pre)
        outlier_flags = [i in outliers for i in df_pre.index]
        df_pre = df_pre[np.logical_not(outlier_flags)]

    return df_pre.drop(columns=[target]), df_pre[target]


def create_pipeline(
        numeric_features: List[str],
        categorical_features: List[str],
        categories: List[List[str]],
        target: str,
        remove_price_neg: bool = True,
        remove_z_zero: bool = True,
        remove_outliers: bool = False
) -> Pipeline:
    
    """Creates a pipeline for the diamonds dataset's model.

    Args:
        numeric_features (List[str]): List of names of the numeric features.
        categorical_features (List[str]): List of names of the categorical features.
        categories (List[List[str]]): Lists with the categories for each categorical feature.
        target (str): Name of the target variable.
        remove_price_neg (bool, optional): Indicates if negative price values should be removed. Defaults to True.
        remove_z_zero (bool, optional): Indicates if null z values should be removed. Defaults to True.
        remove_outliers (bool, optional): Indicates if outliers should be detected and removed. Defaults to False.

    Returns:
        Pipeline: An imbalanced-learn pipeline object with steps described above.
    """

    # Define column transformations
    # Leave numerical features unchanged
    transformers = [('num', 'passthrough', numeric_features)]
    # Encode categorical features
    for i, name in enumerate(categorical_features):
        transformers.append((name, OrdinalEncoder(categories=[categories[i]],
                                                  dtype=int),
                                                  [categorical_features[i]]))
        encoder = ColumnTransformer(transformers)
    
    # Apply function clean to remove observations (rows) with negative prices,
    # null z values and/or outliers
    cleaner = FunctionSampler(func=clean_data, kw_args={
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'target': target,
        'remove_price_neg': remove_price_neg,
        'remove_z_zero': remove_z_zero,
        'remove_outliers': remove_outliers,
    })

    # Drop columns x and y
    dropper = ColumnTransformer([
        ('drop','drop',['x','y']),
    ], remainder='passthrough')
    
    # Define and return pipeline
    return Pipeline(steps=[
        ('encoder', encoder),
        ('cleaner', cleaner),
        ('dropper', dropper),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            )),
    ])


def train(
        path: str='datasets/diamonds/diamonds.csv',
        test_size: float = 0.2,
        dump: bool = True,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        categories: Optional[List[List[str]]] = None,
        target: Optional[str] = None,
) -> None:
    
    """Trains and evaluates the model for the diamonds dataset.

    Args:
        path (str, optional): Dataset location. Defaults to 'datasets/diamonds/diamonds.csv'.
        test_size (float, optional): Fraction of the dataset for testing. Defaults to 0.2.
        dump (bool, optional): Indicates if the model object and a JSON file with metrics should be written to disk. Defaults to True.
        numeric_features (Optional[List[str]], optional): List of names of the numeric features.
        categorical_features (Optional[List[str]], optional): List of names of the categorical features.
        categories (Optional[List[List[str]]], optional): Lists with the categories for each categorical feature. Defaults to None.
        target (Optional[str], optional): Name of the target variable. Defaults to None.
    """
    
    start = time.time()

    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    if categories is None:
        categories = CATEGORIES

    if target is None:
        target = TARGET

    # Load data
    data = pd.read_csv(path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=[target]), data[target], test_size=test_size)
    
    # Initialize and train model
    model = create_pipeline(
        numeric_features = numeric_features,
        categorical_features = categorical_features,
        categories = categories,
        target = target,
    )
    model.fit(X_train, y_train)

    # Calculate and print scores (r-squared) and training time
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    time_train = time.time() - start
    print(f'Train score: {score_train:.4f}')
    print(f'Test score: {score_test:.4f}')
    print(f'Elapsed time: {time_train:.2f}')

    # Store metrics and timestamp in a dictionary
    metrics = dict(
        score_train = score_train,
        score_test = score_test,
        time_train = time_train,
        timestamp = datetime.now().strftime(TIMESTAMP_FMT),
    )

    # Save model object and metrics to disk
    if dump:
        dill.settings['recurse'] = True
        dill.dump(model, open('models/pipeline.joblib', 'wb'))
        json.dump(metrics, open('models/metrics.json', 'w'))


if __name__ == '__main__':

    """Calls train function for diamonds' model training when this script is
    executed.
    """
    
    train()
