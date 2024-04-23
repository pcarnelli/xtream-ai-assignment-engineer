import time
import json
from typing import Optional, Union, List
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


# TODO: Docstrings


TIMESTAMP_FMT = '%Y%m%d-%H:%M:%S'

NUMERIC_FEATURES: List[str] = ['carat', 'depth', 'table', 'x', 'y', 'z']
CATEGORICAL_FEATURES: List[str] = ['cut', 'color', 'clarity']
CATEGORIES_CUT: List[str] = ['Fair', 'Good', 'Very Good', 'Ideal', 'Premium']
CATEGORIES_COLOR: List[str] = list(map(chr, range(ord('D'), ord('J')+1)))[::-1]
CATEGORIES_CLARITY: List[str] = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
CATEGORIES: List[str] = [CATEGORIES_CUT, CATEGORIES_COLOR, CATEGORIES_CLARITY]
TARGET: str = 'price'


def clean(X: Union[np.ndarray, pd.core.frame.DataFrame],
          y: Union[np.ndarray, pd.core.frame.DataFrame],
          numeric_features: List[str],
          categorical_features: List[str],
          target: Optional[str] = None,
          remove_price_neg: bool = True,
          remove_z_zero: bool = True,
          remove_outliers: bool = False,
) -> pd.core.frame.DataFrame:
            
    features = numeric_features + categorical_features
    
    if y is None:
        pass
    else:
        target = target

    # Convert input to dataframe
    if y is None:
        df_pre = pd.DataFrame(X, columns=features)
    else:
        X = pd.DataFrame(X, columns=features)
        y = pd.DataFrame(y, columns=[target])
        df = X.join(y)
        df_pre = df.copy()

    # Remove observations with price <= 0
    if y is None:
        pass
    else:
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

    if y is None:
        return df_pre
    else:
        return df_pre.drop(columns=[target]), df_pre[target]


# Function for outlier detection
def robust_mahalanobis_method(df: pd.core.frame.DataFrame) -> List[int]:
    
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

    # Chi2-based cutoff values for outlier identification
    # Degrees of freedom (df) = number of variables
    # Significance level = 0.1%
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))

    # Identify outliers
    outlier = []
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    
    return outlier


def create_pipeline(
        numeric_features: List[str],
        categorical_features: List[str],
        categories: List[str],
        target: Optional[str] = None,
        remove_price_neg: bool = True,
        remove_z_zero: bool = True,
        remove_outliers: bool = False
) -> Pipeline:
    
    # Define column transformations
    # Leave numerical features unchanged
    transformers = [('num', 'passthrough', numeric_features)]
    # Encode categorical features
    for i, name in enumerate(categorical_features):
        transformers.append((name, OrdinalEncoder(categories=[categories[i]],
                                                  dtype=int),
                                                  [categorical_features[i]]))
        encoder = ColumnTransformer(transformers)
    
    cleaner = FunctionSampler(func=clean, kw_args={
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'target': target,
        'remove_price_neg': remove_price_neg,
        'remove_z_zero': remove_z_zero,
        'remove_outliers': remove_outliers,
    })

    dropper = ColumnTransformer([
        ('drop','drop',['x','y']),
    ], remainder='passthrough')
    
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
        tag: str = '',
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        target: Optional[str] = None,
) -> None:
    
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

    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    time_train = time.time() - start
    
    print(f'Train score: {score_train:.4f}')
    print(f'Test score: {score_test:.4f}')
    print(f'Elapsed time: {time_train:.2f}')

    metrics = dict(
        score_train = score_train,
        score_test = score_test,
        time_train = time_train,
        timestamp = datetime.now().strftime(TIMESTAMP_FMT),
    )

    if dump:
        dill.settings['recurse'] = True
        dill.dump(model, open(f'models/pipeline{tag}.joblib', 'wb'))
        json.dump(metrics, open(f'models/metrics{tag}.json', 'w'))


if __name__ == '__main__':
    train()
