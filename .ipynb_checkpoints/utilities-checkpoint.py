from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
from datetime import date
import datetime


def load_and_split_data():
    df = pd.read_csv('train.csv')
    run_eda(df)
    y = df.revenue
    X = df[['budget', 'popularity', 'runtime','release_date']]
    column_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1)
    return (X_train, X_test, y_train, y_test), column_names

def load_test_data():
    df = pd.read_csv('test.csv')
    run_eda(df)
    X = df[['id','budget', 'popularity', 'runtime','release_date']]
    return X
    


def run_eda(input_df):
    df = input_df.copy()
    df['runtime'].fillna((df['runtime'].mean()), inplace=True)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_date'] = datetime.date.today().year
    return df


def test():
    print('Hello from test1')


def cross_val(estimator, X_train, y_train, nfolds):
    mse = cross_val_score(estimator, X_train, y_train,
                          scoring='neg_mean_squared_error',
                          cv=nfolds, n_jobs=-1) * -1
    # mse multiplied by -1 to make positive
    r2 = cross_val_score(estimator, X_train, y_train,
                         scoring='r2', cv=nfolds, n_jobs=-1)
    mean_mse = mse.mean()
    mean_r2 = r2.mean()
    name = estimator.__class__.__name__
    print("{0:<25s} Train CV | MSE: {1:0.3f} | R2: {2:0.3f}".format(name,
                                                                    mean_mse, mean_r2))
    return mean_mse, mean_r2


def stage_score_plot(estimator, X_train, y_train, X_test, y_test):
    estimator.fit(X_train, y_train)
    name = estimator.__class__.__name__.replace('Regressor', '')
    learn_rate = estimator.learning_rate
    # initialize
    train_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    test_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    # Get train score from each boost
    for i, y_train_pred in enumerate(estimator.staged_predict(X_train)):
        train_scores[i] = mean_squared_error(y_train, y_train_pred)
    # Get test score from each boost
    for i, y_test_pred in enumerate(estimator.staged_predict(X_test)):
        test_scores[i] = mean_squared_error(y_test, y_test_pred)
    plt.plot(train_scores, alpha=.5, label="{0} Train - learning rate {1}".format(
        name, learn_rate))
    plt.plot(test_scores, alpha=.5, label="{0} Test  - learning rate {1}".format(
        name, learn_rate), ls='--')
    plt.title(name, fontsize=16, fontweight='bold')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)


def rf_score_plot(randforest, X_train, y_train, X_test, y_test):
    randforest.fit(X_train, y_train)
    y_test_pred = randforest.predict(X_test)
    test_score = mean_squared_error(y_test, y_test_pred)
    plt.axhline(test_score, alpha=0.7, c='y', lw=3, ls='-.', label='Random Forest Test')
