import pandas as pd
import my_pipelines as pipe
import sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree

def create_data(data):
    x = data.iloc[:, :-2]
    y = data.iloc[:, -2:]
    
    return x, y

def process_data(x, y):
    X_train, X_test, X_val, y_train, y_test, y_val = pipe.split(x, y)
    X_train, X_test, X_val, y_train, y_test, y_val, std_scalers = pipe.normalize(X_train, X_test, X_val, y_train, y_test, y_val)
    
    return X_train, X_test, X_val, y_train, y_test, y_val, std_scalers


def fit(X, y):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model
    
def predict(model, X):
    return model.predict(X)

def fitdecisiontree(X, y, param):
    
    model = tree.DecisionTreeRegressor(criterion=param['criterion'], 
                                       splitter=param['splitter'],
                                       max_depth=param['max_depth'],
                                       min_samples_split=param['min_samples_split'],
                                       min_samples_leaf=param['min_samples_leaf'],
                                       min_weight_fraction_leaf=param['min_weight_fraction_leaf'])
    model.fit(X, y)
    return model

def score(model, X, y):
    return model.score(X, y)