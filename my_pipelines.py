import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# import sklearn

def get_data(filename):
    if filename == "car.data":
        names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        data = pd.read_csv(filename, header=None)
        encoded_data, encoders = encode(data, names)
        return encoded_data.iloc[:, :-1], encoded_data.iloc[:, -1:], encoders
    
    if filename == "ENB2012_data.xlsx":
        data = pd.read_excel(filename)
        return data.iloc[:, :-2], data.iloc[:, -2:]
    
    if filename == "wdbc.data":
        data = pd.read_csv(filename, header=None)
        return data.iloc[:, data.columns!=1], data.iloc[:, 1:2]
    
    raise ValueError("Filename not recognized")
    
def encode(data, names):
    """
    Encodes data used for classification model
    
    inputs:  data -- the raw data from a file
             names -- the data's column names
             
    returns: encoded_data -- the encoded classification data in a pandas
             DataFrame
             encoders -- a list of the encoders used, so the encodeding can 
             be reversed
    """
    encoders = []
    encoded_data = pd.DataFrame()
    
    for i in range(data.shape[1]):
        col = data.iloc[:,i]
        le = preprocessing.LabelEncoder()
        le.fit(col)
        encoded_col = le.transform(col)
        
        encoded_data[names[i]] = encoded_col
        encoders.append(le)
        
    return encoded_data, encoders

def normalize(X_train, X_test, X_val, y_train=None, y_test=None, y_val=None):
    """
    normalizes the data
        Parameters:
            X_train (array) -- features to train
            X_test  (array) -- features to test
            y_train (array) -- y to train
            y_test  (array) -- y to test
        Returns:
            X_train (array) -- normalized features to train
            X_test  (array) -- normalized features to test
            y_train (array) -- normalized y to train
            y_test  (array) -- normalized y to test
    """
    
    stdscaler_X_train = preprocessing.StandardScaler()
    stdscaler_X_train = stdscaler_X_train.fit(X_train)
    X_train = stdscaler_X_train.transform(X_train)
    
    stdscaler_X_test = preprocessing.StandardScaler()
    stdscaler_X_test = stdscaler_X_test.fit(X_test)
    X_test = stdscaler_X_test.transform(X_test)

    stdscaler_X_val = preprocessing.StandardScaler()
    stdscaler_X_val = stdscaler_X_val.fit(X_val)
    X_val = stdscaler_X_val.transform(X_val)
    stdscalers_x = [stdscaler_X_train, stdscaler_X_test, stdscaler_X_val]
    
    if y_train is None:
        return X_train, X_test, X_val, stdscalers_x
    
    stdscaler_y_train = preprocessing.StandardScaler()
    stdscaler_y_train = stdscaler_y_train.fit(y_train)
    y_train = stdscaler_y_train.transform(y_train)
    
    stdscaler_y_test = preprocessing.StandardScaler()
    stdscaler_y_test = stdscaler_y_test.fit(y_test)
    y_test = stdscaler_y_test.transform(y_test)
    
    stdscaler_y_val = preprocessing.StandardScaler()
    stdscaler_y_val = stdscaler_y_val.fit(y_val)
    y_val = stdscaler_y_val.transform(y_val)
    
    
    
    stdscalers_y = [stdscaler_y_train, stdscaler_y_test, stdscaler_y_val]

    return X_train, X_test, X_val, y_train, y_test, y_val, [stdscalers_x, stdscalers_y]

def split(x, y):
    """
    splits the data and shuffles
        Parameters:
            x (array) -- features
            y (array) -- targets
        Returns:
            X_train (array) -- features to train
            X_test  (array) -- features to test
            x_val   (array) -- features for validation
            y_train (array) -- y to train
            y_test  (array) -- y to test
            y_val   (array) -- y for validation
    """
    train_ratio = 0.80
    test_ratio = 0.10
    validation_ratio = 0.10
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio), shuffle=False)
    
    return X_train, X_test, X_val, y_train, y_test, y_val