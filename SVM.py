from sklearn import svm

def fit(X, y):
    model = svm.SVC()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

def score(model, X, y):
    return model.score(X, y)