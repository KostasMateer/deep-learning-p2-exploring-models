from sklearn.linear_model import LogisticRegression

def fit(X, y, solver):
    model = LogisticRegression(solver=solver)
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

def score(model, X, y):
    return model.score(X, y)