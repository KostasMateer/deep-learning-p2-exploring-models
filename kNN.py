from sklearn.neighbors import KNeighborsClassifier

def fit(X, y, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    return model
    
def predict(model, X):
    return model.predict(X)
    
def score(model, X, y):
    return model.score(X, y)