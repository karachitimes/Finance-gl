
from sklearn.ensemble import RandomForestClassifier

def train_classifier(X, y):
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)
