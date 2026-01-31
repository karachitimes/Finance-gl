
from ai.anomaly_engine import detect_anomalies
from ai.classification_engine import train_classifier, predict
from ai.root_cause_engine import correlation_root_cause

class AIOrchestrator:
    def __init__(self):
        self.models = {}

    def run_anomaly(self, df, col):
        return detect_anomalies(df, col)

    def train_classifier(self, name, X, y):
        self.models[name] = train_classifier(X,y)

    def predict(self, name, X):
        return predict(self.models[name], X)

    def root_cause(self, df, target):
        return correlation_root_cause(df, target)
