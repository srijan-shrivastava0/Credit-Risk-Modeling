
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

def train_logistic(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    model = LGBMClassifier()
    model.fit(X_train, y_train)
    return model
