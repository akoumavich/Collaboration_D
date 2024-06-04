import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from HiggsML.datasets import BlackSwan_public_dataset as public_dataset
data = public_dataset()
data.load_train_set()
train_set=data.get_train_set()
data.load_test_set()
test_set=data.get_test_set()
print(type(test_set))
Classifiers={'xgboost':XGBClassifier()}
class BoostedDecisionTree:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """

    def __init__(self, train_data, classifier):
        self.model = Classifiers[classifier]
        self.scaler = StandardScaler()

    def fit(self, train_data, labels, weights=None):

        self.scaler.fit_transform(train_data)

        X_train_data = self.scaler.transform(train_data)
        self.model.fit(X_train_data, labels, weights, eval_metric="logloss")

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]
