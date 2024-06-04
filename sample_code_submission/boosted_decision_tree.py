import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn import ensemble
from HiggsML.datasets import train_test_split
classifiers={'XGBoost':XGBClassifier(),'lightgbm':lgb.LGBMClassifier(),'sklearnbdt':ensemble.HistGradientBoostingClassifier()}
class BoostedDecisionTree:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """

    def __init__(self, train_data, classifier):
        self.model = classifiers[classifier]
        self.scaler = StandardScaler()

    def fit(self, train_data, labels, weights=None):
        self.scaler.fit_transform(train_data)
        X_train_data = self.scaler.transform(train_data)
        self.model.fit(X_train_data, labels, weights, eval_metric="accuracy")

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]
    def get_weights(self,data_set):
        return data_set['weights']
    def auc_score(self,y_test,y_pred,data_set):
        return roc_auc_score(y_test,y_pred,self.get_weights(data_set))
from HiggsML.datasets import BlackSwan_public_dataset as public_dataset

data=public_dataset()
data.load_train_set()
data.load_test_set()
data_set=data.get_train_set()
train_set, test_set= train_test_split(data_set, test_size=0.2, random_state=42,reweight=True)
model=BoostedDecisionTree(data_set)
model.fit(train_set['data'],train_set['labels'],train_set['weights'])
y_pred=model.predict(test_set)
print('roc_auc_score :' ,model.auc_score(test_set,y_pred,train_set) )




    

