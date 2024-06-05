import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from sklearn import ensemble
from HiggsML.datasets import train_test_split
import matplotlib.pyplot as plt
from feature_engineering import feature_engineering
from HiggsML.datasets import BlackSwan_public_dataset as public_dataset
classifiers={'XGBoost':XGBClassifier(learning_rate= 0.4273454179451379, max_depth= 7, n_estimators=240),'lightgbm':lgb.LGBMClassifier(learning_rate= 0.48382410001969056, max_depth= 5,n_estimators=265),'sklearnbdt':ensemble.HistGradientBoostingClassifier(learning_rate= 0.52135689131965, max_depth= 3,max_iter= 115,min_samples_leaf=2)}
pd.set_option('display.max_columns',100)
from significance import *
import time
class BoostedDecisionTree:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """

    def __init__(self, train_data, classifier="XGBoost"):
        self.model = classifiers[classifier]
        self.scaler = StandardScaler()
        self.classifier=classifier
    def fit(self, train_data, labels, weights=None,eval_metric="error"):
        self.scaler.fit_transform(train_data)
        X_train_data = self.scaler.transform(train_data)
        if self.classifier=="XGBoost":
            print('fitting XGBoot model')
            start_time=time.time()
            self.model.fit(X_train_data, labels, weights, eval_metric=eval_metric)
            end_time=time.time()
            print(f"XGBoost model fitted in {end_time-start_time} s")
        if self.classifier=="lightgbm":
            print('fitting Lightgbm model')
            start_time=time.time()
            self.model.fit(X_train_data, labels, weights)
            end_time=time.time()
            print(f"Lightgbm model fitted in {end_time-start_time} s")
        if self.classifier=='sklearnbdt':
            print('fitting skgb model')
            start_time=time.time()
            self.model.fit(X_train_data, labels, weights)
            end_time=time.time()
            print(f"skgb model fitted in {end_time-start_time} s")
    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:,1]
    def predict_binary(self,test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data)
    def get_weights(self,data_set):
        return data_set['weights']
    def auc_score(self,y_test,y_pred,data_set):
        return roc_auc_score(y_test,y_pred,sample_weight=self.get_weights(data_set))

    
    
if __name__=='__main__':
    data=public_dataset()
    data.load_train_set()
    data_set=data.get_train_set()
    data_set["data"]=feature_engineering(data_set["data"])
    train_set, test_set= train_test_split(data_set,test_size=0.2, random_state=42,reweight=True)
    training_data, train_weights, y_train=train_set["data"], train_set["weights"], train_set["labels"]
    valid_data, valid_weights, y_test=test_set["data"], test_set["weights"], test_set["labels"]
    class_weights_train = (
           train_weights[y_train == 0].sum(),
            train_weights[y_train == 1].sum(),
        )
    train_weights[y_train == 0] *= (
                max(class_weights_train) / class_weights_train[0])
    train_weights[y_train == 1] *= (
                max(class_weights_train) / class_weights_train[1])
    xgb=BoostedDecisionTree(train_set,classifier="XGBoost")
    lgb_model=BoostedDecisionTree(train_set,classifier="lightgbm")
    skgb=BoostedDecisionTree(train_set,classifier="sklearnbdt")
    xgb.fit(training_data,y_train,weights=train_weights)
    lgb_model.fit(training_data,y_train,weights=train_weights)
    skgb.fit(training_data,y_train,weights=train_weights)
    y_pred_xgb=xgb.predict(valid_data)
    y_pred_lgb=lgb_model.predict(valid_data)
    y_pred_skgb=skgb.predict(valid_data)
    Z_xgb=significance_vscore(y_test,y_pred_xgb,sample_weight=valid_weights)
    Z_lgb=significance_vscore(y_test,y_pred_lgb,sample_weight=valid_weights)
    Z_skgb=significance_vscore(y_test,y_pred_skgb,sample_weight=valid_weights)
    threshold=np.linspace(0,1,num=len(Z_xgb))
    plt.plot(threshold,Z_xgb,label='XGBoost')
    plt.plot(threshold,Z_lgb,label='Lightgbm')
    plt.plot(threshold,Z_skgb,label='SKlearn GBDT')
    plt.xlabel('Threshold')
    plt.ylabel('Significance')
    plt.title('Significance Curve')
    plt.legend()
    plt.savefig("significance_curve_comparison")
    plt.clf()
    fig, ax=seperation_curve(y_test, y_pred_xgb, sample_weight=valid_weights,bins=30,classifier="XGBoost")
    plt.savefig("Histogram of Scores fore XGBoost")
    plt.clf()
    fig, ax=seperation_curve(y_test, y_pred_lgb, sample_weight=valid_weights,bins=30,classifier="Lightgbm")
    plt.savefig("Histogram of Scores fore Lightgbm")
    plt.clf()
    fig, ax=seperation_curve(y_test, y_pred_skgb, sample_weight=valid_weights,bins=30,classifier="SKlearn GBDT")
    plt.savefig("Histogram of Scores fore SKlearn GBDT")
    plt.clf()