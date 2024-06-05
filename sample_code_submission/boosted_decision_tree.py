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
from Score_train_test import compare_train_test
classifiers={'XGBoost':XGBClassifier(learning_rate= 0.4273454179451379, max_depth= 7, n_estimators=240),'lightgbm':lgb.LGBMClassifier(learning_rate= 0.48382410001969056, max_depth= 4,n_estimators=265),'sklearnbdt':ensemble.HistGradientBoostingClassifier(learning_rate= 0.52135689131965, max_depth= 3,max_iter= 115,min_samples_leaf=2)}
pd.set_option('display.max_columns',100)
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
            self.model.fit(X_train_data, labels, weights, eval_metric=eval_metric)
        if self.classifier=="lightgbm":
            self.model.fit(X_train_data, labels, weights)
        if self.classifier=='sklearnbdt':
            self.model.fit(X_train_data, labels, weights)
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
    def significance_vscore(self,y_true, y_score, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.full(len(y_true), 1.)
        bins = np.linspace(0, 1., 101)
        s_hist, bin_edges = np.histogram(y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1],density=True)
        b_hist, bin_edges = np.histogram(y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0],density=True)
        s_cumul = np.cumsum(s_hist[::-1])[::-1]
        b_cumul = np.cumsum(b_hist[::-1])[::-1]
        
    #max_value = np.max(significance)
        s=np.copy(s_cumul)
        b=np.copy(b_cumul)
        s=np.where( (b_cumul == 0) , 0., s)
        b=np.where( (b_cumul == 0) , 1., b)

        ams = np.sqrt(2*((s+b)*np.log(1+s/b)-s))
        ams=np.where( (s < 0)  | (b < 0), np.nan, ams)
        if np.isscalar(s_cumul):
            return float(ams)
        else:
            return  ams
    def significance_curve(self,y_true, y_score, sample_weight=None):
        Z = self.significance_vscore(y_true, y_score,sample_weight)
        x = np.linspace(0, 1, num=len(Z))
        plt.plot(x, Z)
        plt.title("BDT Significance")
        plt.xlabel("Threshold")
        plt.ylabel("Significance")
        plt.legend()
        plt.show()
    
    
if __name__=='__main__':
