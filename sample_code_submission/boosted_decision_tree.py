import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from featureengineering import feature_engineering as fe
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
from sklearn import ensemble
from HiggsML.datasets import train_test_split
import matplotlib.pyplot as plt
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
        self.model.fit(X_train_data, labels, weights, eval_metric="error")

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
    from HiggsML.datasets import BlackSwan_public_dataset as public_dataset

    data=public_dataset()
    data.load_train_set()
    data_set=data.get_train_set()
    data_set['data']=fe(data_set['data'])
    train_set, test_set= train_test_split(data_set, test_size=0.2, random_state=42,reweight=True)
    model=BoostedDecisionTree(data_set,'XGBoost')
    model.fit(train_set['data'],train_set['labels'],train_set['weights'])
    y_pred=model.predict(test_set['data'])
    y_pred_binary=model.predict_binary(test_set['data'])
    print(y_pred[:10])
    roc_score=model.auc_score(test_set['labels'],y_pred,test_set)
    print('roc_auc_score :' ,model.auc_score(test_set['labels'],y_pred,test_set) )
    accuracy=accuracy_score(test_set['labels'],y_pred_binary)
    print(f"accuracy: {accuracy}")
    from sklearn.metrics import roc_curve
    fpr_xgb,tpr_xgb,_ = roc_curve(test_set['labels'],y_pred,sample_weight=test_set['weights'])
    plt.plot(fpr_xgb, tpr_xgb, color='darkgreen',lw=2, label='XGBoost (AUC  = {:.3f})'.format(roc_score))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Background Efficiency')
    plt.ylabel('Signal Efficiency')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig("ROC_comparingnonoptimizedfe.pdf")
    plt.show()


    

