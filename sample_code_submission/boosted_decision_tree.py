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

classifiers={'XGBoost':XGBClassifier(learning_rate= 0.5611836909914183, max_depth= 7, n_estimators=203),'lightgbm':lgb.LGBMClassifier(),'sklearnbdt':ensemble.HistGradientBoostingClassifier()}
class BoostedDecisionTree:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """

    def __init__(self, train_data, classifier="XGBoost"):
        self.model = classifiers[classifier]
        self.scaler = StandardScaler()

    def fit(self, train_data, labels, weights=None,eval_metric="error"):
        self.scaler.fit_transform(train_data)
        X_train_data = self.scaler.transform(train_data)
        self.model.fit(X_train_data, labels, weights, eval_metric=eval_metric)

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
    threshholds=np.arange(0.0001,0.0011,0.0001)
    roc_auc=[]
    accuracy=[]
    precision=[]
    recall=[]
    f1=[]
    data=public_dataset()
    data.load_train_set()
    data_set=data.get_train_set()
    data_set['data']=feature_engineering(data_set['data'])
    train_set, test_set= train_test_split(data_set, test_size=0.2, random_state=42,reweight=True)
    for threshhold in threshholds:
        model=BoostedDecisionTree(data_set,'XGBoost')
        model.fit(train_set['data'],train_set['labels'],train_set['weights'],eval_metric="error")
        y_pred=model.predict(test_set['data'])
        y_pred_binary=(y_pred>=threshhold).astype(int)
        # roc_auc.append(model.auc_score(test_set['labels'],y_pred,test_set) )
        accuracy.append(accuracy_score(test_set['labels'],y_pred_binary))
        precision.append(precision_score(test_set['labels'], y_pred_binary, average='macro'))
        recall.append(recall_score(test_set['labels'], y_pred_binary, average='macro'))
        f1.append(f1_score(test_set['labels'], y_pred_binary, average='macro'))

    plt.plot(threshholds,accuracy,label='accuracy')
    plt.plot(threshholds,precision,label='precision')
    plt.plot(threshholds,recall,label='recall')
    plt.plot(threshholds,f1,label='f1-score')
    plt.title('Different metrics as a function of threshhold')
    plt.xlabel('Threshhold')
    plt.ylabel('Metric')
    plt.legend()
    plt.show()
