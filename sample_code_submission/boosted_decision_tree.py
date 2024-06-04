import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
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
from HiggsML.datasets import BlackSwan_public_dataset as public_dataset

threshholds=np.arange(0.0001,0.0011,0.0001)
roc_auc=[]
accuracy=[]
data=public_dataset()
data.load_train_set()
data_set=data.get_train_set()
train_set, test_set= train_test_split(data_set, test_size=0.2, random_state=42,reweight=True)
for threshhold in threshholds:
    model=BoostedDecisionTree(data_set,'XGBoost')
    model.fit(train_set['data'],train_set['labels'],train_set['weights'],eval_metric=f"error@{threshhold}")
    y_pred=model.predict(test_set['data'])
    y_pred_binary=model.predict_binary(test_set['data'])
    print(y_pred[:10])
    roc_auc.append(model.auc_score(test_set['labels'],y_pred,test_set) )
    accuracy.append(accuracy_score(test_set['labels'],y_pred_binary))



plt.plot(threshholds,accuracy)
plt.title('Accuracy as a function of threshhold')
plt.xlabel('Threshhold')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

###### ajout #####
#fonction de significance:
def amsasimov(s_in,b_in):
        s=np.copy(s_in)
        b=np.copy(b_in)
        s=np.where( (b_in == 0) , 0., s_in)
        b=np.where( (b_in == 0) , 1., b)

        ams = np.sqrt(2*((s+b)*np.log(1+s/b)-s))
        ams=np.where( (s < 0)  | (b < 0), np.nan, ams)
        if np.isscalar(s_in):
            return float(ams)
        else:
            return  ams

def significance_vscore(y_true, y_score, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.full(len(y_true), 1.)
        bins = np.linspace(0, 1., 101)
        s_hist, bin_edges = np.histogram(y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1])
        b_hist, bin_edges = np.histogram(y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0])
        s_cumul = np.cumsum(s_hist[::-1])[::-1]
        b_cumul = np.cumsum(b_hist[::-1])[::-1]
        significance=amsasimov(s_cumul,b_cumul)
        max_value = np.max(significance)
        return significance

def significance_score(y_true, y_score, sample_weight=None):
        max_value = np.max(significance_vscore(y_true, y_score, sample_weight))

        return max_value
fsignificance_score = sklearn.metrics.make_scorer(significance_score)
Z = significance_score(y_true=y_test, y_score=y_pred_skgb)
print (Z)

    

