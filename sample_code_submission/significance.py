import numpy as np
from sklearn.preprocessing import StandardScaler
from boosted_decision_tree import BoostedDecisionTree
import sklearn.metrics
import matplotlib.pyplot as plt
from HiggsML.datasets import train_test_split

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
    s_hist, bin_edges = np.histogram(y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1],density=True)
    b_hist, bin_edges = np.histogram(y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0],density=True)
    s_cumul = np.cumsum(s_hist[::-1])[::-1]
    b_cumul = np.cumsum(b_hist[::-1])[::-1]
    significance=amsasimov(s_cumul,b_cumul)
    #max_value = np.max(significance)
    return significance

def significance_score(y_true, y_score, sample_weight=None):
    max_value = np.max(significance_vscore(y_true, y_score, sample_weight))

    return max_value
from HiggsML.datasets import BlackSwan_public_dataset as public_dataset
from feature_engineering import feature_engineering


data=public_dataset()
data.load_train_set()
data_set=data.get_train_set()
model=BoostedDecisionTree(data_set,'XGBoost')
train_set, test_set= train_test_split(data_set, test_size=0.2, random_state=42,reweight=True)
model.fit(feature_engineering(train_set['data']),train_set['labels'],train_set['weights'])
y_pred=model.predict(feature_engineering(test_set['data']))

fsignificance_score = sklearn.metrics.make_scorer(significance_score)
Z = significance_vscore(y_true=test_set["labels"], y_score=y_pred,sample_weight=test_set["weights"])
print("Z:",Z)
x = np.linspace(0, 1, num=len(Z))


plt.plot(x, Z)


plt.title("BDT Significance")
plt.xlabel("Threshold")
plt.ylabel("Significance")
plt.legend()
plt.show()