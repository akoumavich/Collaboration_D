import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
import matplotlib.pyplot as plt
from HiggsML.datasets import train_test_split
import pandas as pd
from HiggsML.datasets import BlackSwan_public_dataset as public_dataset
from feature_engineering import feature_engineering


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

def significance_curve(y_true, y_score, sample_weight=None):
    Z = significance_vscore(y_true, y_score,sample_weight)
    print("Z:",Z)
    x = np.linspace(0, 1, num=len(Z))


    plt.plot(x, Z)
    plt.title("BDT Significance")
    plt.xlabel("Threshold")
    plt.ylabel("Significance")
    plt.legend()
    plt.show()

def seperation_curve(y_true, y_score, sample_weight=None,bins=30, classifier="XGBoost"):
    dfall = pd.DataFrame(y_score, columns=["score"])

    fig, ax = plt.subplots(figsize=(10, 6))


    dfall[y_true == 1].hist(
        weights=sample_weight[y_true == 1],
        density=True,
        bins=bins,
        label="signal",
        color='r',
        alpha=0.5,
        ax=ax
    )


    dfall[y_true == 0].hist(
        weights=sample_weight[y_true == 0],
        density=True,
        bins=bins,
        label="background",
        color='b',
        alpha=0.5,
        ax=ax
    )

    # Customization
    ax.set_title(f'Histogram of Scores for {classifier}')
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

    # Display the plot
    return fig ,ax



# data=public_dataset()
# data.load_train_set()
# data_set=data.get_train_set()
# model=BoostedDecisionTree(data_set,'XGBoost')
# train_set, test_set= train_test_split(data_set, test_size=0.2, random_state=42,reweight=True)
# class_weights_train = (
#            train_set['weights'][train_set['labels'] == 0].sum(),
#             train_set['weights'][train_set['labels'] == 1].sum(),
#         )
# train_set['weights'][train_set['labels'] == 0] *= (
#                 max(class_weights_train) / class_weights_train[0])
# train_set['weights'][train_set['labels'] == 1] *= (
#                 max(class_weights_train) / class_weights_train[1])

# model.fit(feature_engineering(train_set['data']),train_set['labels'],train_set['weights'])
# y_pred=model.predict(feature_engineering(test_set['data']))
# print(y_pred)
# fsignificance_score = sklearn.metrics.make_scorer(significance_score)


# dfall=pd.DataFrame(y_pred,columns=["score"])
# ax = dfall[test_set["labels"] == 1].hist(weights= test_set['weights'][test_set["labels"] == 1],density=True,bins=30,label="signal",color='r')
# dfall[test_set["labels"] == 0].hist(weights= test_set['weights'][test_set["labels"] == 0],density=True,bins=30,label="background",color='b')
# if test_set['weights'].size != 0:
#     weights_test_signal = test_set['weights'][test_set["labels"] == 1]
#     weights_test_background = test_set['weights'][test_set["labels"] == 0]
# else:
#     weights_test_signal = None
#     weights_test_background = None
# plt.hist(y_pred[test_set["labels"] == 1],
#                 color='r', alpha=0.5, range=(0,1), bins=30,
#                 histtype='stepfilled', density=True,
#                 label='S (train)', weights=weights_test_signal) # alpha is transparancy
# plt.hist(y_pred[test_set["labels"] == 0],
#                 color='b', alpha=0.5, range=(0,1), bins=30,
#                 histtype='stepfilled', density=True,
#                 label='B (train)', weights=weights_test_background)


# hist, bins = np.histogram(y_pred[test_set["labels"] == 1],
#                                 bins=30, range=(0,1), density=False, weights=weights_test_signal)
# scale = len(y_pred[test_set["labels"] == 1]) / sum(hist)
# err = np.sqrt(hist * scale) / scale

# center = (bins[:-1] + bins[1:]) / 2
# plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

# hist, bins = np.histogram(y_pred[test_set["labels"] == 0],
#                                 bins=30, range=(0,1), density=False, weights=weights_test_background)
# scale = len(y_pred[test_set["labels"] == 0]) / sum(hist)
# err = np.sqrt(hist * scale) / scale

# center = (bins[:-1] + bins[1:]) / 2
# plt.legend(loc='best')
# plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')
    
