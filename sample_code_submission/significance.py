import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from HiggsML.datasets import train_test_split
import pandas as pd
from HiggsML.datasets import BlackSwan_public_dataset as public_dataset
from feature_engineering import feature_engineering
from xgboost import XGBClassifier
from sklearn import ensemble
import lightgbm as lgb
import time
from IPython.display import display
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
    s_hist, bin_edges = np.histogram(y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1],density=False)
    b_hist, bin_edges = np.histogram(y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0],density=False)
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
def convert_to_numpy_if_needed(data):
    if isinstance(data, pd.DataFrame):
        return data.values
    return data
def learning_curve(train_set,test_set,classifier="XGBoost"):
    train_sizes=np.linspace(0.01,1,25)
    ntrains=[]
    test_aucs=[]
    train_aucs=[]
    times=[]

    x_train, y_train, weights_train= train_set['data'], train_set['labels'], train_set['weights']
    x_test, y_test, weights_test = test_set['data'], test_set['labels'], test_set['weights']


    x_train = convert_to_numpy_if_needed(x_train)
    x_test = convert_to_numpy_if_needed(x_test)


    for train_size in train_sizes:
        if classifier=="XGBoost":
            model = XGBClassifier(learning_rate=0.3409175662018834,max_depth=6,n_estimators=261 )
        elif classifier=="lightgbm":
            model= lgb.LGBMClassifier()
        elif classifier=="sklearnbdt":
            model=ensemble.HistGradientBoostingClassifier()
        ntrain = int(len(x_train) * train_size)
        print("training with ", ntrain, " events")
        ntrains += [ntrain]
        starting_time = time.time()

        # Train using the first ntrain events of the training dataset
        model.fit(x_train[:ntrain, :], y_train[:ntrain], sample_weight=weights_train[:ntrain])

        training_time = time.time() - starting_time
        times += [training_time]

        # Score on the test dataset (always the same)
        y_pred = model.predict_proba(x_test)[:, 1]
        auc_test = roc_auc_score(y_true=y_test, y_score=y_pred, sample_weight=weights_test)
        test_aucs += [auc_test]

        # Score on the train dataset
        y_pred_train = model.predict_proba(x_train[:ntrain, :])[:, 1]
        auc_train = roc_auc_score(y_true=y_train[:ntrain], y_score=y_pred_train, sample_weight=weights_train[:ntrain])
        train_aucs += [auc_train]

    dflearning = pd.DataFrame({
        "Ntraining": ntrains,
        "test_auc": test_aucs,
        "train_auc": train_aucs,
        "time": times
    })
    display(dflearning)

    dflearning.plot.scatter("Ntraining", "test_auc")
    # Focus on the last point
    dflearning[4:].plot.scatter("Ntraining", "test_auc")
    dflearning.plot.scatter("Ntraining", "test_auc")
    # Focus on the last point
    dflearning[4:].plot.scatter("Ntraining", "test_auc")

###################plot curve####################""
    plt.figure(figsize=(10, 6))
    plt.plot(dflearning["Ntraining"], dflearning["train_auc"], label='Train AUC', marker='o')
    plt.plot(dflearning["Ntraining"], dflearning["test_auc"], label='Test AUC', marker='o')
    plt.xlabel("Number of training samples")
    plt.ylabel("AUC")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()
    plt.savefig('learning_curve.png')
    plt.show()
def roc_curve_plot(model,valid_data,y_pred,y_test,test_weights):
    roc_score=model.auc_score(y_test,y_pred,valid_data)
    print('roc_auc_score :' ,model.auc_score(y_test,y_pred,valid_data) )
    fpr,tpr,_ = roc_curve(y_test,y_pred,sample_weight=test_weights)
    return fpr,tpr, roc_score
if __name__=="__main__":
    ##########Load data and add feature engineering###############
    data=public_dataset()
    data.load_train_set()
    data_set=data.get_train_set()
    data_set["data"]=feature_engineering(data_set["data"])
    train_set, test_set= train_test_split(data_set, test_size=0.2, random_state=42,reweight=True)
    learning_curve(train_set, test_set)

