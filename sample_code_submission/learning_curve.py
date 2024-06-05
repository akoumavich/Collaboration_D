import time
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from boosted_decision_tree import BoostedDecisionTree
from scipy import stats
from HiggsML.datasets import train_test_split
from HiggsML.datasets import BlackSwan_public_dataset as public_dataset
from sklearn.preprocessing import StandardScaler
from feature_engineering import feature_engineering
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

##########Load data and add feature engineering###############
data=public_dataset()
data.load_train_set()
data_set=data.get_train_set()
train_set, test_set= train_test_split(data_set, test_size=0.2, random_state=42,reweight=True)



def convert_to_numpy_if_needed(data):
    if isinstance(data, pd.DataFrame):
        return data.values
    return data



#############################Learning curve########################

def learning_curve(train_set,test_set):
    train_sizes=np.linspace(0.01,1,25)
    ntrains=[]
    test_aucs=[]
    train_aucs=[]
    times=[]

    x_train, y_train, weights_train= feature_engineering(train_set['data']), train_set['labels'], train_set['weights']
    x_test, y_test, weights_test = feature_engineering(test_set['data']), test_set['labels'], test_set['weights']


    x_train = convert_to_numpy_if_needed(x_train)
    x_test = convert_to_numpy_if_needed(x_test)
    #xgb = XGBClassifier()
    xgb = XGBClassifier() # simpler GBDT, for illustration

    for train_size in train_sizes:
        ntrain = int(len(x_train) * train_size)
        print("training with ", ntrain, " events")
        ntrains += [ntrain]
        starting_time = time.time()

        # Train using the first ntrain events of the training dataset
        xgb.fit(x_train[:ntrain, :], y_train[:ntrain], sample_weight=weights_train[:ntrain])

        training_time = time.time() - starting_time
        times += [training_time]

        # Score on the test dataset (always the same)
        y_pred_xgb = xgb.predict_proba(x_test)[:, 1]
        auc_test_xgb = roc_auc_score(y_true=y_test, y_score=y_pred_xgb, sample_weight=weights_test)
        test_aucs += [auc_test_xgb]

        # Score on the train dataset
        y_train_xgb = xgb.predict_proba(x_train[:ntrain, :])[:, 1]
        auc_train_xgb = roc_auc_score(y_true=y_train[:ntrain], y_score=y_train_xgb, sample_weight=weights_train[:ntrain])
        train_aucs += [auc_train_xgb]

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

learning_curve(train_set, test_set)
