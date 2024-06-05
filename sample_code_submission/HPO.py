from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from boosted_decision_tree import BoostedDecisionTree
from scipy import stats
from HiggsML.datasets import train_test_split
from HiggsML.datasets import BlackSwan_public_dataset as public_dataset
from sklearn.preprocessing import StandardScaler
from feature_engineering import feature_engineering

# Define global variable to store the best model
best_model = None

param_dist_XGB = {'max_depth': stats.randint(3, 9), # default 6
                'n_estimators': stats.randint(30, 300), #default 100
                'learning_rate': stats.uniform(0.1, 0.5)}
# Function to optimize hyperparameters
def optimize_hyperparameters(train_set, param_dist=param_dist_XGB, cv=5, n_iter=20):
    global best_model
    x_train,y_train,weights_train=feature_engineering(train_set['data']),train_set['labels'],train_set['weights']
    gsearch = RandomizedSearchCV(
        estimator=XGBClassifier(),
        param_distributions=param_dist,
        scoring="roc_auc",
        n_iter=n_iter,
        cv=cv,
    )
    # Ensure weights are passed during fitting
    gsearch.fit(x_train, y_train, sample_weight=weights_train)
    best_model = gsearch  # Store the best model for future use
    print("Best parameters: ", gsearch.best_params_)
    print("Best score (on train dataset CV): ", gsearch.best_score_)

# Function to predict with the optimized model
def predict_with_optimized_model(scaler, test_data, y_test, weights_test):
    global best_model

    if best_model is None:
        raise RuntimeError("Hyperparameter optimization not performed yet.")
    
    y_pred_gs = best_model.predict(test_data)
    
    print("... corresponding score on test dataset AUC: ", roc_auc_score(y_true=y_test, y_score=y_pred_gs, sample_weight=weights_test))
    # print("... corresponding score on test dataset signif: ", significance_score(y_true=y_test, y_score=y_pred_gs, sample_weight=weights_test))


##########Load data and add feature engineering###############
data=public_dataset()
data.load_train_set()
data_set=data.get_train_set()
train_set, test_set= train_test_split(data_set, test_size=0.2, random_state=42,reweight=True)
optimize_hyperparameters(train_set)
predict_with_optimized_model(StandardScaler(),feature_engineering(test_set['data']),test_set['labels'],test_set['weights'])

