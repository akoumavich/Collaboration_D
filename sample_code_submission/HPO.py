from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy import stats
from HiggsML.datasets import train_test_split
from HiggsML.datasets import BlackSwan_public_dataset as public_dataset
from sklearn.preprocessing import StandardScaler
from feature_engineering import feature_engineering
from xgboost import XGBClassifier
from sklearn import ensemble
# Define global variable to store the best model
best_model = None

param_dist_LGBM = {
    "num_leaves": stats.randint(25, 35),  # default 6
    "n_estimators": stats.randint(30, 300),  # default 100
    "learning_rate": stats.uniform(0.1, 0.5),
    "lambda_l2": stats.uniform(0, 2),
}


# Function to optimize hyperparameters
def optimize_hyperparameters(x_train,y_train,classifier="XGBoost",sample_weights=None, cv=5, n_iter=20):
    global best_model
    if classifier=="XGBoost":
        param_dist = {'max_depth': stats.randint(3, 20), # default 6
                'n_estimators': stats.randint(30, 370), #default 100
                'learning_rate': stats.uniform(0.05, 0.5)
               }
        gsearch = RandomizedSearchCV(
            estimator=XGBClassifier(),
            param_distributions=param_dist,
            scoring="roc_auc",
            n_iter=n_iter,
            cv=cv,
        )
    elif classifier=="lightgbm":
        param_dist = {'num_leaves': stats.randint(20, 40), 
                'n_estimators': stats.randint(30, 300), #default 100
                'learning_rate': stats.uniform(0.05, 0.5),
                'max_depth':stats.randint(3, 15)}# default 6
                 
        gsearch = RandomizedSearchCV(
            estimator=lgb.LGBMClassifier(),
            param_distributions=param_dist,
            scoring="roc_auc",
            n_iter=n_iter,
            cv=cv,
        )
    elif classifier=="sklearnbdt":
        param_dist = {
        'learning_rate': stats.uniform(0.1, 0.5),
        'max_iter': stats.randint(100, 300),
        'max_leaf_nodes': [31, 63, 127],
        'max_depth': [3, 5, 7, 10],
        'min_samples_leaf': [10, 20, 30]}
        gsearch = RandomizedSearchCV(
            estimator=ensemble.HistGradientBoostingClassifier(),
            param_distributions=param_dist,
            scoring="roc_auc",
            n_iter=n_iter,
            cv=cv,
        )
    # Ensure weights are passed during fitting
    gsearch.fit(x_train, y_train, sample_weight=sample_weights)
    best_model = gsearch  # Store the best model for future use
    print("Best parameters: ", gsearch.best_params_)
    print("Best score (on train dataset CV): ", gsearch.best_score_)
    return gsearch.best_params_


# Function to predict with the optimized model
def predict_with_optimized_model(scaler, test_data, y_test, weights_test):
    global best_model

    if best_model is None:
        raise RuntimeError("Hyperparameter optimization not performed yet.")

    y_pred_gs = best_model.predict(test_data)

    print(
        "... corresponding score on test dataset AUC: ",
        roc_auc_score(y_true=y_test, y_score=y_pred_gs, sample_weight=weights_test),
    )


if __name__ == "__main__":
    ##########Load data and add feature engineering###############
    data = public_dataset()
    data.load_train_set()
    data_set = data.get_train_set()
    data_set["data"] = feature_engineering(data_set["data"])
    train_set, test_set = train_test_split(data_set, test_size=0.2, random_state=42, reweight=True)

    optimize_hyperparameters(train_set)
    predict_with_optimized_model(
        StandardScaler(), test_set["data"], test_set["labels"], test_set["weights"]
    )
