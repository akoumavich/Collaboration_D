from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from boosted_decision_tree import BoostedDecisionTree


# Define global variable to store the best model
best_model = None

param_dist_XGB = {'max_depth': stats.randint(3, 9), # default 6
                'n_estimators': stats.randint(30, 300), #default 100
                'learning_rate': stats.uniform(0.1, 0.5)}
# Function to optimize hyperparameters
def optimize_hyperparameters(x_train, y_train, weights_train, param_dist, cv=2, n_iter=10):
    
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
    
    if best_model is None:
        raise RuntimeError("Hyperparameter optimization not performed yet.")
    
    test_data = scaler.transform(test_data)
    y_pred_gs = best_model.best_estimator_.predict_proba(test_data)[:, 1]
    
    print("... corresponding score on test dataset AUC: ", roc_auc_score(y_true=y_test, y_score=y_pred_gs, sample_weight=weights_test))
    print("... corresponding score on test dataset signif: ", significance_score(y_true=y_test, y_score=y_pred_gs, sample_weight=weights_test))

# Example usage
# Assuming param_dist_XGB, X_train, y_train, weights_train, scaler, X_test, y_test, and weights_test are defined

# optimize_hyperparameters(X_train, y_train, weights_train, param_dist=param_dist_XGB, cv=2, n_iter=10)
# predict_with_optimized_model(scaler, X_test, y_test, weights_test)