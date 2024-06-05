import numpy as np
import warnings
from featureengineering import feature_engineering as fe
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
def roc_curve_plot(model,test_set):
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
