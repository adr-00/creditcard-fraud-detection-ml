import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'validation')]

    if params is None:
        params = params = {
            'objective': 'binary:logistic',  
            'eval_metric': 'aucpr',
            'scale_pos_weight': total_normal / total_fraude,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50
    )
    return model

def evaluate_xgboost(model, X_test, y_test):
    
    dtest = xgb.DMatrix(X_test, label=y_test)

    probabilidades = model.predict(dtest)
    predicciones = [1 if prob >= 0.5 else 0 for prob in probabilidades]

    tn, fp, fn, tp = confusion_matrix(y_test, predicciones).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    aucpr = average_precision_score(y_test, probabilidades)

    return precision, recall, aucpr