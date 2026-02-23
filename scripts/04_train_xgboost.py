import joblib
import pandas as pd
from xgboost import XGBClassifier

X_train = joblib.load('X_train_final.pkl')
y_train = pd.read_csv('y_train.csv')

xgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 5,
    'gamma': 0.2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 4,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'eval_metric': 'logloss'
}

model = XGBClassifier(**xgb_params)

model.fit(X_train, y_train)
joblib.dump(model, 'model_xgb.pkl')
