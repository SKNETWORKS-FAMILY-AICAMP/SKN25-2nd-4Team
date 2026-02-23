from lightgbm import LGBMClassifier
import joblib
import pandas as pd

X_train = joblib.load('X_train_final.pkl')
y_train = pd.read_csv('y_train.csv')

lgbm_params = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': 7,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': 4,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'random_state': 42,
    'verbose': -1
}

model = LGBMClassifier(**lgbm_params)

model.fit(X_train, y_train)
joblib.dump(model, 'model_lgbm.pkl')
