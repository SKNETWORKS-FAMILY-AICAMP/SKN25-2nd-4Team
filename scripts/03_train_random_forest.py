from sklearn.ensemble import RandomForestClassifier
import joblib, pd
X_train, y_train = joblib.load('X_train_final.pkl'), pd.read_csv('y_train.csv')

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=10, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train.values.ravel())
joblib.dump(model, 'model_rf.pkl')