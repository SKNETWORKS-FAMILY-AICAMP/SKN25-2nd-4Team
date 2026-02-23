from sklearn.linear_model import LogisticRegression
import joblib, pd
X_train, y_train = joblib.load('X_train_final.pkl'), pd.read_csv('y_train.csv')

model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1, solver='liblinear', random_state=42)
model.fit(X_train, y_train.values.ravel())
joblib.dump(model, 'model_logistic.pkl')
