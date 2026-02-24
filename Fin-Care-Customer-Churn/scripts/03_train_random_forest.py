from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
X_train, y_train = joblib.load('X_train_final.pkl'), pd.read_csv('y_train.csv')

# [2] 모델 정의 및 학습 
model = RandomForestClassifier(
    n_estimators=200,      # 200개의 의사결정 나무 사용
    class_weight='balanced', # 이탈 고객 데이터에 가중치 부여 (Recall 향상)
    max_depth=10,          # 과적합 방지를 위한 최대 깊이 제한
    min_samples_leaf=5,    # 말단 노드의 최소 샘플 수 제한 (일반화)
    random_state=42
  
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=10, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train.values.ravel())
joblib.dump(model, 'model_rf.pkl')

