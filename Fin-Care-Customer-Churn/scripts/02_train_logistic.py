from sklearn.linear_model import LogisticRegression
import joblib, pd
X_train, y_train = joblib.load('X_train_final.pkl'), pd.read_csv('y_train.csv')

# [2] 모델 정의 및 학습 (상세 파라미터 적용)
model = LogisticRegression(
    max_iter=1000, 
    class_weight='balanced', # 이탈/유지 데이터 불균형 자동 조정
    C=0.1,                   # 규제 강도 (작을수록 강함, 과적합 방지)
    solver='liblinear',      # 이진 분류에 안정적인 알고리즘
    random_state=42
)

model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1, solver='liblinear', random_state=42)
model.fit(X_train, y_train.values.ravel())
joblib.dump(model, 'model_logistic.pkl')

