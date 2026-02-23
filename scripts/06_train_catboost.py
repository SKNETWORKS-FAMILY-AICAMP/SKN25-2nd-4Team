import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# 1. CatBoost 전용 파생변수 생성 함수
def add_catboost_features(df):
    X = df.copy()
    X['HasBalance'] = (X['Balance'] > 0).astype(int)
    X['BalanceSalaryRatio'] = X['Balance'] / (X['EstimatedSalary'] + 1e-6)
    X['Age_Group'] = pd.cut(X['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
    X['Prod_is_1'] = (X['NumOfProducts'] == 1).astype(int)
    X['Prod_is_2'] = (X['NumOfProducts'] == 2).astype(int)
    X['Prod_ge_3'] = (X['NumOfProducts'] >= 3).astype(int)
    X['ZeroBal_Prod1'] = ((X['Balance'] == 0) & (X['NumOfProducts'] == 1)).astype(int)
    X['ZeroBal_Prod2'] = ((X['Balance'] == 0) & (X['NumOfProducts'] == 2)).astype(int)
    X['Prod2_Inactive'] = ((X['NumOfProducts'] == 2) & (X['IsActiveMember'] == 0)).astype(int)
    X['Inactive_Old'] = ((X['IsActiveMember'] == 0) & (X['Age'] >= 45)).astype(int)
    return X

# 2. 데이터 로드 및 전처리 적용
df = pd.read_csv('../data/Customer-Churn-Records.csv')
X_enriched = add_catboost_features(df)

# 타겟과 불필요 변수(Complain 포함) 제거
X = X_enriched.drop(['RowNumber', 'CustomerId', 'Surname', 'Complain', 'Exited'], axis=1)
y = df['Exited']

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. CatBoost 모델 정의 및 학습
cat_features = ['Geography', 'Gender', 'Card Type']

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    scale_pos_weight=4,
    cat_features=cat_features, # 원핫인코딩 대신 사용하는 CatBoost 방식
    random_state=42,
    verbose=100
)

model.fit(X_train, y_train)

# 5. 모델 저장
model.save_model('../models/model_catboost.bin')
print("CatBoost 전용 모델 저장 완료")