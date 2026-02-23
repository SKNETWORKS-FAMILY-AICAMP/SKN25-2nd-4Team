import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize

# [필수 클래스 및 함수 정의]
class CustomWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=[0.05, 0.05]): self.limits = limits
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            for col in X_copy.columns: X_copy[col] = winsorize(X_copy[col], limits=self.limits)
        else: X_copy = winsorize(X_copy, limits=self.limits)
        return X_copy

def add_custom_features(df):
   def add_custom_features(df):
    X = df.copy()
    
    # 1. 잔고 관련 변수
    X['HasBalance'] = (X['Balance'] > 0).astype(int) # 잔고 유무
    X['BalanceSalaryRatio'] = X['Balance'] / (X['EstimatedSalary'] + 1e-6) # 연봉 대비 잔고 비중
    
    # 2. 연령대 변수 (범주형)
    X['Age_Group'] = pd.cut(X['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # 3. 상품 수 세분화 (CatBoost 외 모델은 원핫인코딩 효과)
    X['Prod_is_1'] = (X['NumOfProducts'] == 1).astype(int)
    X['Prod_is_2'] = (X['NumOfProducts'] == 2).astype(int)
    X['Prod_ge_3'] = (X['NumOfProducts'] >= 3).astype(int)
    
    # 4. 상호작용 변수 (모델이 찾기 힘든 특정 패턴 강조)
    # 잔고가 없으면서 상품이 1~2개인 경우 (이탈 위험군)
    X['ZeroBal_Prod1'] = ((X['Balance'] == 0) & (X['NumOfProducts'] == 1)).astype(int)
    X['ZeroBal_Prod2'] = ((X['Balance'] == 0) & (X['NumOfProducts'] == 2)).astype(int)
    
    # 활동성이 없으면서 상품이 2개인 경우
    X['Prod2_Inactive'] = ((X['NumOfProducts'] == 2) & (X['IsActiveMember'] == 0)).astype(int)
    
    # 활동성이 없는 고연령층
    X['Inactive_Old'] = ((X['IsActiveMember'] == 0) & (X['Age'] >= 45)).astype(int)
    return X

# 1. 데이터 로드 및 분할
df = pd.read_csv('Customer-Churn-Records.csv')
X_raw = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Complain', 'Exited'], axis=1) # Complain 제거
y = df['Exited']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

# 2. 파이프라인 적용
num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Point Earned']
cat_cols = ['Geography', 'Gender', 'Card Type']
pass_cols = ['HasBalance', 'Age_Group', 'IsActiveMember', 'NumOfProducts', 'Prod_is_1', 'ZeroBal_Prod2', 'Inactive_Old']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer()), ('win', CustomWinsorizer()), ('scaler', StandardScaler())]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('pass', 'passthrough', pass_cols)
])

# 3. 데이터 가공 및 저장
X_train_enriched = add_custom_features(X_train_raw)
X_train_final = preprocessor.fit_transform(X_train_enriched)

joblib.dump(X_train_final, 'X_train_final.pkl')
y_train.to_csv('y_train.csv', index=False)
joblib.dump(preprocessor, 'preprocessor.pkl') # 나중에 테스트용 전처리를 위해 저장
print("✅ 전처리 데이터 저장 완료")