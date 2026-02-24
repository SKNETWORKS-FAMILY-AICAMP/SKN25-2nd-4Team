import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score

# 1. 데이터 로드 (기존 전처리 결과물 활용)
X_train = joblib.load('X_train_final.pkl')
X_test = joblib.load('X_test_final.pkl')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# 2. 모델 구조 정의
# 뱅가드형 모델링: 깊은 층보다는 안정적인 정규화(Dropout, BN)에 집중
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid') # 이진 분류를 위한 출력층
])

# 3. 모델 컴파일
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name='auc')]
)

# 4. 학습 설정
# 검증 손실(val_loss)이 10회 이상 개선되지 않으면 학습 조기 종료
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 불균형 데이터 대응 (이탈자 1에게 약 4배 가중치 부여)
class_weights = {0: 1, 1: 4}

# 5. 모델 학습
print("ANN 딥러닝 학습 시작...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2, # 8,000개 중 1,600개를 내부 검증용으로 사용
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# 6. 최종 테스트 및 성능 평가
print("\n[Deep Learning (ANN) Evaluation Report]")
print("-" * 50)

# 예측 확률 및 분류값 생성
probs = model.predict(X_test).ravel()
preds = (probs > 0.4).astype(int) # 실무 전략에 따라 임계값 0.4 적용

# 결과 출력
print(f"Final Test AUC: {roc_auc_score(y_test, probs):.4f}")
print("-" * 50)
print(classification_report(y_test, preds))

# 7. 모델 저장 (Keras 형식)
model.save('models/model_ann.h5')
print("\n✅ 모델 저장 완료: models/model_ann.h5")