# SKN25-2nd-4Team

# 1. 팀 소개


# 2. 프로젝트 기간

# 3. 프로젝트 개요
## 프로젝트명
<b>핀-케어(Fin-Care): 행동 편향 기반 이탈 및 자산 유출 예측</b>

## 프로젝트 배경 및 목적

<h3> Background</h3>

<p>
금융 산업에서 <b>고객 이탈(Churn)</b>은 직접적인 수익 감소로 이어지는 핵심 리스크 요소입니다.
특히 기존 고객을 유지하는 비용은 신규 고객을 유치하는 비용보다 훨씬 낮기 때문에,
이탈을 사전에 감지하고 대응하는 것은 중요한 전략 과제가 됩니다.
</p>

<p>
그러나 대부분의 이탈 예측 모델은 단순히 
<b>"이 고객이 떠날 확률이 얼마인가?"</b>에 집중할 뿐,
<b>"왜 평소와 다른 행동을 보이는가?"</b>에 대한 설명은 부족한 경우가 많습니다.
</p>

<p>
금융 데이터는 지역, 카드 타입, 자산 수준, 거래 활동 등
복합적이고 비선형적인 패턴을 가지며,
단순 통계 지표만으로는 행동의 변곡점을 포착하기 어렵습니다.
</p>

<hr/>

<h3> Objective</h3>

<p>
본 프로젝트의 목적은 단순 예측 모델을 넘어,
<b>의사결정 지원이 가능한 이탈 예측 시스템</b>을 구축하는 것입니다.
</p>

<ul>
  <li>✔ 고객 행동 데이터를 기반으로 이탈 확률 사전 예측</li>
  <li>✔ 행동 패턴의 변화(변곡점)를 포착하는 파생변수 설계</li>
  <li>✔ 개별 고객 단위의 이탈 요인 해석(Local Interpretation) 제공</li>
  <li>✔ 고위험 고객 우선순위 자동화 및 실무 대응 전략 지원</li>
</ul>

<p>
이를 통해 단순히 <b>예측 정확도를 높이는 것</b>이 아니라,
<b>고객 유지 전략 수립에 직접 활용 가능한 시스템</b>을 구현하는 것을 목표로 합니다.
</p>

<hr/>

## 프로젝트 소개
본 프로젝트 **'핀-케어(Fin-Care)'**는 은행 고객의 단순 잔고 변화를 넘어, 행동 편향(Behavioral Bias) 기반의 자산 유출 및 이탈 조기 탐지 시스템 구축을 목표로 합니다.

단순히 "이탈할 것인가?"라는 결과값만 제시하는 기존의 블랙박스 모델에서 탈피하여, **"왜 평소와 다른 패턴으로 금융 행동을 하는가?"**에 대한 데이터 기반의 해답을 제시합니다. 비즈니스 맥락을 정교하게 반영한 패턴 변곡점(Pivot) 파생변수 설계와 해석 가능한 AI(Explainable AI) 기술을 결합하여, 실무자가 즉각적인 마케팅 액션(Action Plan)을 수립할 수 있도록 돕는 최적의 의사결정 지원 도구를 제공합니다.

<h2>기술 및 특 장점 </h2>

<!-- 1. CatBoost -->
<details open>
  <summary><b>1️⃣ Why CatBoost?</b> <span style="color:#888;">(Categorical-friendly model)</span></summary>
  <br/>
  <blockquote>
    범주형 데이터가 많은 금융 고객 데이터에 최적화된 선택
  </blockquote>

  <ul>
    <li><b>지역/성별/카드 타입</b> 등 categorical feature 자동 처리</li>
    <li>별도 인코딩(One-Hot) 없이도 <b>높은 성능 유지</b></li>
    <li>Ordered Boosting 기반으로 <b>과적합 방지에 강점</b></li>
  </ul>

  <p>
    ✅ <b>결론:</b> 금융 고객 데이터 구조에 가장 적합한 알고리즘으로 CatBoost 채택
  </p>
</details>

<br/>

<!-- 2. Feature Engineering -->
<details open>
  <summary><b>2️⃣ Business-Driven Feature Engineering</b> <span style="color:#888;">(Change Point Capture)</span></summary>
  <br/>
  <blockquote>
    단순 통계가 아닌, 고객 행동의 <b>‘변곡점(Change Point)’</b>을 포착하는 파생변수 설계
  </blockquote>

  <h4> Core Derived Features</h4>

  <ul>
    <li>
      <b> Age–Wealth Gap</b><br/>
      <span style="color:#444;">동일 연령대 대비 자산 수준의 괴리도</span><br/>
      <span style="color:#888;">→ 기대 자산 대비 과도하게 낮은 경우 이탈 위험 증가</span>
    </li>
    <br/>
    <li>
      <b> Asset Drain Velocity</b><br/>
      <span style="color:#444;">자산 감소 속도 지표</span><br/>
      <span style="color:#888;">→ 급격한 자산 이탈은 불만/서비스 이동 신호 가능</span>
    </li>
  </ul>

  <h4> Behavioral Dynamics Features</h4>
  <ul>
    <li>거래 패턴 변화율</li>
    <li>신용 사용 강도 변화</li>
    <li>활동성 감소 지표</li>
    <li>최근 활동 대비 과거 활동 편차</li>
  </ul>

  <p>
    ✅ <b>핵심:</b> “왜 평소와 다른 금융 행동을 보이는가?”에 답하기 위한 변수 설계
  </p>
</details>

<br/>

<!-- 3. Threshold -->
<details open>
  <summary><b>3️⃣ Threshold Optimization Strategy</b> <span style="color:#888;">(Recall-first)</span></summary>
  <br/>
  <blockquote>
    정확도보다 중요한 건 <b>고위험 고객을 놓치지 않는 것</b>
  </blockquote>

  <ul>
    <li>기본 Threshold 0.5 대신 <b>0.4로 조정</b></li>
    <li>금융 이탈 예측에서는 <b>False Negative</b>(놓침)가 더 치명적</li>
    <li><b>Recall 중심</b>으로 실무 적용 전략 설계</li>
  </ul>

  <h4>📈 Model Performance</h4>

  <table>
    <thead>
      <tr>
        <th align="left">Metric</th>
        <th align="left">Score</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>AUC</b></td>
        <td><b>0.88</b></td>
      </tr>
      <tr>
        <td><b>Recall</b></td>
        <td><b>80%</b></td>
      </tr>
      <tr>
        <td>Precision</td>
        <td>73%</td>
      </tr>
      <tr>
        <td>Accuracy</td>
        <td>84%</td>
      </tr>
    </tbody>
  </table>

  <p>
    🎯 <b>전략적 선택:</b> 재현율 중심 설계로 고위험 고객 선제 대응 가능
  </p>
</details>

<br/>


<h3>💡 Key Takeaway</h3>

<ul>
  <li>✔ 단순 예측 모델이 아닌 <b>의사결정 지원 모델</b></li>
  <li>✔ 비즈니스 맥락 기반 파생변수로 <b>행동 변곡점</b> 포착</li>
  <li>✔ <b>Threshold 최적화(0.4)</b>로 위험 고객을 놓치지 않는 전략</li>
</ul>

<hr/>

<h2>👥 대상 사용자</h2>

<p>
본 시스템은 단순 데이터 분석용 모델이 아니라,
<b>실무 의사결정에 직접 활용 가능한 도구</b>를 목표로 설계되었습니다.
</p>

<h3>1. CRM / 마케팅 전략 담당자</h3>
<ul>
  <li>이탈 위험 고객을 사전에 식별하여 맞춤형 프로모션 설계</li>
  <li>고위험 고객 우선 관리 전략 수립</li>
  <li>마케팅 비용 대비 효율(ROI) 극대화</li>
</ul>

<h3>2. 데이터 분석가 / 데이터 사이언티스트</h3>
<ul>
  <li>이탈 요인 해석을 통한 인사이트 도출</li>
  <li>행동 변곡점 기반 세그먼트 분석</li>
  <li>추가 모델 고도화 및 실험 설계</li>
</ul>

<h3>3️. 리스크 관리 / 전략 기획 부서</h3>
<ul>
  <li>고객 자산 이탈 패턴 모니터링</li>
  <li>이탈률 감소를 위한 정책 수립</li>
  <li>중장기 고객 유지 전략 설계</li>
</ul>

<h3>4️. 경영진 / 의사결정자</h3>
<ul>
  <li>이탈 위험 지표 기반 KPI 관리</li>
  <li>데이터 기반 전략적 의사결정 지원</li>
  <li>고객 유지율 개선을 통한 수익 안정화</li>
</ul>

<hr/>

<h3>기대효과</h3>

<ul>
  <li>고위험 고객 선제적 관리 가능</li>
  <li>마케팅 비용 효율화</li>
  <li>이탈 원인 기반 맞춤 대응 전략 수립</li>
  <li>데이터 기반 CRM 의사결정 고도화</li>
</ul>



# 5. 수행결과

# 6. 한 줄 회고

# 7. 기술스텍
## 🛠 Tech Stack

### Language & Environment
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![VS Code](https://img.shields.io/badge/VSCode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)

---

###  Data Analysis & Preprocessing
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

---

###  Machine Learning Models
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logo=catboost&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-AA4A44?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)
![Random Forest](https://img.shields.io/badge/RandomForest-228B22?style=for-the-badge&logo=treehouse&logoColor=white)

---

### Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-5A9?style=for-the-badge&logo=python&logoColor=white)

---

###  Deployment (Simulator)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
