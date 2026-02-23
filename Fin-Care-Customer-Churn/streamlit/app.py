import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
from catboost import CatBoostClassifier, Pool

st.set_page_config(page_title="Churn Simulator (CatBoost)", page_icon="📉", layout="wide")

import matplotlib as mpl
import matplotlib.font_manager as fm

def set_korean_font():
    # 자주 쓰는 한글 폰트 후보들(맥/윈/리눅스)
    candidates = [
        "AppleGothic", "Apple SD Gothic Neo",
        "NanumGothic", "NanumBarunGothic",
        "Noto Sans CJK KR", "Noto Sans KR",
        "Malgun Gothic"
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.family"] = name
            mpl.rcParams["axes.unicode_minus"] = False
            return name
    # 못 찾으면 기본값으로 두고 경고용 리턴
    mpl.rcParams["axes.unicode_minus"] = False
    return None

chosen = set_korean_font()
if chosen is None:
    print("⚠️ 한글 폰트를 찾지 못했습니다. 후보 폰트를 설치하거나 font.family를 직접 지정하세요.")
else:
    print(f"✅ Using Korean font: {chosen}")
    

# --- 학습과 동일한 파생변수 ---
def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["HasBalance"] = (X["Balance"] > 0).astype(int)
    X["BalanceSalaryRatio"] = X["Balance"] / (X["EstimatedSalary"] + 1e-6)
    X["Age_Group"] = pd.cut(X["Age"], bins=[0, 30, 45, 60, 120], labels=[0, 1, 2, 3]).astype(int)

    X["Prod_is_1"] = (X["NumOfProducts"] == 1).astype(int)
    X["ZeroBal_Prod2"] = ((X["Balance"] == 0) & (X["NumOfProducts"] == 2)).astype(int)
    X["Prod2_Inactive"] = ((X["NumOfProducts"] == 2) & (X["IsActiveMember"] == 0)).astype(int)
    X["Inactive_Old"] = ((X["IsActiveMember"] == 0) & (X["Age"] >= 45)).astype(int)
    return X

@st.cache_resource
def load_artifacts():
    with open("model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = CatBoostClassifier()
    model.load_model("catboost_churn.cbm")
    return model, meta

def slider_num(label, stats, step=None):
    mn, mx, md = stats["min"], stats["max"], stats["median"]
    if step is None:
        step = (mx - mn) / 100 if mx > mn else 1.0
    return st.slider(label, float(mn), float(mx), float(md), float(step))

# 발표용 한글 라벨
KOR_LABEL = {
    "CreditScore": "신용점수",
    "Geography": "지역",
    "Gender": "성별",
    "Age": "나이",
    "Tenure": "거래기간(년)",
    "Balance": "잔고",
    "NumOfProducts": "보유상품수",
    "HasCrCard": "신용카드 보유",
    "IsActiveMember": "활동회원 여부",
    "EstimatedSalary": "추정연봉",
    "Satisfaction Score": "만족도 점수",
    "Card Type": "카드등급",
    "Point Earned": "포인트",
    # 파생변수
    "HasBalance": "잔고 존재 여부(파생)",
    "BalanceSalaryRatio": "잔고/연봉 비율(파생)",
    "Age_Group": "연령대 그룹(파생)",
    "Prod_is_1": "상품 1개 보유(파생)",
    "ZeroBal_Prod2": "잔고0 & 상품2개(파생)",
    "Prod2_Inactive": "상품2개 & 비활동(파생)",
    "Inactive_Old": "45세↑ & 비활동(파생)",
}

def pretty_feat(name: str) -> str:
    return KOR_LABEL.get(name, name)

def predict_proba_one(model, meta, raw_dict: dict):
    """raw_dict(원본 입력) -> 파생변수 -> 모델 입력 정렬 -> prob 반환"""
    FEATURES = meta["feature_names"]
    CAT_FEATURES = meta["cat_features"]

    raw_df = pd.DataFrame([raw_dict])
    feat_df = add_custom_features(raw_df)

    for col in FEATURES:
        if col not in feat_df.columns:
            feat_df[col] = 0

    X_infer = feat_df[FEATURES].copy()
    pool = Pool(X_infer, cat_features=CAT_FEATURES)
    prob = float(model.predict_proba(pool)[0, 1])
    return prob, X_infer, pool

def plot_shap_waterfall(base_value: float, contrib_df: pd.DataFrame, top_n: int = 8):
    """
    CatBoost SHAP 기반 워터폴(예쁜 버전).
    contrib_df columns: feature_kor, shap
    """
    df = contrib_df.copy().head(top_n)
    # 워터폴은 abs가 큰 순서가 자연스러움
    df = df.reindex(df["abs_shap"].sort_values(ascending=False).index)

    labels = df["feature_kor"].tolist()
    vals = df["shap"].tolist()

    # 누적
    cum = [base_value]
    for v in vals:
        cum.append(cum[-1] + v)

    # 막대 시작/끝
    starts = cum[:-1]
    ends = cum[1:]
    widths = [e - s for s, e in zip(starts, ends)]

    # ✅ 발표 화면에서 선명하게 보이도록: 작은 캔버스 + 높은 DPI
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=260)
    y_pos = np.arange(len(labels))[::-1]  # 위에서 아래로
    for i, (lab, s, w) in enumerate(zip(labels[::-1], starts[::-1], widths[::-1])):
        # barh(left=s, width=w)
        ax.barh(y_pos[i], w, left=s, height=0.6)
        ax.text(s + w, y_pos[i], f"{w:+.3f}", va="center", ha="left", fontsize=8)

    ax.axvline(base_value, linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[::-1])
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel("Model log-odds contribution (SHAP)")
    ax.set_title("SHAP Waterfall (Top contributions)", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig, dpi: int = 320) -> bytes:
    """Matplotlib figure -> PNG bytes (Streamlit st.image용)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def risk_message(prob: float, thr: float):
    pct = int(round(prob * 100))
    if prob >= 0.70:
        return "error", f"이 고객은 **{pct}%의 확률로 이탈**할 것으로 예측됩니다. **집중 관리가 필요합니다!**"
    if prob >= thr:
        return "warning", f"이 고객은 **{pct}%의 확률로 이탈**할 가능성이 있습니다. **관리/케어를 권장합니다.**"
    return "success", f"이 고객은 **{pct}%의 확률로 이탈**할 것으로 예측됩니다. 현재는 **안정 구간**입니다."

def risk_badge(prob: float, thr: float):
    """카드 배지용 위험도 라벨/클래스."""
    if prob >= 0.70:
        return "HIGH RISK", "risk-high"
    if prob >= thr:
        return "MEDIUM RISK", "risk-med"
    return "LOW RISK", "risk-low"

model, meta = load_artifacts()
THRESH_DEFAULT = float(meta.get("threshold", 0.40))


st.markdown(
    """
<div class="hero">
  <div class="hero-left">
    <div class="hero-title">📉 은행 가입 고객 이탈 예측 시뮬레이터</div>
    <div class="hero-sub">사용자 입력 → 예측 결과 → SHAP 워터폴(설명)</div>
  </div>
  <div class="hero-badge">CatBoost · SHAP</div>
</div>
""",
    unsafe_allow_html=True,
)


# --- Premium Global UI (폰트/배경/여백/위젯) ---
st.markdown(
    """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

:root{
  --card-bg: rgba(255,255,255,0.95);
  --card-border: rgba(15,23,42,0.10);
  --shadow: 0 12px 32px rgba(2,6,23,0.08);
  --shadow-soft: 0 10px 26px rgba(2,6,23,0.08);
  --radius: 18px;
}

html, body, [class*="css"]{
  font-family: "Pretendard", -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
               "Noto Sans KR", "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* App background (clean) */
.stApp{
  background: linear-gradient(180deg, #F8FAFC 0%, #F3F4F6 100%);
}

/* Layout */
.block-container{
  padding-top: 1.8rem !important;
  padding-bottom: 2.2rem !important;
  max-width: 1200px;
}

/* Clean chrome (발표용) */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Headings */
h1, h2, h3{
  letter-spacing: -0.02em;
}

/* Hero */
.hero{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:16px;
  padding: 18px 18px;
  border-radius: var(--radius);
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  box-shadow: var(--shadow);
  margin: 0 0 16px 0;
  backdrop-filter: blur(10px);
}
.hero-title{
  font-size: 26px;
  font-weight: 900;
  margin: 0;
  line-height: 1.15;
}
.hero-sub{
  margin-top: 8px;
  font-size: 13px;
  font-weight: 700;
  opacity: 0.75;
}
.hero-badge{
  font-size: 12px;
  font-weight: 900;
  padding: 8px 10px;
  border-radius: 999px;
  border: 1px solid rgba(99,102,241,0.25);
  background: rgba(99,102,241,0.08);
  white-space: nowrap;
}

/* Buttons */
.stButton > button, .stDownloadButton > button{
  border-radius: 12px !important;
  padding: 0.65rem 0.95rem !important;
  font-weight: 900 !important;
  border: 1px solid rgba(15,23,42,0.14) !important;
  box-shadow: 0 8px 20px rgba(2,6,23,0.10) !important;
  transition: transform 0.08s ease, box-shadow 0.18s ease, filter 0.18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  transform: translateY(-1px);
  box-shadow: 0 14px 34px rgba(2,6,23,0.12) !important;
  filter: brightness(1.02);
}
div[data-testid="stFormSubmitButton"] button{
  background: linear-gradient(135deg, rgba(99,102,241,1), rgba(16,185,129,1)) !important;
  color: white !important;
  border: none !important;
}

/* Inputs (select / text / number) */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div{
  border-radius: 12px !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  background: rgba(255,255,255,0.92) !important;
  box-shadow: 0 8px 18px rgba(2,6,23,0.06) !important;
}

/* Slider */
div[data-baseweb="slider"]{
  padding-top: 2px;
}
div[data-baseweb="slider"] div[role="slider"]{
  box-shadow: 0 10px 18px rgba(2,6,23,0.14) !important;
}

/* Alerts */
div[data-testid="stAlert"]{
  border-radius: 14px !important;
  border: 1px solid rgba(15,23,42,0.10) !important;
  box-shadow: var(--shadow-soft) !important;
}

/* Horizontal rule */
hr{
  border-top: 1px solid rgba(15,23,42,0.10);
}
</style>
""",
    unsafe_allow_html=True,
)


# --- UI 스타일(발표용 카드) ---
st.markdown(
    """
<style>
/* =========================
   KPI (예측 결과) 카드
   ========================= */
.kpi-card{
  padding: 16px 16px;
  border-radius: 18px;
  border: 1px solid rgba(0,0,0,0.10);
  background: rgba(255,255,255,0.92);
  box-shadow: 0 8px 22px rgba(0,0,0,0.07);

  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;}
.kpi-label{
  font-size: 13px;
  font-weight: 900;
  opacity: 0.75;
  margin-bottom: 8px;
}
.kpi-value{
  font-size: 30px;
  font-weight: 950;
  line-height: 1.05;
}
.kpi-sub{
  margin-top: 8px;
  font-size: 12px;
  font-weight: 900;
  opacity: 0.70;
}


/* =========================
   주요 요인(해석) 패널
   ========================= */
.factor-card{
  padding: 16px 16px;
  border-radius: 18px;
  border: 1px solid rgba(0,0,0,0.10);
  background: rgba(255,255,255,0.92);
  box-shadow: 0 8px 22px rgba(0,0,0,0.07);
}
.factor-header{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:12px;
  margin-bottom: 10px;
}
.factor-kicker{
  font-size: 13px;
  font-weight: 900;
  opacity: 0.80;
  margin: 0;
}
.factor-big{
  font-size: 30px;
  font-weight: 950;
  line-height: 1.0;
  margin-top: 6px;
}
.factor-sub{
  font-size: 12px;
  font-weight: 900;
  opacity: 0.72;
  margin-top: 6px;
}
.factor-section-title{
  margin-top: 14px;
  font-size: 15px;
  font-weight: 950;
}
.factor-list{
  margin: 8px 0 0 0;
  padding: 0;
  list-style: none;
}
.factor-row{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  padding: 10px 10px;
  border-radius: 12px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(250,250,250,0.95);
  margin: 8px 0;
}
.factor-row .left{
  display:flex;
  align-items:center;
  gap:10px;
  min-width: 0;
}
.rank-badge{
  width: 26px;
  height: 26px;
  border-radius: 999px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size: 12px;
  font-weight: 950;
  border: 1px solid rgba(0,0,0,0.14);
  background: rgba(255,255,255,0.98);
  flex: 0 0 auto;
}
.factor-name{
  font-size: 15px;
  font-weight: 900;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: rgba(0,0,0,0.88);
}
.factor-delta{
  font-size: 13px;
  font-weight: 950;
  opacity: 0.85;
  flex: 0 0 auto;
}
.factor-row.up{ border-left: 6px solid rgba(220,38,38,0.55); }
.factor-row.down{ border-left: 6px solid rgba(37,99,235,0.55); }

/* pill */
.pill{
  display:inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 950;
  border: 1px solid rgba(0,0,0,0.14);
  background: rgba(255,255,255,0.92);
  margin-right: 6px;
}

/* 위험도 배지 */
.risk-tag{
  display:inline-block;
  margin-top: 8px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 950;
  border: 1px solid rgba(0,0,0,0.14);
}
.risk-high{ background: rgba(254, 226, 226, 0.95); border-color: rgba(220,38,38,0.25); }
.risk-med{ background: rgba(255, 237, 213, 0.95); border-color: rgba(249,115,22,0.25); }
.risk-low{ background: rgba(220, 252, 231, 0.95); border-color: rgba(22,163,74,0.25); }
</style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# 간단한 화면 전환 (Input -> Result)
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "input"

THR_FIXED = 0.40  # 발표/운영 일관성 목적: 고정

def go_to_result(base_raw: dict, show_shap: bool = True):
    st.session_state.base_raw = base_raw
    st.session_state.show_shap = show_shap
    st.session_state.page = "result"
    st.rerun()

def go_to_input():
    st.session_state.page = "input"
    st.rerun()

# -----------------------------
# 1) 입력 화면
# -----------------------------
if st.session_state.page == "input":
    st.subheader("🎯 사용자 입력 폼")
    num_stats = meta["num_stats"]
    cat_vals = meta.get("cat_values", {})

    with st.form("input_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
            geo = st.selectbox("지역(Geography)", cat_vals.get("Geography", ["France", "Germany", "Spain"]))
            gender = st.selectbox("성별(Gender)", cat_vals.get("Gender", ["Male", "Female"]))
            card_type = st.selectbox("카드등급(Card Type)", cat_vals.get("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"]))

            has_cr = st.checkbox("신용카드 보유(HasCrCard)", value=True)
            is_active = st.checkbox("활동회원(IsActiveMember)", value=True)

        with c2:
            credit = slider_num("신용점수(CreditScore)", num_stats["CreditScore"], step=1.0)
            age = slider_num("나이(Age)", num_stats["Age"], step=1.0)
            tenure = slider_num("거래기간(Tenure)", num_stats["Tenure"], step=1.0)
            balance = slider_num("잔고(Balance)", num_stats["Balance"])
            nprod = slider_num("보유상품수(NumOfProducts)", num_stats["NumOfProducts"], step=1.0)
            salary = slider_num("추정연봉(EstimatedSalary)", num_stats["EstimatedSalary"])
            sat = slider_num("만족도(Satisfaction Score)", num_stats["Satisfaction Score"], step=1.0)
            point = slider_num("포인트(Point Earned)", num_stats["Point Earned"], step=1.0)

        st.markdown("---")
        st.info("임계값(Threshold)을 **0.40**으로 고정했습니다. (발표/운영 일관성 목적)")

        show_shap = st.checkbox("SHAP 워터폴 첨부", value=True)

        submitted = st.form_submit_button("🔮 예측 결과 보기")

        if submitted:
            base_raw = {
                "CreditScore": float(credit),
                "Geography": str(geo),
                "Gender": str(gender),
                "Age": float(age),
                "Tenure": float(tenure),
                "Balance": float(balance),
                "NumOfProducts": float(nprod),
                "HasCrCard": int(has_cr),
                "IsActiveMember": int(is_active),
                "EstimatedSalary": float(salary),
                "Satisfaction Score": float(sat),
                "Card Type": str(card_type),
                "Point Earned": float(point),
            }
            go_to_result(base_raw=base_raw, show_shap=show_shap)

# -----------------------------
# 2) 결과 화면
# -----------------------------
elif st.session_state.page == "result":
    base_raw = st.session_state.get("base_raw")
    show_shap = bool(st.session_state.get("show_shap", True))

    if not base_raw:
        st.warning("입력값이 없습니다. 다시 입력 화면으로 이동합니다.")
        go_to_input()
    

    st.subheader("🎯 예측 결과")
    

    base_prob, base_X, base_pool = predict_proba_one(model, meta, base_raw)

    m1, m2, m3 = st.columns(3, gap="large")

    prob_pct = base_prob * 100
    verdict = "이탈" if base_prob >= THR_FIXED else "유지"
    verdict_icon = "⚠️" if base_prob >= THR_FIXED else "✅"
    verdict_sub = "임계값 이상" if base_prob >= THR_FIXED else "임계값 미만"

    with m1:
        st.markdown(
            f'''
<div class="kpi-card">
  <div class="kpi-label">이탈 확률</div>
  <div class="kpi-value">{prob_pct:.1f}%</div>
  <div class="kpi-sub">모델 예측 확률 (Positive class)</div>
</div>
''',
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f'''
<div class="kpi-card">
  <div class="kpi-label">임계값</div>
  <div class="kpi-value">{THR_FIXED*100:.1f}%</div>
  <div class="kpi-sub">운영 기준 컷오프</div>
</div>
''',
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f'''
<div class="kpi-card">
  <div class="kpi-label">판정</div>
  <div class="kpi-value">{verdict_icon} {verdict}</div>
  <div class="kpi-sub">{verdict_sub}</div>
</div>
''',
            unsafe_allow_html=True,
        )

    level, msg = risk_message(base_prob, THR_FIXED)
    getattr(st, level)(msg)

    if show_shap:
        st.markdown("---")
        st.subheader("🧠 예측 해석")

        try:
            shap_arr = model.get_feature_importance(base_pool, type="ShapValues")
            shap_vals = shap_arr[0, :-1]
            base_val = float(shap_arr[0, -1])

            contrib = pd.DataFrame({
                "feature": meta["feature_names"],
                "shap": shap_vals,
            })
            contrib["abs_shap"] = contrib["shap"].abs()
            contrib["feature_kor"] = contrib["feature"].map(pretty_feat)
            contrib = contrib.sort_values("abs_shap", ascending=False)

            fig = plot_shap_waterfall(base_val, contrib, top_n=7)
            png = fig_to_png_bytes(fig, dpi=600)
            plt.close(fig)

            # ✅ 워터폴(왼쪽) + '주요 요인'(오른쪽) **동일 비율** 배치
            col_left, col_right = st.columns([1, 1], gap="large")

            with col_left:
                st.markdown("#### 📊 SHAP 워터폴: 예측 결과 도출 이유")
                st.image(png, caption="SHAP Waterfall (Top 7)", use_container_width=True)

            with col_right:
                st.markdown("#### ✨ 이탈 확률 주요 요인")
                # ✅ 워터폴 옆 '주요 요인' 카드: 확률 배지 + 컬러 강조 + (이탈↑ 3 / 이탈↓ 3)
                prob_pct = round(float(base_prob) * 100, 1)
                verdict = "이탈(1)" if base_prob >= THR_FIXED else "유지(0)"
                verdict_txt = f"판정: {verdict} · 임계값 {THR_FIXED:.2f}"

                risk_label, risk_class = risk_badge(base_prob, THR_FIXED)

                pos = contrib[contrib["shap"] > 0].nlargest(3, "abs_shap")
                neg = contrib[contrib["shap"] < 0].nlargest(3, "abs_shap")
                up = pos["feature_kor"].tolist()
                down = neg["feature_kor"].tolist()

                html = []
                html.append('<div class="factor-card">')

                # 상단: 판정/위험도 + 큰 확률 숫자
                html.append('<div class="factor-header">')
                html.append('<div>')
                html.append(f'<div class="factor-kicker">{verdict_txt}</div>')
                html.append(f'<div class="risk-tag {risk_class}">{risk_label}</div>')
                html.append('</div>')
                html.append('<div style="text-align:right;">')
                html.append('<div class="factor-kicker">예측 이탈 확률</div>')
                html.append(f'<div class="factor-big">{prob_pct:.1f}%</div>')
                html.append('</div>')
                html.append('</div>')  # header end

                html.append('<div class="factor-sub">SHAP(기여도) 기준 · ↑ Top 3 / ↓ Top 3</div>')

                # ▲ 이탈↑ (양의 SHAP)
                html.append('<div class="factor-section-title"><span class="pill">▲ 이탈↑</span>확률을 올린 요인</div>')
                html.append('<ul class="factor-list">')
                if len(pos) > 0:
                    for i, row in enumerate(pos.itertuples(index=False), start=1):
                        delta = float(getattr(row, "shap"))
                        fname = getattr(row, "feature_kor")
                        html.append(
                            f'<li class="factor-row up">'
                            f'  <div class="left">'
                            f'    <div class="rank-badge">{i}</div>'
                            f'    <div class="factor-name">{fname}</div>'
                            f'  </div>'
                            f'  <div class="factor-delta">+{abs(delta):.3f}</div>'
                            f'</li>'
                        )
                else:
                    html.append('<li class="factor-row up"><div class="factor-name">(해당 없음)</div></li>')
                html.append('</ul>')

                # ▼ 이탈↓ (음의 SHAP)
                html.append('<div class="factor-section-title"><span class="pill">▼ 이탈↓</span>확률을 낮춘 요인</div>')
                html.append('<ul class="factor-list">')
                if len(neg) > 0:
                    for i, row in enumerate(neg.itertuples(index=False), start=1):
                        delta = float(getattr(row, "shap"))
                        fname = getattr(row, "feature_kor")
                        html.append(
                            f'<li class="factor-row down">'
                            f'  <div class="left">'
                            f'    <div class="rank-badge">{i}</div>'
                            f'    <div class="factor-name">{fname}</div>'
                            f'  </div>'
                            f'  <div class="factor-delta">-{abs(delta):.3f}</div>'
                            f'</li>'
                        )
                else:
                    html.append('<li class="factor-row down"><div class="factor-name">(해당 없음)</div></li>')
                html.append('</ul>')

                html.append('</div>')  # card wrapper end
                st.markdown("\n".join(html), unsafe_allow_html=True)
    
 
        except Exception as e:
            st.warning(f"SHAP 워터폴 계산/시각화 중 오류가 발생했어요: {e}")
    st.button("⬅️ 입력으로 돌아가기", on_click=go_to_input)