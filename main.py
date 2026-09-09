import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="🌡️ 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.caption("서울의 연평균기온 데이터를 이용해 미래 기온을 예측합니다.")

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜를 날짜 형식으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# --------------------------------------------------
# 연도별 평균기온 계산
# --------------------------------------------------
yearly = (
    df.dropna(subset=["연도", "평균기온"])
      .groupby("연도")
      .agg(
          연평균기온=("평균기온", "mean"),
          관측일수=("날짜", "nunique")
      )
      .reset_index()
)

# 2025년까지 + 관측일 300일 이상인 해만 사용
regression_data = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

regression_data = regression_data.sort_values("연도")

# --------------------------------------------------
# 회귀 계산
# --------------------------------------------------
x = regression_data["연도"].to_numpy()
y = regression_data["연평균기온"].to_numpy()

if len(regression_data) >= 2:
    slope, intercept = np.polyfit(x, y, 1)

    regression_data["예측기온"] = slope * regression_data["연도"] + intercept

    correlation = np.corrcoef(x, y)[0, 1]
else:
    slope = 0
    intercept = 0
    correlation = np.nan


# --------------------------------------------------
# 상단 정보
# --------------------------------------------------
st.subheader("📊 분석에 사용된 데이터")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "사용한 연도 수",
        f"{len(regression_data)}년"
    )

with col2:
    if len(regression_data) > 0:
        st.metric(
            "시작 연도",
            f"{int(regression_data['연도'].min())}년"
        )
    else:
        st.metric("시작 연도", "-")

with col3:
    if len(regression_data) > 0:
        st.metric(
            "끝 연도",
            f"{int(regression_data['연도'].max())}년"
        )
    else:
        st.metric("끝 연도", "-")

with col4:
    if not np.isnan(correlation):
        st.metric(
            "상관계수",
            f"{correlation:.3f}"
        )
    else:
        st.metric("상관계수", "-")

st.info(
    "📌 회귀 직선은 2025년까지의 자료 중 "
    "관측일수가 300일 이상인 연도만 사용해 계산했습니다."
)

# --------------------------------------------------
# 산점도 + 회귀 직선
# --------------------------------------------------
st.subheader("📈 연도와 연평균기온의 관계")

if len(regression_data) >= 2:

    fig = go.Figure()

    # 산점도
    fig.add_trace(
        go.Scatter(
            x=regression_data["연도"],
            y=regression_data["연평균기온"],
            mode="markers",
            name="실제 연평균기온",
            marker=dict(
                size=9,
                opacity=0.75
            ),
            hovertemplate=(
                "<b>%{x}년</b><br>"
                "연평균기온: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

    # 회귀 직선을 위한 x 범위
    line_x = np.linspace(
        regression_data["연도"].min(),
        regression_data["연도"].max(),
        200
    )

    line_y = slope * line_x + intercept

    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="회귀 직선",
            line=dict(width=3),
            hovertemplate=(
                "예측기온: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="연평균기온 (℃)",
        height=550,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # 회귀식
    sign = "+" if intercept >= 0 else "-"

    st.markdown(
        f"""
        ### 🔎 회귀 분석 결과

        **상관계수:** `{correlation:.3f}`

        **회귀식:**  
        `예상 기온 = {slope:.4f} × 연도 {sign} {abs(intercept):.2f}`
        """
    )

else:
    st.error("회귀 분석을 수행하기에 충분한 데이터가 없습니다.")

# --------------------------------------------------
# 연도 선택
# --------------------------------------------------
st.divider()

st.subheader("🔮 미래 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

predicted_temp = slope * selected_year + intercept

# --------------------------------------------------
# 선택한 연도의 예상 기온
# --------------------------------------------------
st.markdown(
    f"""
    <div style="
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        background-color: rgba(128, 128, 128, 0.10);
        margin-top: 20px;
        margin-bottom: 20px;
    ">
        <div style="font-size: 24px;">
            🌡️ {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size: 64px;
            font-weight: bold;
            margin-top: 10px;
        ">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# 선택한 연도를 그래프에 표시
# --------------------------------------------------
if len(regression_data) >= 2:

    fig_prediction = go.Figure()

    # 실제 데이터
    fig_prediction.add_trace(
        go.Scatter(
            x=regression_data["연도"],
            y=regression_data["연평균기온"],
            mode="markers",
            name="실제 기온",
            marker=dict(size=7),
            hovertemplate="%{x}년<br>%{y:.2f}℃<extra></extra>"
        )
    )

    # 회귀 직선 전체
    prediction_x = np.linspace(1900, 2100, 300)
    prediction_y = slope * prediction_x + intercept

    fig_prediction.add_trace(
        go.Scatter(
            x=prediction_x,
            y=prediction_y,
            mode="lines",
            name="회귀 직선",
            line=dict(width=3),
            hovertemplate="%{x:.0f}년<br>%{y:.2f}℃<extra></extra>"
        )
    )

    # 선택한 연도
    fig_prediction.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[predicted_temp],
            mode="markers",
            name=f"{selected_year}년 예상",
            marker=dict(
                size=18,
                symbol="star"
            ),
            hovertemplate=(
                f"<b>{selected_year}년</b><br>"
                f"예상 기온: {predicted_temp:.2f}℃"
                "<extra></extra>"
            )
        )
    )

    fig_prediction.update_layout(
        title=f"🔮 {selected_year}년 기온 예측",
        xaxis_title="연도",
        yaxis_title="연평균기온 (℃)",
        height=500,
        hovermode="closest"
    )

    st.plotly_chart(
        fig_prediction,
        use_container_width=True
    )

# --------------------------------------------------
# 사용한 회귀 데이터 정보
# --------------------------------------------------
st.divider()

st.subheader("📋 회귀 직선을 만드는 데 사용한 연도")

st.write(
    f"총 **{len(regression_data)}개 연도**를 사용했습니다. "
    f"({int(regression_data['연도'].min())}년 ~ "
    f"{int(regression_data['연도'].max())}년)"
)

st.dataframe(
    regression_data[
        ["연도", "연평균기온", "관측일수"]
    ].round(2),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "※ 2025년 이후 자료와 관측일수가 300일 미만인 연도는 "
    "회귀 분석에서 제외했습니다."
)
