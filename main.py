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
st.caption("서울의 연평균기온을 이용해 기온 변화 추세와 미래 기온을 예측합니다.")

# --------------------------------------------------
# 데이터
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# --------------------------------------------------
# 연도별 평균기온 + 관측일수
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

# 2025년까지 + 관측일수 300일 이상
regression_data = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

regression_data = regression_data.sort_values("연도").reset_index(drop=True)

# --------------------------------------------------
# 전체 기간 회귀
# --------------------------------------------------
x_all = regression_data["연도"].to_numpy()
y_all = regression_data["연평균기온"].to_numpy()

if len(regression_data) >= 2:
    slope_all, intercept_all = np.polyfit(x_all, y_all, 1)
    correlation = np.corrcoef(x_all, y_all)[0, 1]
else:
    slope_all = 0
    intercept_all = 0
    correlation = np.nan

# 1년당 변화량 → 100년당 변화량
rise_100_all = slope_all * 100

# --------------------------------------------------
# 최근 20년 데이터
# --------------------------------------------------
if len(regression_data) > 0:

    latest_year = int(regression_data["연도"].max())

    recent_20 = regression_data[
        regression_data["연도"] >= latest_year - 19
    ].copy()

else:
    recent_20 = pd.DataFrame()

if len(recent_20) >= 2:

    x_recent = recent_20["연도"].to_numpy()
    y_recent = recent_20["연평균기온"].to_numpy()

    slope_recent, intercept_recent = np.polyfit(
        x_recent,
        y_recent,
        1
    )

    rise_100_recent = slope_recent * 100

else:
    slope_recent = np.nan
    intercept_recent = np.nan
    rise_100_recent = np.nan


# --------------------------------------------------
# 사용 데이터 정보
# --------------------------------------------------
st.subheader("📊 분석에 사용된 데이터")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "사용한 연도 수",
        f"{len(regression_data)}년"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{int(regression_data['연도'].min())}년"
        if len(regression_data) > 0 else "-"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{int(regression_data['연도'].max())}년"
        if len(regression_data) > 0 else "-"
    )

with col4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
        if not np.isnan(correlation) else "-"
    )

st.info(
    "📌 2025년까지의 자료 중 관측일수가 300일 이상인 연도만 "
    "회귀 분석에 사용했습니다."
)

# --------------------------------------------------
# 100년당 기온 상승량 크게 표시
# --------------------------------------------------
st.divider()

st.subheader("🌡️ 100년에 기온이 얼마나 변할까?")

compare1, compare2 = st.columns(2)

with compare1:

    st.markdown("### 🌎 전체 기간")

    if rise_100_all >= 0:
        text_all = f"100년에 +{rise_100_all:.2f}℃"
    else:
        text_all = f"100년에 {rise_100_all:.2f}℃"

    st.markdown(
        f"""
        <div style="
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            background-color: rgba(128, 128, 128, 0.10);
            margin-bottom: 20px;
        ">
            <div style="font-size: 22px;">
                전체 기간의 기온 변화
            </div>
            <div style="
                font-size: 48px;
                font-weight: bold;
                margin-top: 10px;
            ">
                {text_all}
            </div>
            <div style="font-size: 16px; margin-top: 10px;">
                {int(regression_data["연도"].min())}~{int(regression_data["연도"].max())}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with compare2:

    st.markdown("### 🔥 최근 20년")

    if not np.isnan(rise_100_recent):

        if rise_100_recent >= 0:
            text_recent = f"100년에 +{rise_100_recent:.2f}℃"
        else:
            text_recent = f"100년에 {rise_100_recent:.2f}℃"

        recent_start = int(recent_20["연도"].min())
        recent_end = int(recent_20["연도"].max())

        st.markdown(
            f"""
            <div style="
                padding: 30px;
                border-radius: 20px;
                text-align: center;
                background-color: rgba(128, 128, 128, 0.10);
                margin-bottom: 20px;
            ">
                <div style="font-size: 22px;">
                    최근 20년의 기온 변화
                </div>
                <div style="
                    font-size: 48px;
                    font-weight: bold;
                    margin-top: 10px;
                ">
                    {text_recent}
                </div>
                <div style="font-size: 16px; margin-top: 10px;">
                    {recent_start}~{recent_end}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("최근 20년의 회귀 분석에 필요한 데이터가 부족합니다.")

# --------------------------------------------------
# 두 기울기 비교
# --------------------------------------------------
if not np.isnan(rise_100_recent):

    difference = rise_100_recent - rise_100_all

    st.markdown(
        f"""
        ### 🔎 기울기 비교

        전체 기간에 비해 최근 20년의 100년당 기온 변화량은  
        **{abs(difference):.2f}℃ {'더 큰' if difference > 0 else '더 작은'} 값**입니다.
        """
    )

# --------------------------------------------------
# 산점도 + 회귀 직선
# --------------------------------------------------
st.divider()

st.subheader("📈 연도와 연평균기온의 관계")

if len(regression_data) >= 2:

    fig = go.Figure()

    # 전체 데이터 산점도
    fig.add_trace(
        go.Scatter(
            x=regression_data["연도"],
            y=regression_data["연평균기온"],
            mode="markers",
            name="전체 연도",
            marker=dict(
                size=8,
                opacity=0.7
            ),
            hovertemplate=(
                "<b>%{x}년</b><br>"
                "연평균기온: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

    # 최근 20년 데이터 강조
    if len(recent_20) > 0:

        fig.add_trace(
            go.Scatter(
                x=recent_20["연도"],
                y=recent_20["연평균기온"],
                mode="markers",
                name="최근 20년",
                marker=dict(
                    size=11,
                    symbol="circle-open",
                    line=dict(width=2)
                ),
                hovertemplate=(
                    "<b>%{x}년</b><br>"
                    "최근 20년 기온: %{y:.2f}℃"
                    "<extra></extra>"
                )
            )
        )

    # 전체 기간 회귀 직선
    line_x_all = np.linspace(
        regression_data["연도"].min(),
        regression_data["연도"].max(),
        300
    )

    line_y_all = slope_all * line_x_all + intercept_all

    fig.add_trace(
        go.Scatter(
            x=line_x_all,
            y=line_y_all,
            mode="lines",
            name="전체 기간 회귀 직선",
            line=dict(width=4),
            hovertemplate=(
                "전체 추세: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

    # 최근 20년 회귀 직선
    if len(recent_20) >= 2:

        line_x_recent = np.linspace(
            recent_20["연도"].min(),
            recent_20["연도"].max(),
            100
        )

        line_y_recent = (
            slope_recent * line_x_recent
            + intercept_recent
        )

        fig.add_trace(
            go.Scatter(
                x=line_x_recent,
                y=line_y_recent,
                mode="lines",
                name="최근 20년 회귀 직선",
                line=dict(
                    width=4,
                    dash="dash"
                ),
                hovertemplate=(
                    "최근 20년 추세: %{y:.2f}℃"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="연평균기온 (℃)",
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# 미래 기온 예측
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

# 전체 기간 회귀식을 이용한 예측
predicted_temp = (
    slope_all * selected_year
    + intercept_all
)

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
# 예측 그래프
# --------------------------------------------------
fig_prediction = go.Figure()

# 실제 데이터
fig_prediction.add_trace(
    go.Scatter(
        x=regression_data["연도"],
        y=regression_data["연평균기온"],
        mode="markers",
        name="실제 기온",
        marker=dict(size=7),
        hovertemplate=(
            "%{x}년<br>"
            "%{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 전체 회귀 직선
prediction_x = np.linspace(1900, 2100, 300)

prediction_y = (
    slope_all * prediction_x
    + intercept_all
)

fig_prediction.add_trace(
    go.Scatter(
        x=prediction_x,
        y=prediction_y,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "%{x:.0f}년<br>"
            "%{y:.2f}℃"
            "<extra></extra>"
        )
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
# 회귀 데이터
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
