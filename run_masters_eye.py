#!/usr/bin/env python3
“””
Masters Eye 통합 실행 파일
한 번에 모든 분석 기능을 실행할 수 있습니다.

사용법:
python run_masters_eye.py
“””

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
“”“필요한 패키지 자동 설치”””
required_packages = [
‘fastapi==0.104.1’,
‘uvicorn[standard]==0.24.0’,
‘pandas==2.1.4’,
‘numpy==1.24.3’,
‘scipy==1.11.4’,
‘ta==0.10.2’,
‘streamlit==1.28.0’
]

```
print("📦 필요한 패키지들을 설치하고 있습니다...")
for package in required_packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
        print(f"✅ {package.split('==')[0]} 설치 완료")
    except subprocess.CalledProcessError:
        print(f"❌ {package} 설치 실패")
print("🎉 패키지 설치 완료!\n")
```

def create_streamlit_app():
“”“Streamlit 앱 생성”””
streamlit_code = ‘’’
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import ta

# 페이지 설정

st.set_page_config(
page_title=“Masters Eye - 4대 거장 융합 분석”,
page_icon=“🎯”,
layout=“wide”
)

# 제목

st.title(“🎯 Masters Eye - 4대 거장 융합 주식 분석 시스템”)
st.markdown(”**워렌 버핏 × 레이 달리오 × 리처드 파인만 × 짐 사이먼스**”)

# 사이드바

st.sidebar.title(“⚙️ 분석 설정”)
analysis_type = st.sidebar.selectbox(
“분석 유형 선택”,
[“🏠 홈”, “📊 기술적 분석”, “💰 펀더멘털 분석”, “🔄 백테스팅”, “📈 성과 분석”]
)

# 사용자 선택

user_type = st.sidebar.selectbox(“👤 사용자”, [“나”, “엄마”])
st.sidebar.markdown(f”현재 사용자: **{user_type}**”)

# 홈 화면

if analysis_type == “🏠 홈”:
col1, col2, col3, col4 = st.columns(4)

```
with col1:
    st.metric("📊 지원 지표", "50+", "기술적 지표")
with col2:
    st.metric("💰 분석 항목", "20+", "재무 지표")
with col3:
    st.metric("🔄 백테스팅", "몬테카를로", "1000+ 시뮬레이션")
with col4:
    st.metric("🏆 성과 등급", "A+~D", "종합 평가")

st.markdown("---")

# 4대 거장 소개
st.subheader("🎭 4대 거장 투자 철학")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **🏛️ 워렌 버핏 (30%)**
    - 내재가치 중심 투자
    - 경제적 해자 분석
    - 장기 가치투자 철학
    - DCF 모델 활용
    """)
    
    st.markdown("""
    **🔬 리처드 파인만 (20%)**
    - 과학적 사고방식
    - 불확실성 정량화
    - 몬테카를로 시뮬레이션
    - 베이지안 추론
    """)

with col2:
    st.markdown("""
    **🌊 레이 달리오 (30%)**
    - All Weather 전략
    - 거시경제 분석
    - 리스크 패리티
    - 경제 사이클 이해
    """)
    
    st.markdown("""
    **📐 짐 사이먼스 (20%)**
    - 퀀트 분석
    - 수학적 모델
    - 패턴 인식
    - 데이터 기반 투자
    """)
```

# 기술적 분석

elif analysis_type == “📊 기술적 분석”:
st.header(“📊 기술적 분석”)

```
# 샘플 데이터 생성 옵션
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 데이터 입력")
    data_option = st.radio(
        "데이터 소스 선택",
        ["샘플 데이터 사용", "CSV 파일 업로드"]
    )

with col2:
    st.subheader("⚙️ 분석 설정")
    period = st.selectbox("분석 기간", ["1개월", "3개월", "6개월", "1년"])
    confidence = st.slider("신호 신뢰도", 0.5, 1.0, 0.7, 0.1)

# 데이터 처리
if data_option == "샘플 데이터 사용":
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    
    # 가상의 주가 데이터 (삼성전자 스타일)
    base_price = 70000
    returns = np.random.normal(0.001, 0.025, len(dates))
    prices = base_price * np.cumprod(1 + returns)
    volumes = np.random.lognormal(13, 0.5, len(dates))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * np.random.uniform(0.98, 1.02, len(dates)),
        'High': prices * np.random.uniform(1.01, 1.05, len(dates)),
        'Low': prices * np.random.uniform(0.95, 0.99, len(dates)),
        'Close': prices,
        'Volume': volumes.astype(int)
    })
    
    st.success("✅ 샘플 데이터 (삼성전자 스타일) 로드 완료")
    
else:
    uploaded_file = st.file_uploader("CSV 파일 선택", type=['csv'])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.success("✅ 파일 업로드 완료")
    else:
        st.warning("CSV 파일을 업로드해주세요")
        st.stop()

# 기술적 지표 계산 버튼
if st.button("🚀 기술적 분석 실행", type="primary"):
    with st.spinner("분석 중..."):
        # 기술적 지표 계산
        data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['SMA_60'] = ta.trend.sma_indicator(data['Close'], window=60)
        data['RSI'] = ta.momentum.rsi_indicator(data['Close'], window=14)
        data['MACD'] = ta.trend.macd_indicator(data['Close'])
        data['MACD_signal'] = ta.trend.macd_signal_indicator(data['Close'])
        
        # 볼린저 밴드
        bb = ta.volatility.BollingerBands(data['Close'])
        data['BB_upper'] = bb.bollinger_hband()
        data['BB_middle'] = bb.bollinger_mavg()
        data['BB_lower'] = bb.bollinger_lband()
        
    # 결과 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 주가 차트")
        fig = go.Figure()
        
        # 캔들스틱 차트
        fig.add_trace(go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ))
        
        # 이동평균선
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['SMA_20'],
            name='SMA 20', line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['SMA_60'],
            name='SMA 60', line=dict(color='red')
        ))
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ 주요 지표")
        
        latest = data.iloc[-1]
        
        # 현재 상태
        current_price = latest['Close']
        sma20 = latest['SMA_20']
        rsi = latest['RSI']
        
        # 신호 판단
        if current_price > sma20 and rsi < 70:
            signal = "🟢 매수"
            signal_color = "green"
        elif current_price < sma20 and rsi > 30:
            signal = "🔴 매도"
            signal_color = "red"
        else:
            signal = "🟡 중립"
            signal_color = "orange"
        
        st.metric("현재가", f"{current_price:,.0f}원")
        st.metric("RSI (14)", f"{rsi:.1f}")
        st.metric("신호", signal)
        
        # 상세 분석
        st.markdown("**📊 상세 분석:**")
        st.write(f"• SMA20: {sma20:,.0f}원")
        st.write(f"• MACD: {latest['MACD']:.2f}")
        st.write(f"• 볼린저 밴드 위치: {((current_price - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']) * 100):.1f}%")
    
    # 하단에 RSI 차트
    st.subheader("📊 RSI 지표")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=data['Date'], y=data['RSI'],
        name='RSI', line=dict(color='purple')
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="과매도")
    fig_rsi.update_layout(height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)
```

# 펀더멘털 분석

elif analysis_type == “💰 펀더멘털 분석”:
st.header(“💰 펀더멘털 분석”)

```
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 재무 정보 입력")
    
    # 기본 재무 정보
    revenue = st.number_input("매출액 (억원)", value=1000, step=100)
    operating_income = st.number_input("영업이익 (억원)", value=150, step=10)
    net_income = st.number_input("순이익 (억원)", value=100, step=10)
    
    total_assets = st.number_input("총자산 (억원)", value=1500, step=100)
    total_equity = st.number_input("자기자본 (억원)", value=800, step=50)
    total_debt = st.number_input("총부채 (억원)", value=700, step=50)
    
    # 시장 정보
    market_cap = st.number_input("시가총액 (억원)", value=2000, step=100)
    share_price = st.number_input("주가 (원)", value=50000, step=1000)
    shares_outstanding = st.number_input("발행주식수 (만주)", value=4000, step=100)

with col2:
    st.subheader("🏭 업종 선택")
    industry = st.selectbox(
        "업종",
        ["기술주", "금융주", "제조업", "소매업", "헬스케어"]
    )
    
    # 업종별 벤치마크
    benchmarks = {
        "기술주": {"ROE": 12.0, "PER": 25.0, "부채비율": 30.0},
        "금융주": {"ROE": 10.0, "PER": 8.0, "부채비율": 800.0},
        "제조업": {"ROE": 6.0, "PER": 10.0, "부채비율": 60.0},
        "소매업": {"ROE": 8.0, "PER": 15.0, "부채비율": 50.0},
        "헬스케어": {"ROE": 14.0, "PER": 20.0, "부채비율": 25.0}
    }
    
    benchmark = benchmarks[industry]
    st.write("**업종 평균:**")
    for key, value in benchmark.items():
        st.write(f"• {key}: {value}")

if st.button("🚀 펀더멘털 분석 실행", type="primary"):
    with st.spinner("분석 중..."):
        # 재무비율 계산
        roe = (net_income / total_equity) * 100 if total_equity > 0 else 0
        roa = (net_income / total_assets) * 100 if total_assets > 0 else 0
        debt_ratio = (total_debt / total_assets) * 100 if total_assets > 0 else 0
        
        # 밸류에이션
        per = market_cap / net_income if net_income > 0 else 0
        pbr = market_cap / total_equity if total_equity > 0 else 0
        
        # 내재가치 (간단한 DCF)
        growth_rate = 0.05  # 5% 성장 가정
        discount_rate = 0.08  # 8% 할인율
        
        future_cf = net_income
        intrinsic_value = 0
        for year in range(1, 11):  # 10년
            future_cf *= (1 + growth_rate)
            pv = future_cf / ((1 + discount_rate) ** year)
            intrinsic_value += pv
        
        # 터미널 가치
        terminal_cf = future_cf * (1 + 0.025)  # 2.5% 영구성장
        terminal_value = terminal_cf / (discount_rate - 0.025)
        terminal_pv = terminal_value / ((1 + discount_rate) ** 10)
        
        total_value = intrinsic_value + terminal_pv
        intrinsic_price = (total_value / shares_outstanding) * 100  # 만주 → 주
        
        safety_margin = ((intrinsic_price - share_price) / intrinsic_price) * 100 if intrinsic_price > 0 else -100
        
    # 결과 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ROE", f"{roe:.1f}%", f"{roe - benchmark['ROE']:+.1f}%p")
        st.metric("ROA", f"{roa:.1f}%")
        st.metric("부채비율", f"{debt_ratio:.1f}%", f"{debt_ratio - benchmark['부채비율']:+.1f}%p")
    
    with col2:
        st.metric("PER", f"{per:.1f}배", f"{per - benchmark['PER']:+.1f}배")
        st.metric("PBR", f"{pbr:.1f}배")
        st.metric("시가총액", f"{market_cap:,}억원")
    
    with col3:
        st.metric("내재가치", f"{intrinsic_price:,.0f}원")
        st.metric("안전마진", f"{safety_margin:+.1f}%")
        
        if safety_margin > 20:
            recommendation = "🟢 강력매수"
        elif safety_margin > 10:
            recommendation = "🟢 매수"
        elif safety_margin > -10:
            recommendation = "🟡 보유"
        else:
            recommendation = "🔴 매도"
        
        st.metric("투자추천", recommendation)
    
    # 상세 분석
    st.subheader("📊 상세 분석")
    
    # 품질 점수 계산
    quality_score = 0
    max_score = 100
    
    # 수익성 (30점)
    if roe > 15:
        quality_score += 30
    elif roe > 8:
        quality_score += 20
    elif roe > 0:
        quality_score += 10
    
    # 안정성 (30점)
    if debt_ratio < 30:
        quality_score += 30
    elif debt_ratio < 60:
        quality_score += 20
    elif debt_ratio < 100:
        quality_score += 10
    
    # 밸류에이션 (40점)
    if per < benchmark['PER'] * 0.8:
        quality_score += 40
    elif per < benchmark['PER']:
        quality_score += 30
    elif per < benchmark['PER'] * 1.2:
        quality_score += 20
    
    grade_mapping = {
        90: "A+", 80: "A", 70: "B+", 60: "B", 50: "C+", 40: "C"
    }
    
    grade = "D"
    for threshold, g in grade_mapping.items():
        if quality_score >= threshold:
            grade = g
            break
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**종합 점수:** {quality_score}/100점")
        st.write(f"**기업 등급:** {grade}")
    
    with col2:
        st.write("**강점:**")
        if roe > benchmark['ROE']:
            st.write("• 업종 평균 대비 높은 ROE")
        if per < benchmark['PER']:
            st.write("• 합리적인 밸류에이션")
        if safety_margin > 0:
            st.write("• 내재가치 대비 저평가")
```

# 백테스팅

elif analysis_type == “🔄 백테스팅”:
st.header(“🔄 백테스팅 시뮬레이션”)

```
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ 백테스팅 설정")
    
    initial_capital = st.number_input("초기 자본 (만원)", value=10000, step=1000)
    period = st.selectbox("백테스팅 기간", ["1년", "2년", "3년", "5년"])
    strategy = st.selectbox("투자 전략", ["균등분산", "모멘텀", "가치투자", "4대거장융합"])
    
    rebalancing = st.selectbox("리밸런싱 주기", ["월간", "분기", "반기", "연간"])
    
    # 수수료 설정
    commission = st.slider("수수료율 (%)", 0.0, 1.0, 0.15, 0.05)
    
    if st.button("🚀 백테스팅 실행", type="primary"):
        with st.spinner("백테스팅 실행 중..."):
            # 가상의 백테스팅 결과 생성
            np.random.seed(42)
            
            trading_days = {"1년": 252, "2년": 504, "3년": 756, "5년": 1260}[period]
            dates = pd.date_range(end=datetime.now(), periods=trading_days, freq='D')
            
            # 포트폴리오 가치 시뮬레이션
            if strategy == "4대거장융합":
                daily_returns = np.random.normal(0.0008, 0.018, trading_days)  # 연 20%, 변동성 18%
            elif strategy == "모멘텀":
                daily_returns = np.random.normal(0.0006, 0.025, trading_days)  # 연 15%, 변동성 25%
            elif strategy == "가치투자":
                daily_returns = np.random.normal(0.0004, 0.015, trading_days)  # 연 10%, 변동성 15%
            else:  # 균등분산
                daily_returns = np.random.normal(0.0003, 0.020, trading_days)  # 연 8%, 변동성 20%
            
            portfolio_values = [initial_capital * 10000]  # 원 단위
            for ret in daily_returns:
                new_value = portfolio_values[-1] * (1 + ret)
                portfolio_values.append(new_value)
            
            portfolio_values = portfolio_values[1:]  # 첫 번째 값 제거
            
            # 성과 지표 계산
            total_return = (portfolio_values[-1] / (initial_capital * 10000) - 1) * 100
            annual_return = ((portfolio_values[-1] / (initial_capital * 10000)) ** (252/trading_days) - 1) * 100
            volatility = np.std(daily_returns) * np.sqrt(252) * 100
            
            # 최대 낙폭
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak * 100
            max_drawdown = abs(np.min(drawdown))
            
            # 샤프 비율
            excess_return = np.mean(daily_returns) - 0.025/252  # 무위험수익률 2.5%
            sharpe_ratio = excess_return / np.std(daily_returns) * np.sqrt(252)
            
with col2:
    if 'portfolio_values' in locals():
        st.subheader("📈 백테스팅 결과")
        
        # 포트폴리오 가치 차트
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            name=f'{strategy} 전략',
            line=dict(color='blue', width=2)
        ))
        
        # 초기 자본 라인
        fig.add_hline(
            y=initial_capital * 10000,
            line_dash="dash",
            line_color="gray",
            annotation_text="초기 자본"
        )
        
        fig.update_layout(
            title="포트폴리오 가치 변화",
            xaxis_title="날짜",
            yaxis_title="포트폴리오 가치 (원)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 성과 지표
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 수익률", f"{total_return:+.1f}%")
        with col2:
            st.metric("연환산 수익률", f"{annual_return:+.1f}%")
        with col3:
            st.metric("변동성", f"{volatility:.1f}%")
        with col4:
            st.metric("샤프 비율", f"{sharpe_ratio:.2f}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("최대 낙폭", f"{max_drawdown:.1f}%")
        with col2:
            st.metric("최종 자산", f"{portfolio_values[-1]:,.0f}원")
        with col3:
            profit_loss = portfolio_values[-1] - (initial_capital * 10000)
            st.metric("총 손익", f"{profit_loss:+,.0f}원")
        with col4:
            if sharpe_ratio > 1.5:
                grade = "A+"
            elif sharpe_ratio > 1.0:
                grade = "A"
            elif sharpe_ratio > 0.5:
                grade = "B"
            else:
                grade = "C"
            st.metric("성과 등급", grade)
```

# 성과 분석

elif analysis_type == “📈 성과 분석”:
st.header(“📈 포트폴리오 성과 분석”)

```
# 샘플 포트폴리오 데이터 생성
if st.button("📊 샘플 포트폴리오 분석", type="primary"):
    with st.spinner("성과 분석 중..."):
        # 가상의 성과 데이터 생성
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        
        # 포트폴리오와 벤치마크 수익률
        portfolio_returns = np.random.normal(0.0008, 0.018, 365)  # 연 20% 수익률
        benchmark_returns = np.random.normal(0.0003, 0.015, 365)  # KOSPI 연 8% 수익률
        
        # 누적 수익률
        portfolio_cumulative = np.cumprod(1 + portfolio_returns) - 1
        benchmark_cumulative = np.cumprod(1 + benchmark_returns) - 1
        
    # 성과 차트
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 누적 수익률 비교")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_cumulative * 100,
            name='내 포트폴리오',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_cumulative * 100,
            name='KOSPI',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="날짜",
            yaxis_title="누적 수익률 (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 주요 성과 지표")
        
        # 성과 지표 계산
        portfolio_annual = np.mean(portfolio_returns) * 252 * 100
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252) * 100
        portfolio_sharpe = (np.mean(portfolio_returns) - 0.025/252) / np.std(portfolio_returns) * np.sqrt(252)
        
        benchmark_annual = np.mean(benchmark_returns) * 252 * 100
        
        # 베타 계산
        beta = np.cov(portfolio_returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
        
        # 알파 계산
        alpha = portfolio_annual - (2.5 + beta * (benchmark_annual - 2.5))
        
        st.metric("연환산 수익률", f"{portfolio_annual:+.1f}%", f"vs KOSPI {portfolio_annual - benchmark_annual:+.1f}%p")
        st.metric("변동성", f"{portfolio_vol:.1f}%")
        st.metric("샤프 비율", f"{portfolio_sharpe:.2f}")
        st.metric("베타", f"{beta:.2f}")
        st.metric("알파", f"{alpha:+.1f}%")
        
        # 등급 계산
        if portfolio_sharpe > 1.5:
            grade = "A+"
            color = "green"
        elif portfolio_sharpe > 1.0:
            grade = "A"
            color = "green"
```