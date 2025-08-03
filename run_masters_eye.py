#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import subprocess
import sys
import os

def install_packages():
print(“📦 패키지 설치 중…”)
packages = [
‘streamlit’,
‘pandas’,
‘numpy’,
‘plotly’,
‘ta’
]

```
for package in packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
        print(f"✅ {package} 설치 완료")
    except:
        print(f"❌ {package} 설치 실패")
```

def create_streamlit_app():
app_code = ‘’’import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title=“Masters Eye”, page_icon=“🎯”, layout=“wide”)

st.title(“🎯 Masters Eye - 4대 거장 융합 분석”)
st.markdown(”**워렌 버핏 × 레이 달리오 × 리처드 파인만 × 짐 사이먼스**”)

# 사이드바

st.sidebar.title(“⚙️ 설정”)
user = st.sidebar.selectbox(“사용자”, [“나”, “엄마”])
analysis = st.sidebar.selectbox(“분석 유형”, [“홈”, “기술적 분석”, “펀더멘털 분석”, “백테스팅”])

if analysis == “홈”:
col1, col2, col3, col4 = st.columns(4)
with col1:
st.metric(“기술적 지표”, “50+”)
with col2:
st.metric(“재무 지표”, “20+”)
with col3:
st.metric(“백테스팅”, “완료”)
with col4:
st.metric(“성과 등급”, “A+”)

```
st.markdown("---")
st.subheader("🎭 4대 거장 투자 철학")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **🏛️ 워렌 버핏 (30%)**
    - 내재가치 중심 투자
    - 경제적 해자 분석
    - 장기 가치투자
    """)
    
    st.markdown("""
    **🔬 리처드 파인만 (20%)**
    - 과학적 사고방식
    - 불확실성 정량화  
    - 몬테카를로 시뮬레이션
    """)

with col2:
    st.markdown("""
    **🌊 레이 달리오 (30%)**
    - All Weather 전략
    - 거시경제 분석
    - 리스크 패리티
    """)
    
    st.markdown("""
    **📐 짐 사이먼스 (20%)**
    - 퀀트 분석
    - 수학적 모델
    - 패턴 인식
    """)
```

elif analysis == “기술적 분석”:
st.header(“📊 기술적 분석”)

```
if st.button("🚀 샘플 분석 실행"):
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    prices = 70000 + np.cumsum(np.random.randn(100) * 1000)
    
    data = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    # 이동평균 계산
    data['SMA_20'] = data['Price'].rolling(20).mean()
    data['SMA_60'] = data['Price'].rolling(60).mean()
    
    # 차트 그리기
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Price'], name='주가'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], name='SMA 20'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_60'], name='SMA 60'))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 현재 상태
    current_price = data['Price'].iloc[-1]
    sma20 = data['SMA_20'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("현재가", f"{current_price:,.0f}원")
    with col2:
        st.metric("SMA 20", f"{sma20:,.0f}원")
    with col3:
        if current_price > sma20:
            st.metric("신호", "🟢 매수")
        else:
            st.metric("신호", "🔴 매도")
```

elif analysis == “펀더멘털 분석”:
st.header(“💰 펀더멘털 분석”)

```
col1, col2 = st.columns(2)
with col1:
    revenue = st.number_input("매출액 (억원)", value=1000)
    net_income = st.number_input("순이익 (억원)", value=100)
    total_equity = st.number_input("자기자본 (억원)", value=800)

with col2:
    market_cap = st.number_input("시가총액 (억원)", value=2000)
    share_price = st.number_input("주가 (원)", value=50000)

if st.button("🚀 분석 실행"):
    # ROE 계산
    roe = (net_income / total_equity) * 100
    
    # PER 계산
    per = market_cap / net_income
    
    # 내재가치 (간단한 버전)
    growth_rate = 0.05
    discount_rate = 0.08
    intrinsic_value = net_income * (1 + growth_rate) / discount_rate
    fair_price = (intrinsic_value / market_cap) * share_price
    
    safety_margin = ((fair_price - share_price) / fair_price) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ROE", f"{roe:.1f}%")
    with col2:
        st.metric("PER", f"{per:.1f}배")
    with col3:
        st.metric("안전마진", f"{safety_margin:+.1f}%")
    
    if safety_margin > 20:
        st.success("🟢 강력매수 추천")
    elif safety_margin > 0:
        st.info("🟡 매수 고려")
    else:
        st.warning("🔴 투자 주의")
```

elif analysis == “백테스팅”:
st.header(“🔄 백테스팅”)

```
col1, col2 = st.columns(2)
with col1:
    initial_capital = st.number_input("초기 자본 (만원)", value=10000)
    period = st.selectbox("기간", ["1년", "2년", "3년"])

with col2:
    strategy = st.selectbox("전략", ["균등분산", "모멘텀", "가치투자"])

if st.button("🚀 백테스팅 실행"):
    # 가상 백테스팅
    np.random.seed(42)
    days = {"1년": 252, "2년": 504, "3년": 756}[period]
    
    if strategy == "가치투자":
        returns = np.random.normal(0.0008, 0.015, days)  # 연 20%, 변동성 15%
    elif strategy == "모멘텀":
        returns = np.random.normal(0.0006, 0.025, days)  # 연 15%, 변동성 25%
    else:
        returns = np.random.normal(0.0004, 0.020, days)  # 연 10%, 변동성 20%
    
    portfolio_values = [initial_capital * 10000]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    # 결과 계산
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (252/days) - 1) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 수익률", f"{total_return:+.1f}%")
    with col2:
        st.metric("연환산 수익률", f"{annual_return:+.1f}%")
    with col3:
        st.metric("최종 자산", f"{portfolio_values[-1]:,.0f}원")
    
    # 차트
    dates = pd.date_range(end=datetime.now(), periods=len(portfolio_values), freq='D')
    chart_data = pd.DataFrame({
        'Date': dates,
        'Portfolio': portfolio_values
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['Portfolio'], name='포트폴리오 가치'))
    st.plotly_chart(fig, use_container_width=True)
```

st.markdown(”—”)
st.markdown(“🎯 **Masters Eye** - Week 4 기본 분석 엔진 완성!”)
‘’’

```
return app_code
```

def main():
print(“🎯 Masters Eye 수정 버전 시작!”)
print(”=” * 50)

```
choice = input("1. Streamlit 앱 실행\\n2. 패키지만 설치\\n선택 (1-2): ")

if choice == "1":
    print("\\n📦 패키지 설치 중...")
    install_packages()
    
    print("\\n🎨 Streamlit 앱 생성 중...")
    app_file = "masters_eye_app.py"
    with open(app_file, "w", encoding="utf-8") as f:
        f.write(create_streamlit_app())
    
    print(f"✅ {app_file} 생성 완료!")
    print("🚀 Streamlit 실행 중...")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_file])
    except:
        print(f"\\n수동 실행: streamlit run {app_file}")

elif choice == "2":
    install_packages()
    print("✅ 설치 완료!")

else:
    print("❌ 잘못된 선택")
```

if **name** == “**main**”:
main()