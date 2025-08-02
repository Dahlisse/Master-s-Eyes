# backend/app/analysis/performance_metrics.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings(‘ignore’)

@dataclass
class RiskMetrics:
“”“리스크 지표”””
volatility: float = 0.0  # 변동성
var_95: float = 0.0  # 95% VaR
var_99: float = 0.0  # 99% VaR
cvar_95: float = 0.0  # 95% CVaR (Expected Shortfall)
cvar_99: float = 0.0  # 99% CVaR
max_drawdown: float = 0.0  # 최대 낙폭
calmar_ratio: float = 0.0  # 칼마 비율
sterling_ratio: float = 0.0  # 스털링 비율
ulcer_index: float = 0.0  # 궤양 지수
downside_deviation: float = 0.0  # 하방 편차
tracking_error: float = 0.0  # 추적 오차

@dataclass
class ReturnMetrics:
“”“수익률 지표”””
total_return: float = 0.0  # 총 수익률
annualized_return: float = 0.0  # 연환산 수익률
geometric_mean: float = 0.0  # 기하평균 수익률
arithmetic_mean: float = 0.0  # 산술평균 수익률
compound_annual_growth_rate: float = 0.0  # CAGR
excess_return: float = 0.0  # 초과 수익률
active_return: float = 0.0  # 능동 수익률

@dataclass
class RiskAdjustedMetrics:
“”“위험조정 지표”””
sharpe_ratio: float = 0.0  # 샤프 비율
sortino_ratio: float = 0.0  # 소르티노 비율
treynor_ratio: float = 0.0  # 트레이너 비율
information_ratio: float = 0.0  # 정보 비율
modigliani_ratio: float = 0.0  # 모딜리아니 비율
jensen_alpha: float = 0.0  # 젠센 알파
beta: float = 0.0  # 베타
correlation: float = 0.0  # 상관계수

@dataclass
class TradingMetrics:
“”“거래 관련 지표”””
total_trades: int = 0  # 총 거래 수
winning_trades: int = 0  # 승리 거래 수
losing_trades: int = 0  # 패배 거래 수
win_rate: float = 0.0  # 승률
avg_win: float = 0.0  # 평균 승리 금액
avg_loss: float = 0.0  # 평균 손실 금액
profit_factor: float = 0.0  # 수익 팩터
expectancy: float = 0.0  # 기댓값
largest_win: float = 0.0  # 최대 승리
largest_loss: float = 0.0  # 최대 손실
avg_trade_duration: float = 0.0  # 평균 거래 기간
turnover_rate: float = 0.0  # 회전율

class PerformanceAnalyzer:
“”“성과 분석기”””

```
def __init__(self, risk_free_rate: float = 0.025):
    self.risk_free_rate = risk_free_rate

def analyze_returns(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
    """수익률 종합 분석"""
    if returns.empty:
        return self._get_empty_analysis()
    
    # 기본 수익률 지표
    return_metrics = self._calculate_return_metrics(returns)
    
    # 리스크 지표
    risk_metrics = self._calculate_risk_metrics(returns, benchmark_returns)
    
    # 위험조정 지표
    risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(returns, benchmark_returns)
    
    # 분포 분석
    distribution_analysis = self._analyze_return_distribution(returns)
    
    # 시계열 분석
    time_series_analysis = self._analyze_time_series(returns)
    
    return {
        'return_metrics': return_metrics,
        'risk_metrics': risk_metrics,
        'risk_adjusted_metrics': risk_adjusted_metrics,
        'distribution_analysis': distribution_analysis,
        'time_series_analysis': time_series_analysis,
        'summary': self._generate_performance_summary(return_metrics, risk_metrics, risk_adjusted_metrics)
    }

def _calculate_return_metrics(self, returns: pd.Series) -> ReturnMetrics:
    """수익률 지표 계산"""
    try:
        # 기본 통계
        total_periods = len(returns)
        years = total_periods / 252  # 연간 거래일 기준
        
        # 총 수익률
        total_return = (1 + returns).prod() - 1
        
        # 연환산 수익률 (복리)
        if years > 0:
            annualized_return = (1 + total_return) ** (1/years) - 1
            cagr = annualized_return
        else:
            annualized_return = 0
            cagr = 0
        
        # 기하평균과 산술평균
        geometric_mean = (1 + returns).prod() ** (1/len(returns)) - 1 if len(returns) > 0 else 0
        arithmetic_mean = returns.mean()
        
        # 초과 수익률 (무위험수익률 대비)
        daily_risk_free = self.risk_free_rate / 252
        excess_return = returns.mean() - daily_risk_free
        
        return ReturnMetrics(
            total_return=round(total_return * 100, 2),
            annualized_return=round(annualized_return * 100, 2),
            geometric_mean=round(geometric_mean * 100, 4),
            arithmetic_mean=round(arithmetic_mean * 100, 4),
            compound_annual_growth_rate=round(cagr * 100, 2),
            excess_return=round(excess_return * 100, 4)
        )
        
    except Exception as e:
        print(f"수익률 지표 계산 오류: {e}")
        return ReturnMetrics()

def _calculate_risk_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> RiskMetrics:
    """리스크 지표 계산"""
    try:
        # 변동성 (연환산)
        volatility = returns.std() * np.sqrt(252)
        
        # VaR 계산 (히스토리컬 방법)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # CVaR (Expected Shortfall) 계산
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # 최대 낙폭 계산
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # 칼마 비율 (연환산수익률 / 최대낙폭)
        ann_return = ((1 + returns).prod() ** (252/len(returns)) - 1) if len(returns) > 0 else 0
        calmar_ratio = ann_return / max_drawdown if max_drawdown > 0 else 0
        
        # 스털링 비율 (평균 최대낙폭 대비)
        rolling_max_dd = []
        window = min(252, len(returns))  # 1년 또는 전체 기간
        for i in range(window, len(returns)):
            period_returns = returns.iloc[i-window:i]
            period_cum = (1 + period_returns).cumprod()
            period_peak = period_cum.cummax()
            period_dd = ((period_cum - period_peak) / period_peak).min()
            rolling_max_dd.append(abs(period_dd))
        
        avg_max_dd = np.mean(rolling_max_dd) if rolling_max_dd else max_drawdown
        sterling_ratio = ann_return / avg_max_dd if avg_max_dd > 0
```