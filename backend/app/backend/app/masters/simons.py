“””
짐 사이먼스 퀀트 알고리즘 구현

- 멀티 팩터 모델
- 머신러닝 예측
- 시장 이상현상 탐지
- 통계적 차익거래
- 수학적 엄밀성
  “””

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.orm import Session
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings(‘ignore’)

from app.masters.base import BaseMaster, MasterScore, PortfolioRecommendation
from app.core.logging import get_logger

logger = get_logger(**name**)

@dataclass
class SimonsScore:
“”“사이먼스 평가 점수 데이터 클래스”””
total_score: float
factor_score: float
momentum_score: float
mean_reversion_score: float
anomaly_score: float
ml_prediction_score: float
statistical_significance: float
sharpe_ratio: float
information_ratio: float
max_drawdown: float
win_rate: float
expected_alpha: float

```
def to_dict(self) -> Dict:
    return {
        'total_score': self.total_score,
        'factor_score': self.factor_score,
        'momentum_score': self.momentum_score,
        'mean_reversion_score': self.mean_reversion_score,
        'anomaly_score': self.anomaly_score,
        'ml_prediction_score': self.ml_prediction_score,
        'statistical_significance': self.statistical_significance,
        'sharpe_ratio': self.sharpe_ratio,
        'information_ratio': self.information_ratio,
        'max_drawdown': self.max_drawdown,
        'win_rate': self.win_rate,
        'expected_alpha': self.expected_alpha
    }
```

@dataclass
class FactorScores:
“”“팩터별 점수”””
value: float
growth: float
quality: float
momentum: float
low_volatility: float
size: float
profitability: float

```
def to_dict(self) -> Dict:
    return {
        'value': self.value,
        'growth': self.growth,
        'quality': self.quality,
        'momentum': self.momentum,
        'low_volatility': self.low_volatility,
        'size': self.size,
        'profitability': self.profitability
    }
```

class SimonsQuantInvestor(BaseMaster):
“””
짐 사이먼스의 퀀트 투자를 구현한 클래스

```
핵심 원칙:
1. 수학적 엄밀성 - 통계적 유의성 확보
2. 시장 비효율성 발굴 - 패턴과 이상현상 탐지
3. 리스크 관리 - 포지션 사이징과 다양화
4. 지속적 학습 - 모델의 적응과 진화
"""

def __init__(self, db: Session):
    super().__init__(db, "Jim Simons")
    self.lookback_period = 252  # 1년
    self.min_observations = 60  # 최소 관측치
    self.significance_level = 0.01  # 1% 유의수준 (엄격)
    self.rebalance_frequency = 21  # 월간 리밸런싱
    
    # 머신러닝 모델 설정
    self.models = {
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1)
    }
    
    self.scaler = RobustScaler()  # 이상치에 강건한 스케일러
    
async def evaluate_stock(self, ticker: str) -> Optional[SimonsScore]:
    """
    사이먼스 스타일 퀀트 평가
    
    Args:
        ticker: 종목 코드
        
    Returns:
        SimonsScore 객체 또는 None
    """
    try:
        if not await self.validate_ticker(ticker):
            return None
        
        logger.info(f"Starting Simons quant evaluation for {ticker}")
        
        # 1. 멀티 팩터 분석
        factor_scores = await self._multi_factor_analysis(ticker)
        
        # 2. 모멘텀 vs 평균회귀 분석
        momentum_score = await self._momentum_analysis(ticker)
        mean_reversion_score = await self._mean_reversion_analysis(ticker)
        
        # 3. 시장 이상현상 탐지
        anomaly_score = await self._detect_market_anomalies(ticker)
        
        # 4. 머신러닝 예측
        ml_prediction = await self._machine_learning_prediction(ticker)
        
        # 5. 통계적 유의성 검증
        statistical_significance = await self._statistical_significance_test(ticker)
        
        # 6. 리스크 조정 성과 지표
        risk_metrics = await self._calculate_risk_metrics(ticker)
        
        # 7. 알파 추정
        expected_alpha = await self._estimate_alpha(ticker, factor_scores)
        
        # 8. 종합 점수 계산
        total_score = self._calculate_simons_total_score(
            factor_scores, momentum_score, mean_reversion_score,
            anomaly_score, ml_prediction, statistical_significance, risk_metrics
        )
        
        return SimonsScore(
            total_score=total_score,
            factor_score=np.mean(list(factor_scores.to_dict().values())),
            momentum_score=momentum_score,
            mean_reversion_score=mean_reversion_score,
            anomaly_score=anomaly_score,
            ml_prediction_score=ml_prediction['score'],
            statistical_significance=statistical_significance,
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            information_ratio=risk_metrics['information_ratio'],
            max_drawdown=risk_metrics['max_drawdown'],
            win_rate=risk_metrics['win_rate'],
            expected_alpha=expected_alpha
        )
        
    except Exception as e:
        logger.error(f"Error in Simons evaluation for {ticker}: {str(e)}")
        return None

async def _multi_factor_analysis(self, ticker: str) -> FactorScores:
    """
    멀티 팩터 모델 분석
    
    7가지 주요 팩터:
    1. Value (가치)
    2. Growth (성장)
    3. Quality (품질)
    4. Momentum (모멘텀)
    5. Low Volatility (저변동성)
    6. Size (규모)
    7. Profitability (수익성)
    """
    try:
        # 기본 데이터 수집
        financial_data = await self._get_financial_data(ticker)
        price_data = await self._get_price_data(ticker)
        
        # 1. Value Factor
        value_score = await self._calculate_value_factor(financial_data, price_data)
        
        # 2. Growth Factor
        growth_score = await self._calculate_growth_factor(financial_data)
        
        # 3. Quality Factor
        quality_score = await self._calculate_quality_factor(financial_data)
        
        # 4. Momentum Factor
        momentum_score = await self._calculate_momentum_factor(price_data)
        
        # 5. Low Volatility Factor
        volatility_score = await self._calculate_volatility_factor(price_data)
        
        # 6. Size Factor
        size_score = await self._calculate_size_factor(financial_data)
        
        # 7. Profitability Factor
        profitability_score = await self._calculate_profitability_factor(financial_data)
        
        return FactorScores(
            value=value_score,
            growth=growth_score,
            quality=quality_score,
            momentum=momentum_score,
            low_volatility=volatility_score,
            size=size_score,
            profitability=profitability_score
        )
        
    except Exception as e:
        logger.warning(f"Multi-factor analysis failed for {ticker}: {str(e)}")
        return FactorScores(50, 50, 50, 50, 50, 50, 50)

async def _calculate_value_factor(self, financial_data: Dict, price_data: Dict) -> float:
    """가치 팩터 계산"""
    try:
        # 여러 밸류에이션 지표 결합
        pe_ratio = financial_data.get('pe_ratio', 15)
        pb_ratio = financial_data.get('pb_ratio', 1.5)
        ev_ebitda = financial_data.get('ev_ebitda', 10)
        
        # 업종 대비 상대 밸류에이션
        sector_pe_avg = await self._get_sector_average_pe(financial_data.get('sector'))
        sector_pb_avg = await self._get_sector_average_pb(financial_data.get('sector'))
        
        # Z-score 계산 (낮을수록 저평가)
        pe_zscore = -((pe_ratio - sector_pe_avg) / (sector_pe_avg * 0.3))
        pb_zscore = -((pb_ratio - sector_pb_avg) / (sector_pb_avg * 0.3))
        ev_ebitda_zscore = -(ev_ebitda - 12) / 5  # 기준값 12, 표준편차 5
        
        # 복합 가치 점수
        value_composite = (pe_zscore + pb_zscore + ev_ebitda_zscore) / 3
        
        # 0-100 점수로 변환
        value_score = max(0, min(100, 50 + value_composite * 20))
        
        return value_score
        
    except Exception as e:
        logger.warning(f"Value factor calculation failed: {str(e)}")
        return 50.0

async def _calculate_growth_factor(self, financial_data: Dict) -> float:
    """성장 팩터 계산"""
    try:
        # 성장률 지표들
        revenue_growth = financial_data.get('revenue_growth_3y', 0.05)
        earnings_growth = financial_data.get('earnings_growth_3y', 0.05)
        book_value_growth = financial_data.get('bv_growth_3y', 0.05)
        
        # 성장률 가속도 (최근 성장률 vs 과거 성장률)
        recent_growth = financial_data.get('revenue_growth_1y', 0.05)
        historical_growth = financial_data.get('revenue_growth_5y', 0.05)
        growth_acceleration = recent_growth - historical_growth
        
        # 성장 지속성 (성장률 변동성이 낮을수록 좋음)
        growth_consistency = 1 / (1 + financial_data.get('growth_volatility', 0.1))
        
        # 복합 성장 점수
        growth_components = {
            'revenue_growth': min(revenue_growth * 10, 1),  # 10% 성장시 만점
            'earnings_growth': min(earnings_growth * 8, 1),
            'bv_growth': min(book_value_growth * 12, 1),
            'acceleration': max(-1, min(1, growth_acceleration * 20)),
            'consistency': growth_consistency
        }
        
        weights = [0.3, 0.3, 0.15, 0.15, 0.1]
        growth_score = sum(score * weight for score, weight in zip(growth_components.values(), weights))
        
        return max(0, min(100, growth_score * 100))
        
    except Exception as e:
        logger.warning(f"Growth factor calculation failed: {str(e)}")
        return 50.0

async def _calculate_quality_factor(self, financial_data: Dict) -> float:
    """품질 팩터 계산"""
    try:
        # 수익성 지표
        roe = financial_data.get('roe', 0.1)
        roa = financial_data.get('roa', 0.05)
        roic = financial_data.get('roic', 0.08)
        
        # 재무 안정성
        debt_to_equity = financial_data.get('debt_to_equity', 0.5)
        current_ratio = financial_data.get('current_ratio', 1.5)
        interest_coverage = financial_data.get('interest_coverage', 5)
        
        # 수익 품질
        accruals_ratio = financial_data.get('accruals_ratio', 0)  # 낮을수록 좋음
        cash_flow_to_earnings = financial_data.get('cf_to_earnings', 1)
        
        # 경영 효율성
        asset_turnover = financial_data.get('asset_turnover', 1)
        inventory_turnover = financial_data.get('inventory_turnover', 6)
        
        # 품질 점수 계산
        profitability_score = min(100, (roe * 5 + roa * 10 + roic * 6) * 100 / 3)
        stability_score = min(100, (
            (1 / (1 + debt_to_equity)) * 40 +
            min(current_ratio / 2, 1) * 30 +
            min(interest_coverage / 10, 1) * 30
        ))
        earnings_quality_score = min(100, (
            (1 - abs(accruals_ratio)) * 50 +
            min(cash_flow_to_earnings, 1) * 50
        ))
        efficiency_score = min(100, (asset_turnover + inventory_turnover / 6) * 50)
        
        # 가중 평균
        quality_score = (
            profitability_score * 0.4 +
            stability_score * 0.3 +
            earnings_quality_score * 0.2 +
            efficiency_score * 0.1
        )
        
        return quality_score
        
    except Exception as e:
        logger.warning(f"Quality factor calculation failed: {str(e)}")
        return 50.0

async def _momentum_analysis(self, ticker: str) -> float:
    """모멘텀 분석"""
    try:
        price_data = await self._get_price_data(ticker)
        returns = price_data['returns']
        
        if len(returns) < self.min_observations:
            return 50.0
        
        # 다양한 기간의 모멘텀
        momentum_1m = np.mean(returns[-21:]) if len(returns) >= 21 else 0
        momentum_3m = np.mean(returns[-63:]) if len(returns) >= 63 else 0
        momentum_6m = np.mean(returns[-126:]) if len(returns) >= 126 else 0
        momentum_12m = np.mean(returns[-252:]) if len(returns) >= 252 else 0
        
        # 최근 모멘텀 (지난 1개월 제외)
        if len(returns) >= 42:
            recent_momentum = np.mean(returns[-42:-21])
        else:
            recent_momentum = 0
        
        # 모멘텀 강도 계산
        momentum_scores = []
        for period_momentum in [momentum_1m, momentum_3m, momentum_6m, momentum_12m, recent_momentum]:
            # 연환산 수익률로 변환
            annualized_momentum = period_momentum * 252
            momentum_score = 50 + annualized_momentum * 100  # 10% 수익률 = 60점
            momentum_scores.append(max(0, min(100, momentum_score)))
        
        # 가중 평균 (최근 기간에 더 높은 가중치)
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        final_momentum_score = sum(score * weight for score, weight in zip(momentum_scores, weights))
        
        return final_momentum_score
        
    except Exception as e:
        logger.warning(f"Momentum analysis failed for {ticker}: {str(e)}")
        return 50.0

async def _mean_reversion_analysis(self, ticker: str) -> float:
    """평균회귀 분석"""
    try:
        price_data = await self._get_price_data(ticker)
        prices = price_data['prices']
        
        if len(prices) < self.min_observations:
            return 50.0
        
        # 이동평균선 대비 괴리율
        ma_20 = np.mean(prices[-20:])
        ma_60 = np.mean(prices[-60:]) if len(prices) >= 60 else ma_20
        ma_120 = np.mean(prices[-120:]) if len(prices) >= 120 else ma_60
        
        current_price = prices[-1]
        
        # 각 이동평균 대비 괴리율
        deviation_20 = (current_price - ma_20) / ma_20
        deviation_60 = (current_price - ma_60) / ma_60
        deviation_120 = (current_price - ma_120) / ma_120
        
        # RSI 계산
        rsi = self._calculate_rsi(prices)
        
        # 볼린저 밴드 위치
        bb_position = self._calculate_bollinger_position(prices)
        
        # 평균회귀 신호 강도
        mean_reversion_signals = []
        
        # 과매도 신호 (평균회귀 기회)
        if deviation_20 < -0.1:  # 20일선 대비 10% 이상 하락
            mean_reversion_signals.append(70)
        elif deviation_20 < -0.05:
            mean_reversion_signals.append(60)
        else:
            mean_reversion_signals.append(40)
        
        # RSI 과매도/과매수
        if rsi < 30:
            mean_reversion_signals.append(70)
        elif rsi > 70:
            mean_reversion_signals.append(30)
        else:
            mean_reversion_signals.append(50)
        
        # 볼린저 밴드 신호
        if bb_position < 0.2:  # 하단 밴드 근처
            mean_reversion_signals.append(70)
        elif bb_position > 0.8:  # 상단 밴드 근처
            mean_reversion_signals.append(30)
        else:
            mean_reversion_signals.append(50)
        
        return np.mean(mean_reversion_signals)
        
    except Exception as e:
        logger.warning(f"Mean reversion analysis failed for {ticker}: {str(e)}")
        return 50.0

async def _detect_market_anomalies(self, ticker: str) -> float:
    """시장 이상현상 탐지"""
    try:
        anomaly_score = 0.0
        
        # 1. 요일 효과
        day_of_week_effect = await self._test_day_of_week_effect(ticker)
        anomaly_score += day_of_week_effect * 0.2
        
        # 2. 월별 효과
        monthly_effect = await self._test_monthly_effect(ticker)
        anomaly_score += monthly_effect * 0.2
        
        # 3. 실적 발표 후 드리프트
        earnings_drift = await self._test_earnings_drift(ticker)
        anomaly_score += earnings_drift * 0.3
        
        # 4. 52주 고가/저가 효과
        high_low_effect = await self._test_52_week_effect(ticker)
        anomaly_score += high_low_effect * 0.15
        
        # 5. 거래량 이상현상
        volume_anomaly = await self._test_volume_anomaly(ticker)
        anomaly_score += volume_anomaly * 0.15
        
        return max(0, min(100, anomaly_score))
        
    except Exception as e:
        logger.warning(f"Anomaly detection failed for {ticker}: {str(e)}")
        return 50.0

async def _machine_learning_prediction(self, ticker: str) -> Dict[str, float]:
    """머신러닝 예측 모델"""
    try:
        # 피처 엔지니어링
        features, targets = await self._prepare_ml_features(ticker)
        
        if len(features) < self.min_observations:
            return {'score': 50.0, 'confidence': 0.0, 'prediction': 0.0}
        
        # 시계열 분할 (미래 정보 유출 방지)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 모델별 성능 평가
        model_performances = {}
        
        for model_name, model in self.models.items():
            try:
                # 피처 스케일링
                features_scaled = self.scaler.fit_transform(features)
                
                # 교차 검증
                cv_scores = cross_val_score(
                    model, features_scaled, targets,
                    cv=tscv, scoring='neg_mean_squared_error'
                )
                
                # 모델 학습
                model.fit(features_scaled, targets)
                
                # 예측
                prediction = model.predict(features_scaled[-1:].reshape(1, -1))[0]
                
                model_performances[model_name] = {
                    'cv_score': -np.mean(cv_scores),
                    'prediction': prediction,
                    'r2': np.mean(cross_val_score(model, features_scaled, targets, cv=tscv, scoring='r2'))
                }
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {str(e)}")
                continue
        
        if not model_performances:
            return {'score': 50.0, 'confidence': 0.0, 'prediction': 0.0}
        
        # 최고 성능 모델 선택
        best_model = min(model_performances.items(), key=lambda x: x[1]['cv_score'])
        best_performance = best_model[1]
        
        # 예측 신뢰도 계산
        confidence = max(0, best_performance['r2'])
        
        # ML 점수 계산 (R² 기준)
        ml_score = max(0, min(100, confidence * 100))
        
        return {
            'score': ml_score,
            'confidence': confidence,
            'prediction': best_performance['prediction'],
            'best_model': best_model[0]
        }
        
    except Exception as e:
        logger.warning(f"ML prediction failed for {ticker}: {str(e)}")
        return {'score': 50.0, 'confidence': 0.0, 'prediction': 0.0}

def _calculate_simons_total_score(self, factor_scores: FactorScores, momentum: float,
                                mean_reversion: float, anomaly: float, ml_prediction: Dict,
                                statistical_significance: float, risk_metrics: Dict) -> float:
    """사이먼스 스타일 종합 점수 계산"""
    try:
        # 팩터 점수 평균
        factor_avg = np.mean(list(factor_scores.to_dict().values()))
        
        # 가중치 설정
        weights = {
            'factors': 0.25,           # 멀티 팩터
            'momentum': 0.15,          # 모멘텀
            'mean_reversion': 0.10,    # 평균회귀
            'anomaly': 0.15,           # 이상현상
            'ml_prediction': 0.15,     # ML 예측
            'statistical_sig': 0.10,  # 통계적 유의성
            'risk_adjusted': 0.10      # 리스크 조정
        }
        
        # 리스크 조정 점수
        risk_adjusted_score = min(100, max(0, 
            50 + risk_metrics['sharpe_ratio'] * 25
        ))
        
        # 종합 점수
        components = {
            'factors': factor_avg,
            'momentum': momentum,
            'mean_reversion': mean_reversion,
            'anomaly': anomaly,
            'ml_prediction': ml_prediction['score'],
            'statistical_sig': statistical_significance,
            'risk_adjusted': risk_adjusted_score
        }
        
        total_score = sum(components[key] * weights[key] for key in components)
        
        # 통계적 유의성이 낮으면 페널티
        if statistical_significance < 50:
            total_score *= 0.8
        
        return max(0, min(100, total_score))
        
    except Exception as e:
        logger.warning(f"Total score calculation failed: {str(e)}")
        return 50.0

# 헬퍼 메서드들 (간소화된 구현)
async def _get_financial_data(self, ticker: str) -> Dict:
    """재무 데이터 조회"""
    # Mock 데이터 반환
    return {
        'pe_ratio': 15.0,
        'pb_ratio': 1.5,
        'ev_ebitda': 10.0,
        'roe': 0.12,
        'roa': 0.08,
        'debt_to_equity': 0.3,
        'revenue_growth_3y': 0.08,
        'sector': '반도체'
    }

async def _get_price_data(self, ticker: str) -> Dict:
    """가격 데이터 조회"""
    # Mock 데이터 생성
    np.random.seed(42)
    prices = 50000 * np.cumprod(1 + np.random.normal(0.0008, 0.02, self.lookback_period))
    returns = np.diff(np.log(prices))
    
    return {
        'prices': prices,
        'returns': returns,
        'volume': np.random.normal(1000000, 200000, len(prices))
    }

def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
    """RSI 계산"""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def _calculate_bollinger_position(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> float:
    """볼린저 밴드 내 위치 계산"""
    if len(prices) < period:
        return 0.5
    
    ma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    upper_band = ma + (std_dev * std)
    lower_band = ma - (std_dev * std)
    current_price = prices[-1]
    
    # 0 = 하단 밴드, 1 = 상단 밴드
    position = (current_price - lower_band) / (upper_band - lower_band)
    return max(0, min(1, position))

async def _prepare_ml_features(self, ticker: str) -> Tuple[np.ndarray, np.ndarray]:
    """ML을 위한 피처 준비"""
    price_data = await self._get_price_data(ticker)
    prices = price_data['prices']
    returns = price_data['returns']
    volume = price_data['volume']
    
    if len(prices) < 50:
        return np.array([]), np.array([])
    
    features = []
    targets = []
    
    for i in range(20, len(returns) - 5):  # 20일 lookback, 5일 forward
        # 가격 기반 피처
        price_features = [
            # 모멘텀
            np.mean(returns[i-5:i]),    # 5일 모멘텀
            np.mean(returns[i-20:i]),   # 20일 모멘텀
            
            # 변동성
            np.std(returns[i-20:i]),    # 20일 변동성
            
            # 기술적 지표
            self._calculate_rsi(prices[:i+1]),
            self._calculate_bollinger_position(prices[:i+1]),
            
            # 거래량
            np.log(volume[i] / np.mean(volume[i-20:i])),  # 상대 거래량
            
            # 추가 피처들
            prices[i] / np.mean(prices[i-20:i]) - 1,    # 20일 MA 대비 괴리율
            np.mean(returns[i-5:i]) / np.std(returns[i-20:i])  # 리스크 조정 모멘텀
        ]
        
        features.append(price_features)
        
        # 타겟: 향후 5일 수익률
        future_return = np.mean(returns[i:i+5])
        targets.append(future_return)
    
    return np.array(features), np.array(targets)

# 간소화된 이상현상 테스트 메서드들
async def _test_day_of_week_effect(self, ticker: str) -> float:
    """요일 효과 테스트"""
    return 60.0  # 기본값

async def _test_monthly_effect(self, ticker: str) -> float:
    """월별 효과 테스트"""
    return 55.0

async def _test_earnings_drift(self, ticker: str) -> float:
    """실적 발표 후 드리프트"""
    return 65.0

async def _test_52_week_effect(self, ticker: str) -> float:
    """52주 고가/저가 효과"""
    return 50.0

async def _test_volume_anomaly(self, ticker: str) -> float:
    """거래량 이상현상"""
    return 55.0

async def _statistical_significance_test(self, ticker: str) -> float:
    """통계적 유의성 검증"""
    # t-test 등을 통한 검증 (간소화)
    return 75.0

async def _calculate_risk_metrics(self, ticker: str) -> Dict[str, float]:
    """리스크 지표 계산"""
    price_data = await self._get_price_data(ticker)
    returns = price_data['returns']
    
    if len(returns) < 30:
        return {
            'sharpe_ratio': 0.0,
            'information_ratio': 0.0,
            'max_drawdown': -0.2,
            'win_rate': 0.5
        }
    
    # 샤프 비율
    excess_returns = returns - 0.035/252  # 무위험수익률
    sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    # 최대 낙폭
    cumulative_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # 승률
    win_rate = np.sum(returns > 0) / len(returns)
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'information_ratio': sharpe_ratio,  # 간소화
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }

async def _estimate_alpha(self, ticker: str, factor_scores: FactorScores) -> float:
    """알파 추정"""
    factor_avg = np.mean(list(factor_scores.to_dict().values()))
    expected_alpha = (factor_avg - 50) / 100 * 0.1  # 10% 최대 알파
    return expected_alpha

async def _get_sector_average_pe(self, sector: str) -> float:
    """섹터 평균 PER"""
    sector_pe_map = {
        '반도체': 18.0,
        '금융': 8.0,
        'IT서비스': 25.0,
        '기타': 15.0
    }
    return sector_pe_map.get(sector, 15.0)

async def _get_sector_average_pb(self, sector: str) -> float:
    """섹터 평균 PBR"""
    sector_pb_map = {
        '반도체': 2.0,
        '금융': 0.8,
        'IT서비스': 3.0,
        '기타': 1.5
    }
    return sector_pb_map.get(sector, 1.5)
```

async def create_simons_portfolio(available_tickers: List[str], db: Session,
target_allocation: float = 0.4) -> Dict:
“””
사이먼스 스타일 퀀트 포트폴리오 생성

```
Args:
    available_tickers: 사용 가능한 종목 리스트
    db: 데이터베이스 세션
    target_allocation: 목표 비중
    
Returns:
    포트폴리오 구성 결과
"""
try:
    simons_investor = SimonsQuantInvestor(db)
    
    # 모든 종목에 대해 퀀트 평가 수행
    evaluations = []
    
    for ticker in available_tickers:
        try:
            score = await simons_investor.evaluate_stock(ticker)
            
            if score and score.total_score > 70:  # 70점 이상만 선택 (엄격한 기준)
                evaluations.append({
                    'ticker': ticker,
                    'score': score.total_score,
                    'factor_score': score.factor_score,
                    'sharpe_ratio': score.sharpe_ratio,
                    'expected_alpha': score.expected_alpha,
                    'statistical_significance': score.statistical_significance,
                    'details': score.to_dict()
                })
                
        except Exception as e:
            logger.warning(f"Simons evaluation failed for {ticker}: {str(e)}")
            continue
    
    if not evaluations:
        logger.warning("No stocks passed Simons evaluation")
        return {}
    
    # 알파 기준으로 정렬
    evaluations.sort(key=lambda x: x['expected_alpha'], reverse=True)
    
    # 상위 종목 선택 (최대 20개 - 분산투자)
    max_holdings = min(20, len(evaluations))
    selected_stocks = evaluations[:max_holdings]
    
    # 리스크 패리티 기반 가중치 계산
    portfolio = {}
    total_weight = 0.0
    
    for stock in selected_stocks:
        # 알파와 샤프비율을 결합한 점수
        alpha_score = max(0, stock['expected_alpha'] * 100 + 50)  # 0% 알파 = 50점
        sharpe_score = max(0, min(100, stock['sharpe_ratio'] * 25 + 50))  # 샤프 2.0 = 100점
        stat_sig_score = stock['statistical_significance']
        
        # 복합 점수 (알파 중심)
        composite_score = (
            alpha_score * 0.4 +
            sharpe_score * 0.3 +
            stat_sig_score * 0.2 +
            stock['factor_score'] * 0.1
        )
        
        # 최대 개별 종목 비중 제한 (5%)
        weight = min(0.05, composite_score / 100 * target_allocation / len(selected_stocks))
        total_weight += weight
        
        portfolio[stock['ticker']] = {
            'weight': weight,
            'simons_score': stock['score'],
            'expected_alpha': stock['expected_alpha'],
            'sharpe_ratio': stock['sharpe_ratio'],
            'factor_score': stock['factor_score'],
            'statistical_significance': stock['statistical_significance'],
            'reasoning': f"퀀트 {stock['score']:.1f}점 - 예상알파 {stock['expected_alpha']:.2%}, 샤프 {stock['sharpe_ratio']:.2f}"
        }
    
    # 비중 정규화
    if total_weight > 0:
        normalization_factor = target_allocation / total_weight
        for ticker_info in portfolio.values():
            ticker_info['weight'] *= normalization_factor
    
    return {
        'portfolio': portfolio,
        'total_allocation': target_allocation,
        'selected_count': len(selected_stocks),
        'average_alpha': np.mean([s['expected_alpha'] for s in selected_stocks]),
        'average_sharpe': np.mean([s['sharpe_ratio'] for s in selected_stocks]),
        'strategy': 'Jim Simons Quantitative',
        'philosophy': '수학적 엄밀성과 통계적 패턴 발굴을 통한 투자'
    }
    
except Exception as e:
    logger.error(f"Error creating Simons portfolio: {str(e)}")
    return {}
```