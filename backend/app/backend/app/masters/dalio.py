“””
레이 달리오 All Weather 전략 및 Economic Machine 구현

- 4가지 경제 환경별 자산 배분
- Economic Machine 이해 기반 거시경제 분석
- 리스크 패리티 시스템
- 다차원 분산투자
  “””

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
from sqlalchemy.orm import Session
from scipy.optimize import minimize
import yfinance as yf

from app.models.market import MarketData, EconomicIndicator
from app.utils.calculations import calculate_correlation, calculate_volatility
from app.core.logging import get_logger

logger = get_logger(**name**)

class EconomicEnvironment(Enum):
“”“4가지 경제 환경”””
RECOVERY = “recovery”        # 성장↑ + 인플레↓
BOOM = “boom”               # 성장↑ + 인플레↑
STAGFLATION = “stagflation” # 성장↓ + 인플레↑
RECESSION = “recession”     # 성장↓ + 인플레↓

@dataclass
class EconomicIndicators:
“”“경제 지표 데이터 클래스”””
gdp_growth: float
inflation_rate: float
unemployment_rate: float
interest_rate: float
credit_growth: float
debt_to_gdp: float
productivity_growth: float
currency_strength: float

```
def to_dict(self) -> Dict:
    return {
        'gdp_growth': self.gdp_growth,
        'inflation_rate': self.inflation_rate,
        'unemployment_rate': self.unemployment_rate,
        'interest_rate': self.interest_rate,
        'credit_growth': self.credit_growth,
        'debt_to_gdp': self.debt_to_gdp,
        'productivity_growth': self.productivity_growth,
        'currency_strength': self.currency_strength
    }
```

@dataclass
class DalioScore:
“”“달리오 평가 점수 데이터 클래스”””
total_score: float
economic_environment: EconomicEnvironment
environment_confidence: float
risk_parity_score: float
diversification_score: float
macro_score: float
cycle_position: str
recommended_allocation: Dict[str, float]

```
def to_dict(self) -> Dict:
    return {
        'total_score': self.total_score,
        'economic_environment': self.economic_environment.value,
        'environment_confidence': self.environment_confidence,
        'risk_parity_score': self.risk_parity_score,
        'diversification_score': self.diversification_score,
        'macro_score': self.macro_score,
        'cycle_position': self.cycle_position,
        'recommended_allocation': self.recommended_allocation
    }
```

class EconomicMachine:
“””
달리오의 경제 머신 모델

```
3가지 주요 동력:
1. 단기 부채 사이클 (5-8년)
2. 장기 부채 사이클 (50-75년)
3. 생산성 증가
"""

def __init__(self, db: Session):
    self.db = db
    
def analyze_economic_environment(self, indicators: EconomicIndicators) -> Tuple[EconomicEnvironment, float]:
    """
    현재 경제 환경 분석
    
    Returns:
        (경제환경, 확신도)
    """
    growth_score = self._analyze_growth_trend(indicators)
    inflation_score = self._analyze_inflation_trend(indicators)
    
    # 4가지 환경 매트릭스
    if growth_score > 0 and inflation_score <= 0:
        environment = EconomicEnvironment.RECOVERY
        confidence = min(growth_score, abs(inflation_score))
    elif growth_score > 0 and inflation_score > 0:
        environment = EconomicEnvironment.BOOM
        confidence = min(growth_score, inflation_score)
    elif growth_score <= 0 and inflation_score > 0:
        environment = EconomicEnvironment.STAGFLATION
        confidence = min(abs(growth_score), inflation_score)
    else:  # growth <= 0 and inflation <= 0
        environment = EconomicEnvironment.RECESSION
        confidence = min(abs(growth_score), abs(inflation_score))
    
    return environment, confidence

def _analyze_growth_trend(self, indicators: EconomicIndicators) -> float:
    """
    성장 트렌드 분석
    
    고려 요소:
    - GDP 성장률
    - 실업률 변화
    - 생산성 증가
    - 신용 확장
    """
    score = 0.0
    
    # GDP 성장률 (기준: 잠재성장률 2.5%)
    if indicators.gdp_growth > 0.035:  # 3.5% 이상
        score += 2
    elif indicators.gdp_growth > 0.025:  # 2.5% 이상
        score += 1
    elif indicators.gdp_growth > 0.015:  # 1.5% 이상
        score += 0
    elif indicators.gdp_growth > 0:
        score -= 1
    else:
        score -= 2
    
    # 실업률 (기준: 자연실업률 3.5%)
    if indicators.unemployment_rate < 0.03:
        score += 1
    elif indicators.unemployment_rate < 0.035:
        score += 0.5
    elif indicators.unemployment_rate > 0.05:
        score -= 1
    
    # 신용 증가율
    if indicators.credit_growth > 0.05:
        score += 1
    elif indicators.credit_growth > 0:
        score += 0.5
    else:
        score -= 1
    
    # 생산성 증가
    if indicators.productivity_growth > 0.02:
        score += 1
    elif indicators.productivity_growth > 0:
        score += 0.5
    else:
        score -= 0.5
    
    return max(-3, min(3, score))

def _analyze_inflation_trend(self, indicators: EconomicIndicators) -> float:
    """
    인플레이션 트렌드 분석
    
    고려 요소:
    - 소비자물가상승률
    - 통화정책 기조
    - 환율 변화
    """
    score = 0.0
    
    # 인플레이션율 (기준: 목표 인플레이션 2%)
    if indicators.inflation_rate > 0.04:  # 4% 이상
        score += 2
    elif indicators.inflation_rate > 0.03:  # 3% 이상
        score += 1
    elif indicators.inflation_rate > 0.02:  # 2% 이상
        score += 0.5
    elif indicators.inflation_rate > 0.01:  # 1% 이상
        score += 0
    elif indicators.inflation_rate > 0:
        score -= 0.5
    else:  # 디플레이션
        score -= 2
    
    # 금리 정책 기조
    if indicators.interest_rate > 0.03:  # 긴축적
        score += 0.5
    elif indicators.interest_rate < 0.01:  # 완화적
        score -= 0.5
    
    # 환율 (원화 약세는 인플레이션 압력)
    if indicators.currency_strength < -0.05:  # 5% 이상 약세
        score += 1
    elif indicators.currency_strength > 0.05:  # 5% 이상 강세
        score -= 0.5
    
    return max(-3, min(3, score))

def analyze_debt_cycle(self, indicators: EconomicIndicators) -> Dict[str, str]:
    """
    부채 사이클 분석
    """
    short_term_cycle = self._analyze_short_term_debt_cycle(indicators)
    long_term_cycle = self._analyze_long_term_debt_cycle(indicators)
    
    return {
        'short_term_cycle': short_term_cycle,
        'long_term_cycle': long_term_cycle
    }

def _analyze_short_term_debt_cycle(self, indicators: EconomicIndicators) -> str:
    """단기 부채 사이클 (5-8년) 분석"""
    
    # 신용 확장 단계
    if indicators.credit_growth > 0.08 and indicators.gdp_growth > 0.03:
        return "expansion"  # 확장기
    
    # 버블 단계
    elif indicators.credit_growth > 0.05 and indicators.inflation_rate > 0.03:
        return "bubble"  # 버블기
    
    # 수축 단계
    elif indicators.credit_growth < 0.02 and indicators.gdp_growth < 0.02:
        return "contraction"  # 수축기
    
    # 회복 단계
    else:
        return "recovery"  # 회복기

def _analyze_long_term_debt_cycle(self, indicators: EconomicIndicators) -> str:
    """장기 부채 사이클 (50-75년) 분석"""
    
    debt_ratio = indicators.debt_to_gdp
    productivity = indicators.productivity_growth
    
    # 건전한 성장기 (부채 < 250% GDP)
    if debt_ratio < 2.5 and productivity > 0.015:
        return "healthy_growth"
    
    # 부채 누적기 (부채 250-400% GDP)
    elif debt_ratio < 4.0:
        return "debt_accumulation"
    
    # 디레버리징 필요 (부채 > 400% GDP)
    else:
        return "deleveraging_needed"
```

class AllWeatherStrategy:
“””
레이 달리오의 All Weather 전략

```
핵심 원칙:
1. 리스크 패리티 (동일한 리스크 기여도)
2. 4가지 경제 환경에 대한 균형
3. 낮은 상관관계 자산 조합
"""

def __init__(self, db: Session):
    self.db = db
    self.economic_machine = EconomicMachine(db)
    
    # 자산군별 기본 설정
    self.asset_classes = {
        'growth_stocks': ['005930', '000660', '035420'],  # 삼성전자, SK하이닉스, NAVER
        'value_stocks': ['055550', '015760', '105560'],   # 신한지주, 한국전력, KB금융
        'defensive_stocks': ['090430', '271560', '000810'], # 아모레퍼시픽, 엘지전자, 삼성화재
        'materials': ['005490', '051910', '010950'],      # POSCO, LG화학, S-Oil
        'utilities': ['015760', '036460', '267250'],      # 한국전력, 한국가스공사, 현대중공업
        'reits': ['114800', '365550'],                    # 코람코리츠, ESR켄달스퀘어리츠
        'bonds': ['148070', '114260'],                    # KOSEF 국고채10년, KODEX 국고채3년
        'cash': ['069500']                                # KODEX 200
    }

async def create_all_weather_portfolio(self, target_allocation: float = 0.3) -> Dict:
    """
    All Weather 포트폴리오 생성
    
    Args:
        target_allocation: 전체 포트폴리오에서 달리오 전략 비중
    """
    try:
        # 1. 현재 경제 환경 분석
        indicators = await self._get_economic_indicators()
        environment, confidence = self.economic_machine.analyze_economic_environment(indicators)
        
        # 2. 부채 사이클 분석
        debt_cycles = self.economic_machine.analyze_debt_cycle(indicators)
        
        # 3. 환경별 최적 자산 배분
        base_allocation = self._get_environment_allocation(environment)
        
        # 4. 리스크 패리티 조정
        risk_adjusted_allocation = await self._apply_risk_parity(base_allocation)
        
        # 5. 구체적 종목 선택
        portfolio = await self._select_specific_stocks(risk_adjusted_allocation, target_allocation)
        
        # 6. 성과 지표 계산
        score = self._calculate_dalio_score(
            environment, confidence, risk_adjusted_allocation, portfolio
        )
        
        return {
            'portfolio': portfolio,
            'total_allocation': target_allocation,
            'economic_environment': environment.value,
            'environment_confidence': confidence,
            'debt_cycles': debt_cycles,
            'score': score.to_dict(),
            'strategy': 'Ray Dalio All Weather'
        }
        
    except Exception as e:
        logger.error(f"Error creating All Weather portfolio: {str(e)}")
        return {}

def _get_environment_allocation(self, environment: EconomicEnvironment) -> Dict[str, float]:
    """
    경제 환경별 기본 자산 배분
    """
    allocations = {
        EconomicEnvironment.RECOVERY: {
            'growth_stocks': 0.35,    # 성장주 비중 확대
            'value_stocks': 0.20,
            'defensive_stocks': 0.10,
            'materials': 0.10,
            'utilities': 0.05,
            'reits': 0.05,
            'bonds': 0.10,
            'cash': 0.05
        },
        EconomicEnvironment.BOOM: {
            'growth_stocks': 0.20,    # 성장주 비중 축소
            'value_stocks': 0.15,
            'defensive_stocks': 0.15,
            'materials': 0.20,        # 원자재 비중 확대
            'utilities': 0.10,
            'reits': 0.10,
            'bonds': 0.05,           # 채권 비중 축소
            'cash': 0.05
        },
        EconomicEnvironment.STAGFLATION: {
            'growth_stocks': 0.10,    # 성장주 대폭 축소
            'value_stocks': 0.15,
            'defensive_stocks': 0.25, # 방어주 비중 확대
            'materials': 0.20,        # 인플레이션 헤지
            'utilities': 0.15,
            'reits': 0.10,
            'bonds': 0.00,           # 채권 회피
            'cash': 0.05
        },
        EconomicEnvironment.RECESSION: {
            'growth_stocks': 0.15,
            'value_stocks': 0.15,
            'defensive_stocks': 0.20,
            'materials': 0.05,        # 원자재 비중 축소
            'utilities': 0.15,
            'reits': 0.05,
            'bonds': 0.20,           # 채권 비중 확대
            'cash': 0.05
        }
    }
    
    return allocations[environment]

async def _apply_risk_parity(self, base_allocation: Dict[str, float]) -> Dict[str, float]:
    """
    리스크 패리티 적용
    
    각 자산군의 리스크 기여도를 균등화
    """
    try:
        # 자산군별 변동성 계산
        volatilities = {}
        correlations = {}
        
        for asset_class, tickers in self.asset_classes.items():
            if asset_class in base_allocation:
                vol = await self._calculate_asset_class_volatility(tickers)
                volatilities[asset_class] = vol
        
        # 상관관계 매트릭스 계산
        correlations = await self._calculate_correlation_matrix(
            list(base_allocation.keys())
        )
        
        # 리스크 패리티 최적화
        optimized_weights = self._optimize_risk_parity(
            base_allocation, volatilities, correlations
        )
        
        return optimized_weights
        
    except Exception as e:
        logger.warning(f"Risk parity optimization failed: {str(e)}")
        return base_allocation

def _optimize_risk_parity(self, base_allocation: Dict[str, float], 
                        volatilities: Dict[str, float], 
                        correlations: np.ndarray) -> Dict[str, float]:
    """
    리스크 패리티 최적화
    """
    asset_names = list(base_allocation.keys())
    n_assets = len(asset_names)
    
    # 초기 가중치
    initial_weights = np.array([base_allocation[name] for name in asset_names])
    
    # 변동성 벡터
    vol_vector = np.array([volatilities.get(name, 0.15) for name in asset_names])
    
    def risk_parity_objective(weights):
        """리스크 패리티 목적함수"""
        portfolio_vol = np.sqrt(weights.T @ correlations @ weights * vol_vector)
        
        # 각 자산의 리스크 기여도
        risk_contributions = weights * (correlations @ weights * vol_vector) / portfolio_vol
        
        # 리스크 기여도의 분산을 최소화
        target_risk_contribution = 1.0 / n_assets
        risk_diff = risk_contributions - target_risk_contribution
        
        return np.sum(risk_diff ** 2)
    
    # 제약조건
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 가중치 합 = 1
    ]
    
    # 경계조건 (0 ~ 50%)
    bounds = [(0.0, 0.5) for _ in range(n_assets)]
    
    # 최적화 실행
    result = minimize(
        risk_parity_objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        optimized_weights = result.x
        return {asset_names[i]: optimized_weights[i] for i in range(n_assets)}
    else:
        logger.warning("Risk parity optimization failed, using base allocation")
        return base_allocation

async def _select_specific_stocks(self, allocation: Dict[str, float], 
                                target_allocation: float) -> Dict[str, Dict]:
    """
    자산군별 구체적 종목 선택
    """
    portfolio = {}
    
    for asset_class, class_weight in allocation.items():
        if class_weight > 0 and asset_class in self.asset_classes:
            tickers = self.asset_classes[asset_class]
            
            # 자산군 내 종목별 가중치 계산
            stock_weights = await self._calculate_stock_weights_in_class(
                tickers, asset_class
            )
            
            # 포트폴리오에 추가
            for ticker, stock_weight in stock_weights.items():
                final_weight = class_weight * stock_weight * target_allocation
                
                if final_weight > 0.005:  # 0.5% 이상만 포함
                    portfolio[ticker] = {
                        'weight': final_weight,
                        'asset_class': asset_class,
                        'dalio_reasoning': f"All Weather {asset_class} 전략, "
                                         f"경제환경 대응 비중 {final_weight:.1%}"
                    }
    
    return portfolio

async def _calculate_stock_weights_in_class(self, tickers: List[str], 
                                          asset_class: str) -> Dict[str, float]:
    """
    자산군 내 종목별 가중치 계산
    """
    if asset_class in ['growth_stocks', 'value_stocks', 'defensive_stocks']:
        # 주식은 시가총액 가중 + 품질 점수
        weights = await self._calculate_quality_weighted_allocation(tickers)
    elif asset_class == 'materials':
        # 원자재는 균등 가중
        weights = {ticker: 1.0/len(tickers) for ticker in tickers}
    else:
        # 기타는 균등 가중
        weights = {ticker: 1.0/len(tickers) for ticker in tickers}
    
    return weights

async def _calculate_quality_weighted_allocation(self, tickers: List[str]) -> Dict[str, float]:
    """
    품질 점수 기반 가중치 계산
    """
    quality_scores = {}
    
    for ticker in tickers:
        # 품질 점수 계산 (ROE, 부채비율, 현금흐름 등)
        score = await self._calculate_quality_score(ticker)
        quality_scores[ticker] = score
    
    # 점수 기반 가중치 정규화
    total_score = sum(quality_scores.values())
    if total_score > 0:
        return {ticker: score/total_score for ticker, score in quality_scores.items()}
    else:
        # 균등 가중
        return {ticker: 1.0/len(tickers) for ticker in tickers}

async def _calculate_quality_score(self, ticker: str) -> float:
    """
    종목별 품질 점수 계산
    """
    try:
        # 실제 구현에서는 재무 데이터를 사용
        # 여기서는 기본 점수 반환
        return 1.0
    except:
        return 1.0

async def _get_economic_indicators(self) -> EconomicIndicators:
    """
    경제 지표 수집
    """
    # 실제 구현에서는 한국은행, FRED API 등에서 수집
    return EconomicIndicators(
        gdp_growth=0.025,          # 2.5%
        inflation_rate=0.035,      # 3.5%
        unemployment_rate=0.032,   # 3.2%
        interest_rate=0.035,       # 3.5%
        credit_growth=0.045,       # 4.5%
        debt_to_gdp=2.8,          # 280%
        productivity_growth=0.015, # 1.5%
        currency_strength=0.02     # 2% 강세
    )

async def _calculate_asset_class_volatility(self, tickers: List[str]) -> float:
    """
    자산군 변동성 계산
    """
    # 대표 종목의 변동성으로 근사
    if tickers:
        return 0.20  # 기본값 20%
    return 0.15

async def _calculate_correlation_matrix(self, asset_classes: List[str]) -> np.ndarray:
    """
    자산군간 상관관계 매트릭스 계산
    """
    n = len(asset_classes)
    
    # 기본 상관관계 매트릭스 (한국 시장 기준)
    correlation_map = {
        ('growth_stocks', 'growth_stocks'): 1.0,
        ('growth_stocks', 'value_stocks'): 0.7,
        ('growth_stocks', 'defensive_stocks'): 0.5,
        ('growth_stocks', 'materials'): 0.6,
        ('growth_stocks', 'utilities'): 0.3,
        ('growth_stocks', 'reits'): 0.4,
        ('growth_stocks', 'bonds'): -0.2,
        ('growth_stocks', 'cash'): 0.0,
        
        ('value_stocks', 'value_stocks'): 1.0,
        ('value_stocks', 'defensive_stocks'): 0.6,
        ('value_stocks', 'materials'): 0.5,
        ('value_stocks', 'utilities'): 0.4,
        ('value_stocks', 'reits'): 0.3,
        ('value_stocks', 'bonds'): -0.1,
        ('value_stocks', 'cash'): 0.0,
        
        ('defensive_stocks', 'defensive_stocks'): 1.0,
        ('defensive_stocks', 'materials'): 0.3,
        ('defensive_stocks', 'utilities'): 0.5,
        ('defensive_stocks', 'reits'): 0.4,
        ('defensive_stocks', 'bonds'): 0.1,
        ('defensive_stocks', 'cash'): 0.0,
        
        ('materials', 'materials'): 1.0,
        ('materials', 'utilities'): 0.2,
        ('materials', 'reits'): 0.3,
        ('materials', 'bonds'): -0.3,
        ('materials', 'cash'): 0.0,
        
        ('utilities', 'utilities'): 1.0,
        ('utilities', 'reits'): 0.6,
        ('utilities', 'bonds'): 0.3,
        ('utilities', 'cash'): 0.0,
        
        ('reits', 'reits'): 1.0,
        ('reits', 'bonds'): 0.2,
        ('reits', 'cash'): 0.0,
        
        ('bonds', 'bonds'): 1.0,
        ('bonds', 'cash'): 0.1,
        
        ('cash', 'cash'): 1.0
    }
    
    # 매트릭스 생성
    matrix = np.eye(n)
    for i, asset1 in enumerate(asset_classes):
        for j, asset2 in enumerate(asset_classes):
            if i != j:
                key = (asset1, asset2) if (asset1, asset2) in correlation_map else (asset2, asset1)
                matrix[i, j] = correlation_map.get(key, 0.3)  # 기본값 0.3
    
    return matrix

def _calculate_dalio_score(self, environment: EconomicEnvironment, confidence: float,
                         allocation: Dict[str, float], portfolio: Dict) -> DalioScore:
    """
    달리오 전략 종합 점수 계산
    """
    # 1. 거시경제 분석 점수
    macro_score = confidence * 100
    
    # 2. 리스크 패리티 점수
    risk_parity_score = self._evaluate_risk_parity_quality(allocation)
    
    # 3. 분산투자 점수
    diversification_score = self._evaluate_diversification_quality(portfolio)
    
    # 4. 종합 점수 (가중평균)
    total_score = (
        macro_score * 0.3 +
        risk_parity_score * 0.4 +
        diversification_score * 0.3
    )
    
    return DalioScore(
        total_score=total_score,
        economic_environment=environment,
        environment_confidence=confidence,
        risk_parity_score=risk_parity_score,
        diversification_score=diversification_score,
        macro_score=macro_score,
        cycle_position=self._determine_cycle_position(environment),
        recommended_allocation=allocation
    )

def _evaluate_risk_parity_quality(self, allocation: Dict[str, float]) -> float:
    """
    리스크 패리티 품질 평가
    """
    # 자산군별 분산도 측정
    weights = list(allocation.values())
    
    if not weights:
        return 0.0
    
    # 허핀달 지수 계산 (집중도 측정)
    hhi = sum(w**2 for w in weights)
    
    # 이상적인 분산 대비 점수 (낮을수록 좋음)
    ideal_hhi = 1.0 / len(weights)
    concentration_penalty = min(hhi / ideal_hhi, 2.0)
    
    score = max(0, 100 - (concentration_penalty - 1) * 50)
    return score

def _evaluate_diversification_quality(self, portfolio: Dict) -> float:
    """
    분산투자 품질 평가
    """
    if not portfolio:
        return 0.0
    
    # 종목 수
    num_stocks = len(portfolio)
    
    # 최대 집중도
    max_weight = max(stock['weight'] for stock in portfolio.values())
    
    # 섹터 분산도
    sectors = set()
    for stock_info in portfolio.values():
        sectors.add(stock_info.get('asset_class', 'unknown'))
    
    sector_count = len(sectors)
    
    # 점수 계산
    score = 0.0
    
    # 종목 수 점수 (10-30개가 이상적)
    if 15 <= num_stocks <= 25:
        score += 40
    elif 10 <= num_stocks <= 30:
        score += 30
    else:
        score += 20
    
    # 집중도 점수 (개별 종목 15% 이하가 이상적)
    if max_weight <= 0.10:
        score += 30
    elif max_weight <= 0.15:
        score += 25
    elif max_weight <= 0.20:
        score += 15
    else:
        score += 5
    
    # 섹터 분산 점수
    if sector_count >= 6:
        score += 30
    elif sector_count >= 4:
        score += 20
    else:
        score += 10
    
    return score

def _determine_cycle_position(self, environment: EconomicEnvironment) -> str:
    """
    경제 사이클 내 위치 판단
    """
    cycle_map = {
        EconomicEnvironment.RECOVERY: "초기 회복기",
        EconomicEnvironment.BOOM: "성장 극대기",
        EconomicEnvironment.STAGFLATION: "조정기",
        EconomicEnvironment.RECESSION: "침체기"
    }
    
    return cycle_map[environment]
```

# 달리오 포트폴리오 구성 함수

async def create_dalio_portfolio(available_tickers: List[str], db: Session,
target_allocation: float = 0.3) -> Dict:
“””
달리오 All Weather 포트폴리오 구성

```
Args:
    available_tickers: 사용 가능한 종목 리스트
    db: 데이터베이스 세션
    target_allocation: 전체 포트폴리오에서 달리오 전략 비중

Returns:
    포트폴리오 구성 결과
"""
try:
    all_weather = AllWeatherStrategy(db)
    
    # All Weather 포트폴리오 생성
    result = await all_weather.create_all_weather_portfolio(target_allocation)
    
    if not result:
        logger.error("Failed to create All Weather portfolio")
        return {}
    
    # 추가 메타데이터
    result.update({
        'creation_time': datetime.now().isoformat(),
        'rebalance_frequency': 'quarterly',
        'risk_target': 'medium_volatility',
        'philosophy': 'Economic Machine based asset allocation with risk parity'
    })
    
    logger.info(f"Created Dalio portfolio with {len(result['portfolio'])} holdings")
    return result
    
except Exception as e:
    logger.error(f"Error creating Dalio portfolio: {str(e)}")
    return {}
```

# 추가 유틸리티 함수들

def calculate_economic_regime_probability(indicators: EconomicIndicators) -> Dict[str, float]:
“””
각 경제 환경별 확률 계산
“””
machine = EconomicMachine(None)  # db 없이 분석만

```
growth_score = machine._analyze_growth_trend(indicators)
inflation_score = machine._analyze_inflation_trend(indicators)

# 각 환경별 확률 계산 (소프트맥스 방식)
scores = {
    'recovery': max(0, growth_score) * max(0, -inflation_score),
    'boom': max(0, growth_score) * max(0, inflation_score),
    'stagflation': max(0, -growth_score) * max(0, inflation_score),
    'recession': max(0, -growth_score) * max(0, -inflation_score)
}

total = sum(scores.values())
if total > 0:
    probabilities = {k: v/total for k, v in scores.items()}
else:
    probabilities = {k: 0.25 for k in scores.keys()}  # 균등 확률

return probabilities
```

def create_scenario_analysis(base_indicators: EconomicIndicators) -> Dict[str, Dict]:
“””
시나리오 분석 생성
“””
scenarios = {
‘base_case’: base_indicators,
‘optimistic’: EconomicIndicators(
gdp_growth=base_indicators.gdp_growth + 0.01,
inflation_rate=base_indicators.inflation_rate - 0.005,
unemployment_rate=base_indicators.unemployment_rate - 0.005,
interest_rate=base_indicators.interest_rate,
credit_growth=base_indicators.credit_growth + 0.02,
debt_to_gdp=base_indicators.debt_to_gdp - 0.1,
productivity_growth=base_indicators.productivity_growth + 0.005,
currency_strength=base_indicators.currency_strength + 0.02
),
‘pessimistic’: EconomicIndicators(
gdp_growth=base_indicators.gdp_growth - 0.015,
inflation_rate=base_indicators.inflation_rate + 0.01,
unemployment_rate=base_indicators.unemployment_rate + 0.01,
interest_rate=base_indicators.interest_rate + 0.01,
credit_growth=base_indicators.credit_growth - 0.03,
debt_to_gdp=base_indicators.debt_to_gdp + 0.2,
productivity_growth=base_indicators.productivity_growth - 0.005,
currency_strength=base_indicators.currency_strength - 0.05
)
}

```
results = {}
for scenario_name, indicators in scenarios.items():
    probabilities = calculate_economic_regime_probability(indicators)
    results[scenario_name] = {
        'indicators': indicators.to_dict(),
        'regime_probabilities': probabilities
    }

return results
```