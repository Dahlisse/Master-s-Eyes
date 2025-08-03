“””
4대 거장 베이스 클래스 및 공통 인터페이스

- 모든 거장 알고리즘의 공통 인터페이스 정의
- 공통 유틸리티 함수
- 데이터 검증 및 품질 관리
  “””

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from app.core.logging import get_logger

logger = get_logger(**name**)

@dataclass
class MasterScore:
“”“거장별 평가 점수 기본 클래스”””
master_name: str
total_score: float
confidence: float
reasoning: str
sub_scores: Dict[str, float]
metadata: Dict[str, Any]

```
def to_dict(self) -> Dict:
    return {
        'master_name': self.master_name,
        'total_score': self.total_score,
        'confidence': self.confidence,
        'reasoning': self.reasoning,
        'sub_scores': self.sub_scores,
        'metadata': self.metadata
    }
```

@dataclass
class PortfolioRecommendation:
“”“포트폴리오 추천 기본 클래스”””
ticker: str
weight: float
score: float
reasoning: str
risk_level: str
expected_return: float
volatility: float

```
def to_dict(self) -> Dict:
    return {
        'ticker': self.ticker,
        'weight': self.weight,
        'score': self.score,
        'reasoning': self.reasoning,
        'risk_level': self.risk_level,
        'expected_return': self.expected_return,
        'volatility': self.volatility
    }
```

class BaseMaster(ABC):
“””
모든 거장 알고리즘의 베이스 클래스

```
공통 인터페이스:
1. evaluate_stock: 개별 종목 평가
2. create_portfolio: 포트폴리오 생성
3. get_insights: 투자 인사이트 제공
"""

def __init__(self, db: Session, master_name: str):
    self.db = db
    self.master_name = master_name
    self.logger = get_logger(f"{__name__}.{master_name}")
    
    # 공통 설정
    self.min_market_cap = 1e11  # 최소 시가총액 1000억원
    self.min_trading_volume = 1e8  # 최소 거래대금 1억원
    self.max_position_size = 0.15  # 최대 개별 종목 비중 15%
    self.min_position_size = 0.005  # 최소 개별 종목 비중 0.5%
    
@abstractmethod
async def evaluate_stock(self, ticker: str) -> Optional[MasterScore]:
    """
    개별 종목 평가
    
    Args:
        ticker: 종목 코드
        
    Returns:
        MasterScore 객체 또는 None
    """
    pass

@abstractmethod
async def create_portfolio(self, 
                          available_tickers: List[str], 
                          target_allocation: float) -> Dict[str, PortfolioRecommendation]:
    """
    포트폴리오 생성
    
    Args:
        available_tickers: 사용 가능한 종목 리스트
        target_allocation: 목표 비중
        
    Returns:
        포트폴리오 딕셔너리
    """
    pass

@abstractmethod
def get_investment_philosophy(self) -> Dict[str, str]:
    """
    투자 철학 반환
    
    Returns:
        투자 철학 딕셔너리
    """
    pass

async def validate_ticker(self, ticker: str) -> bool:
    """
    종목 유효성 검증
    
    Args:
        ticker: 종목 코드
        
    Returns:
        유효성 여부
    """
    try:
        # 기본 검증 로직
        market_data = await self._get_basic_market_data(ticker)
        
        if not market_data:
            return False
        
        # 시가총액 검증
        if market_data.get('market_cap', 0) < self.min_market_cap:
            self.logger.debug(f"Market cap too small for {ticker}")
            return False
        
        # 거래량 검증
        if market_data.get('avg_volume', 0) < self.min_trading_volume:
            self.logger.debug(f"Trading volume too low for {ticker}")
            return False
        
        # 상장폐지 위험 검증
        if market_data.get('delisting_risk', False):
            self.logger.debug(f"Delisting risk for {ticker}")
            return False
        
        return True
        
    except Exception as e:
        self.logger.error(f"Error validating ticker {ticker}: {str(e)}")
        return False

async def _get_basic_market_data(self, ticker: str) -> Optional[Dict]:
    """
    기본 시장 데이터 조회
    
    Args:
        ticker: 종목 코드
        
    Returns:
        시장 데이터 딕셔너리
    """
    try:
        # 실제 구현에서는 데이터베이스에서 조회
        # 여기서는 더미 데이터 반환
        return {
            'market_cap': 2e12,  # 2조원
            'avg_volume': 5e8,   # 5억원
            'delisting_risk': False,
            'current_price': 50000,
            'beta': 1.0
        }
    except Exception as e:
        self.logger.error(f"Error getting market data for {ticker}: {str(e)}")
        return None

def normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """
    점수 정규화
    
    Args:
        score: 원본 점수
        min_val: 최소값
        max_val: 최대값
        
    Returns:
        정규화된 점수
    """
    return max(min_val, min(max_val, score))

def calculate_confidence_interval(self, scores: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    신뢰구간 계산
    
    Args:
        scores: 점수 리스트
        confidence_level: 신뢰수준
        
    Returns:
        (하한, 상한) 튜플
    """
    if not scores:
        return (0.0, 0.0)
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # t-분포 기반 신뢰구간
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence_level) / 2, len(scores) - 1)
    margin_error = t_value * std_score / np.sqrt(len(scores))
    
    return (mean_score - margin_error, mean_score + margin_error)

def weight_by_conviction(self, scores: Dict[str, float], 
                       target_allocation: float,
                       conviction_threshold: float = 70.0) -> Dict[str, float]:
    """
    확신도 기반 가중치 계산
    
    Args:
        scores: 종목별 점수
        target_allocation: 목표 비중
        conviction_threshold: 확신 임계값
        
    Returns:
        종목별 가중치
    """
    # 임계값 이상 종목만 선택
    qualified_stocks = {k: v for k, v in scores.items() if v >= conviction_threshold}
    
    if not qualified_stocks:
        return {}
    
    # 점수 기반 가중치 계산
    total_score = sum(qualified_stocks.values())
    base_weights = {k: (v / total_score) for k, v in qualified_stocks.items()}
    
    # 목표 비중으로 조정
    final_weights = {}
    for ticker, weight in base_weights.items():
        adjusted_weight = weight * target_allocation
        
        # 최소/최대 비중 제한
        adjusted_weight = max(self.min_position_size, 
                            min(self.max_position_size, adjusted_weight))
        
        if adjusted_weight >= self.min_position_size:
            final_weights[ticker] = adjusted_weight
    
    # 비중 정규화
    total_weight = sum(final_weights.values())
    if total_weight > 0 and total_weight != target_allocation:
        normalization_factor = target_allocation / total_weight
        final_weights = {k: v * normalization_factor for k, v in final_weights.items()}
    
    return final_weights

async def get_sector_exposure(self, portfolio: Dict[str, float]) -> Dict[str, float]:
    """
    섹터 노출도 계산
    
    Args:
        portfolio: 포트폴리오 딕셔너리
        
    Returns:
        섹터별 비중
    """
    sector_weights = {}
    
    for ticker, weight in portfolio.items():
        try:
            # 종목의 섹터 정보 조회
            sector = await self._get_ticker_sector(ticker)
            
            if sector:
                if sector in sector_weights:
                    sector_weights[sector] += weight
                else:
                    sector_weights[sector] = weight
                    
        except Exception as e:
            self.logger.warning(f"Error getting sector for {ticker}: {str(e)}")
    
    return sector_weights

async def _get_ticker_sector(self, ticker: str) -> Optional[str]:
    """
    종목의 섹터 정보 조회
    
    Args:
        ticker: 종목 코드
        
    Returns:
        섹터명
    """
    try:
        # 실제 구현에서는 데이터베이스에서 조회
        sector_map = {
            '005930': '반도체',
            '000660': '반도체', 
            '035420': 'IT서비스',
            '055550': '금융',
            '051910': '화학',
            # ... 더 많은 매핑
        }
        
        return sector_map.get(ticker, '기타')
        
    except Exception as e:
        self.logger.error(f"Error getting sector for {ticker}: {str(e)}")
        return None

def calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
    """
    리스크 지표 계산
    
    Args:
        returns: 수익률 리스트
        
    Returns:
        리스크 지표 딕셔너리
    """
    if not returns:
        return {}
    
    returns_array = np.array(returns)
    
    # 기본 통계
    mean_return = np.mean(returns_array)
    volatility = np.std(returns_array)
    
    # 샤프 비율 (무위험수익률 3.5% 가정)
    risk_free_rate = 0.035
    sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # 최대 낙폭 (Maximum Drawdown)
    cumulative_returns = np.cumprod(1 + returns_array)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # VaR (Value at Risk) 95%
    var_95 = np.percentile(returns_array, 5)
    
    # 소르티노 비율 (하방 위험 기준)
    downside_returns = returns_array[returns_array < 0]
    downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
    
    return {
        'mean_return': mean_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'sortino_ratio': sortino_ratio,
        'downside_volatility': downside_volatility
    }

def generate_explanation(self, ticker: str, score: MasterScore) -> str:
    """
    투자 논리 설명 생성
    
    Args:
        ticker: 종목 코드
        score: 평가 점수
        
    Returns:
        설명 텍스트
    """
    explanation_template = (
        f"[{self.master_name} 관점] {ticker} 분석 결과:\n"
        f"• 종합 점수: {score.total_score:.1f}점 (확신도: {score.confidence:.1%})\n"
        f"• 핵심 논리: {score.reasoning}\n"
    )
    
    if score.sub_scores:
        explanation_template += "• 세부 점수:\n"
        for category, sub_score in score.sub_scores.items():
            explanation_template += f"  - {category}: {sub_score:.1f}점\n"
    
    return explanation_template
```

class DataQualityManager:
“””
데이터 품질 관리 클래스
“””

```
def __init__(self):
    self.logger = get_logger(f"{__name__}.DataQuality")
    
def validate_financial_data(self, data: Dict) -> Dict[str, bool]:
    """
    재무 데이터 유효성 검증
    
    Args:
        data: 재무 데이터 딕셔너리
        
    Returns:
        검증 결과 딕셔너리
    """
    validations = {}
    
    # 필수 필드 검증
    required_fields = ['revenue', 'net_income', 'total_assets', 'total_equity']
    validations['required_fields'] = all(field in data for field in required_fields)
    
    # 논리적 일관성 검증
    if 'total_assets' in data and 'total_equity' in data and 'total_debt' in data:
        # 자산 = 부채 + 자본 (오차 허용 5%)
        assets = data['total_assets']
        equity = data['total_equity']
        debt = data.get('total_debt', 0)
        
        balance_check = abs(assets - (equity + debt)) / assets < 0.05
        validations['balance_sheet_consistency'] = balance_check
    
    # 이상치 검증
    if 'roe' in data:
        # ROE가 -100% ~ 100% 범위 내
        validations['roe_reasonable'] = -1.0 <= data['roe'] <= 1.0
    
    if 'debt_ratio' in data:
        # 부채비율이 0% ~ 500% 범위 내
        validations['debt_ratio_reasonable'] = 0 <= data['debt_ratio'] <= 5.0
    
    # 시계열 일관성 검증
    if 'revenue_history' in data:
        revenue_history = data['revenue_history']
        if len(revenue_history) >= 2:
            # 연속된 0 또는 음수 매출 없음
            validations['revenue_continuity'] = all(rev > 0 for rev in revenue_history)
    
    return validations

def clean_outliers(self, values: List[float], method: str = 'iqr') -> List[float]:
    """
    이상치 제거
    
    Args:
        values: 데이터 리스트
        method: 제거 방법 ('iqr', 'zscore')
        
    Returns:
        정제된 데이터 리스트
    """
    if not values:
        return values
    
    values_array = np.array(values)
    
    if method == 'iqr':
        Q1 = np.percentile(values_array, 25)
        Q3 = np.percentile(values_array, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (values_array >= lower_bound) & (values_array <= upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((values_array - np.mean(values_array)) / np.std(values_array))
        mask = z_scores < 3  # 3-sigma 룰
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return values_array[mask].tolist()

def interpolate_missing_data(self, data: Dict[str, List[float]], 
                           method: str = 'linear') -> Dict[str, List[float]]:
    """
    결측값 보간
    
    Args:
        data: 시계열 데이터 딕셔너리
        method: 보간 방법
        
    Returns:
        보간된 데이터 딕셔너리
    """
    interpolated_data = {}
    
    for key, values in data.items():
        if not values:
            interpolated_data[key] = values
            continue
        
        # pandas를 사용한 보간
        series = pd.Series(values)
        
        if method == 'linear':
            interpolated = series.interpolate(method='linear')
        elif method == 'forward_fill':
            interpolated = series.fillna(method='ffill')
        elif method == 'backward_fill':
            interpolated = series.fillna(method='bfill')
        else:
            interpolated = series
        
        interpolated_data[key] = interpolated.tolist()
    
    return interpolated_data
```

# 공통 계산 함수들

def calculate_compound_growth_rate(values: List[float], periods: int = None) -> float:
“””
복합연간성장률(CAGR) 계산

```
Args:
    values: 값 리스트
    periods: 기간 수 (None이면 자동 계산)
    
Returns:
    CAGR
"""
if len(values) < 2 or values[0] <= 0:
    return 0.0

if periods is None:
    periods = len(values) - 1

start_value = values[0]
end_value = values[-1]

cagr = ((end_value / start_value) ** (1 / periods)) - 1

# 이상치 제거 (-50% ~ 100%)
return max(-0.5, min(1.0, cagr))
```

def calculate_rolling_correlation(series1: List[float], series2: List[float],
window: int = 12) -> List[float]:
“””
이동 상관계수 계산

```
Args:
    series1: 첫 번째 시계열
    series2: 두 번째 시계열
    window: 윈도우 크기
    
Returns:
    이동 상관계수 리스트
"""
if len(series1) != len(series2) or len(series1) < window:
    return []

correlations = []

for i in range(window - 1, len(series1)):
    subset1 = series1[i - window + 1:i + 1]
    subset2 = series2[i - window + 1:i + 1]
    
    corr = np.corrcoef(subset1, subset2)[0, 1]
    correlations.append(corr if not np.isnan(corr) else 0.0)

return correlations
```

def calculate_information_ratio(portfolio_returns: List[float],
benchmark_returns: List[float]) -> float:
“””
정보비율 계산

```
Args:
    portfolio_returns: 포트폴리오 수익률
    benchmark_returns: 벤치마크 수익률
    
Returns:
    정보비율
"""
if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
    return 0.0

excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)

mean_excess = np.mean(excess_returns)
std_excess = np.std(excess_returns)

return mean_excess / std_excess if std_excess > 0 else 0.0
```

def calculate_beta(stock_returns: List[float], market_returns: List[float]) -> float:
“””
베타 계산

```
Args:
    stock_returns: 개별 주식 수익률
    market_returns: 시장 수익률
    
Returns:
    베타값
"""
if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
    return 1.0

# 공분산 / 시장분산
covariance = np.cov(stock_returns, market_returns)[0, 1]
market_variance = np.var(market_returns)

return covariance / market_variance if market_variance > 0 else 1.0
```

def calculate_tracking_error(portfolio_returns: List[float],
benchmark_returns: List[float]) -> float:
“””
추적오차 계산

```
Args:
    portfolio_returns: 포트폴리오 수익률
    benchmark_returns: 벤치마크 수익률
    
Returns:
    추적오차
"""
if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
    return 0.0

excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)
return np.std(excess_returns)
```

# 국가별/시장별 특화 함수들

def get_korean_market_characteristics() -> Dict[str, Any]:
“””
한국 시장 특성 반환
“””
return {
‘market_cap_tiers’: {
‘large_cap’: 1e13,      # 10조원 이상
‘mid_cap’: 1e12,        # 1조원 이상
‘small_cap’: 1e11       # 1000억원 이상
},
‘sector_weights’: {
‘반도체’: 0.25,
‘IT서비스’: 0.15,
‘금융’: 0.12,
‘화학’: 0.08,
‘자동차’: 0.08,
‘기타’: 0.32
},
‘trading_characteristics’: {
‘daily_volume_requirement’: 1e8,  # 일 거래대금 1억원
‘liquidity_threshold’: 0.005,     # 최소 유동성 비율
‘price_limit’: 0.30               # 가격제한폭 30%
},
‘corporate_governance’: {
‘chaebol_discount’: 0.15,         # 대기업집단 할인율
‘foreign_ownership_bonus’: 0.05,  # 외국인투자 보너스
‘dividend_tax_rate’: 0.154        # 배당세율 15.4%
}
}