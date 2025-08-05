“””
4대 거장 융합 엔진 - Master’s Eye
파일 위치: backend/app/masters/fusion.py

워렌 버핏, 레이 달리오, 리처드 파인만, 짐 사이먼스의 투자 철학을 융합하여
최적의 포트폴리오를 생성하는 핵심 엔진
“””

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import logging

from .buffett import WarrenBuffettAnalyzer
from .dalio import RayDalioAnalyzer
from .feynman import RichardFeynmanAnalyzer
from .simons import JimSimonsAnalyzer
from ..utils.calculations import calculate_portfolio_metrics, calculate_risk_adjusted_return

logger = logging.getLogger(**name**)

@dataclass
class MasterWeight:
“”“4대 거장 가중치 설정”””
buffett: float
dalio: float
feynman: float
simons: float

```
def __post_init__(self):
    """가중치 합이 1이 되도록 정규화"""
    total = self.buffett + self.dalio + self.feynman + self.simons
    if abs(total - 1.0) > 0.01:
        self.buffett /= total
        self.dalio /= total
        self.feynman /= total
        self.simons /= total
```

@dataclass
class PortfolioStrategy:
“”“포트폴리오 전략 설정”””
name: str
description: str
master_weights: MasterWeight
risk_tolerance: float  # 0.0 ~ 1.0
expected_return: float
max_volatility: float
max_drawdown: float
constraints: Dict[str, Any]

class MastersFusionEngine:
“”“4대 거장 융합 엔진”””

```
def __init__(self):
    self.buffett = WarrenBuffettAnalyzer()
    self.dalio = RayDalioAnalyzer()
    self.feynman = RichardFeynmanAnalyzer()
    self.simons = JimSimonsAnalyzer()
    
    # 전략별 사전 정의
    self.strategies = {
        "conservative": PortfolioStrategy(
            name="안전형",
            description="안정적인 수익 추구, 변동성 최소화",
            master_weights=MasterWeight(0.4, 0.4, 0.15, 0.05),
            risk_tolerance=0.3,
            expected_return=0.08,
            max_volatility=0.15,
            max_drawdown=0.10,
            constraints={
                "max_single_stock": 0.10,
                "max_sector": 0.30,
                "min_diversification": 15,
                "beta_limit": 0.7
            }
        ),
        "balanced": PortfolioStrategy(
            name="균형형",
            description="수익과 위험의 균형, 중장기 성장",
            master_weights=MasterWeight(0.3, 0.3, 0.2, 0.2),
            risk_tolerance=0.5,
            expected_return=0.11,
            max_volatility=0.20,
            max_drawdown=0.15,
            constraints={
                "max_single_stock": 0.15,
                "max_sector": 0.40,
                "min_diversification": 12,
                "beta_limit": 1.1
            }
        ),
        "aggressive": PortfolioStrategy(
            name="공격형",
            description="높은 수익 추구, 적극적 성장 전략",
            master_weights=MasterWeight(0.2, 0.2, 0.2, 0.4),
            risk_tolerance=0.8,
            expected_return=0.14,
            max_volatility=0.25,
            max_drawdown=0.20,
            constraints={
                "max_single_stock": 0.20,
                "max_sector": 0.50,
                "min_diversification": 10,
                "beta_limit": None
            }
        )
    }

async def analyze_stock_universe(self, tickers: List[str]) -> Dict[str, Dict]:
    """전체 종목에 대한 4대 거장 분석 실행"""
    results = {}
    
    for ticker in tickers:
        try:
            # 각 거장별 분석 수행
            buffett_score = await self.buffett.analyze_stock(ticker)
            dalio_score = await self.dalio.analyze_stock(ticker)
            feynman_score = await self.feynman.analyze_stock(ticker)
            simons_score = await self.simons.analyze_stock(ticker)
            
            results[ticker] = {
                "buffett": buffett_score,
                "dalio": dalio_score,
                "feynman": feynman_score,
                "simons": simons_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            continue
    
    return results

def calculate_fusion_score(self, 
                         analysis_results: Dict[str, Dict], 
                         master_weights: MasterWeight) -> Dict[str, float]:
    """융합 점수 계산"""
    fusion_scores = {}
    
    for ticker, scores in analysis_results.items():
        try:
            # 각 거장의 점수에 가중치 적용
            fusion_score = (
                scores["buffett"]["total_score"] * master_weights.buffett +
                scores["dalio"]["total_score"] * master_weights.dalio +
                scores["feynman"]["total_score"] * master_weights.feynman +
                scores["simons"]["total_score"] * master_weights.simons
            )
            
            fusion_scores[ticker] = fusion_score
            
        except (KeyError, TypeError) as e:
            logger.warning(f"Error calculating fusion score for {ticker}: {e}")
            fusion_scores[ticker] = 0.0
    
    return fusion_scores

def optimize_portfolio_weights(self, 
                             candidate_stocks: List[str],
                             fusion_scores: Dict[str, float],
                             strategy: PortfolioStrategy,
                             market_data: pd.DataFrame) -> Dict[str, float]:
    """포트폴리오 가중치 최적화"""
    
    # 상위 종목 선별 (융합 점수 기준)
    sorted_stocks = sorted(
        candidate_stocks, 
        key=lambda x: fusion_scores.get(x, 0), 
        reverse=True
    )
    
    # 다양성 제약 고려하여 선별
    selected_stocks = self._select_diversified_stocks(
        sorted_stocks[:30], market_data, strategy
    )
    
    # 평균-분산 최적화
    returns = market_data[selected_stocks].pct_change().dropna()
    mean_returns = returns.mean() * 252  # 연환산
    cov_matrix = returns.cov() * 252
    
    # 목적함수: 샤프 비율 최대화
    def objective(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_vol  # 음수로 변환 (최대화 → 최소화)
    
    # 제약조건
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # 가중치 합 = 1
    ]
    
    # 경계조건
    bounds = []
    max_weight = strategy.constraints.get("max_single_stock", 0.2)
    for _ in selected_stocks:
        bounds.append((0.01, max_weight))
    
    # 초기 가중치 (동일 가중)
    x0 = np.array([1.0 / len(selected_stocks)] * len(selected_stocks))
    
    # 최적화 실행
    result = minimize(
        objective, x0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if result.success:
        return dict(zip(selected_stocks, result.x))
    else:
        logger.warning("Optimization failed, using equal weights")
        equal_weight = 1.0 / len(selected_stocks)
        return {stock: equal_weight for stock in selected_stocks}

def _select_diversified_stocks(self, 
                             candidates: List[str], 
                             market_data: pd.DataFrame,
                             strategy: PortfolioStrategy) -> List[str]:
    """섹터/테마 다양성을 고려한 종목 선별"""
    
    # 섹터 정보 로드 (임시 구현)
    sector_info = self._get_sector_info(candidates)
    
    selected = []
    sector_counts = {}
    max_per_sector = max(2, len(candidates) // 5)  # 섹터당 최대 종목 수
    
    for stock in candidates:
        sector = sector_info.get(stock, "기타")
        
        if sector_counts.get(sector, 0) < max_per_sector:
            selected.append(stock)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            if len(selected) >= strategy.constraints.get("min_diversification", 15):
                break
    
    return selected

def _get_sector_info(self, tickers: List[str]) -> Dict[str, str]:
    """종목별 섹터 정보 조회 (임시 구현)"""
    # 실제 구현에서는 데이터베이스에서 조회
    sector_mapping = {
        "005930": "반도체", "000660": "음료", "035420": "통신",
        "005380": "자동차", "055550": "통신", "035720": "화학",
        # ... 더 많은 매핑
    }
    return {ticker: sector_mapping.get(ticker, "기타") for ticker in tickers}

async def generate_portfolio(self, 
                           strategy_type: str,
                           investment_amount: float,
                           custom_weights: Optional[MasterWeight] = None,
                           excluded_stocks: Optional[List[str]] = None) -> Dict[str, Any]:
    """포트폴리오 생성 메인 메서드"""
    
    try:
        # 전략 설정
        strategy = self.strategies.get(strategy_type)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_type}")
        
        # 커스텀 가중치 적용
        if custom_weights:
            strategy.master_weights = custom_weights
        
        # 종목 유니버스 조회
        universe = await self._get_stock_universe(excluded_stocks)
        
        # 4대 거장 분석 실행
        analysis_results = await self.analyze_stock_universe(universe)
        
        # 융합 점수 계산
        fusion_scores = self.calculate_fusion_score(
            analysis_results, strategy.master_weights
        )
        
        # 시장 데이터 조회
        market_data = await self._get_market_data(universe)
        
        # 포트폴리오 최적화
        optimal_weights = self.optimize_portfolio_weights(
            universe, fusion_scores, strategy, market_data
        )
        
        # 포지션 계산
        positions = self._calculate_positions(
            optimal_weights, investment_amount, market_data
        )
        
        # 백테스팅 실행
        backtest_results = await self._run_backtest(optimal_weights, market_data)
        
        # 결과 구성
        portfolio = {
            "strategy": strategy_type,
            "master_weights": {
                "buffett": strategy.master_weights.buffett,
                "dalio": strategy.master_weights.dalio,
                "feynman": strategy.master_weights.feynman,
                "simons": strategy.master_weights.simons
            },
            "positions": positions,
            "expected_metrics": {
                "return": strategy.expected_return,
                "volatility": strategy.max_volatility,
                "sharpe_ratio": strategy.expected_return / strategy.max_volatility
            },
            "backtest_results": backtest_results,
            "analysis_details": analysis_results,
            "fusion_scores": fusion_scores,
            "total_amount": investment_amount,
            "cash_remaining": investment_amount - sum(p["amount"] for p in positions.values())
        }
        
        logger.info(f"Portfolio generated: {len(positions)} positions, "
                   f"total allocation: {sum(p['weight'] for p in positions.values()):.2%}")
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Error generating portfolio: {e}")
        raise

async def _get_stock_universe(self, excluded_stocks: Optional[List[str]] = None) -> List[str]:
    """투자 유니버스 조회"""
    # 실제 구현에서는 데이터베이스에서 조회
    # KOSPI 200 + 우량 중소형주
    base_universe = [
        "005930", "000660", "035420", "005380", "055550", "035720",
        "005490", "051910", "006400", "028260", "105560", "096770",
        # ... 더 많은 종목
    ]
    
    if excluded_stocks:
        base_universe = [s for s in base_universe if s not in excluded_stocks]
    
    return base_universe

async def _get_market_data(self, tickers: List[str]) -> pd.DataFrame:
    """시장 데이터 조회"""
    # 실제 구현에서는 데이터베이스에서 조회
    # 임시로 더미 데이터 생성
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)
    
    data = {}
    for ticker in tickers:
        # 임시 주가 데이터 생성
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 50000 * np.exp(np.cumsum(returns))  # 5만원 시작
        data[ticker] = prices
    
    return pd.DataFrame(data, index=dates)

def _calculate_positions(self, 
                       weights: Dict[str, float], 
                       total_amount: float,
                       market_data: pd.DataFrame) -> Dict[str, Dict]:
    """포지션 계산"""
    positions = {}
    current_prices = market_data.iloc[-1]  # 최신 가격
    
    for ticker, weight in weights.items():
        if weight > 0:
            amount = total_amount * weight
            price = current_prices[ticker]
            shares = int(amount / price)
            actual_amount = shares * price
            
            positions[ticker] = {
                "weight": weight,
                "target_amount": amount,
                "actual_amount": actual_amount,
                "shares": shares,
                "price": price,
                "ticker": ticker
            }
    
    return positions

async def _run_backtest(self, 
                      weights: Dict[str, float], 
                      market_data: pd.DataFrame) -> Dict[str, Any]:
    """백테스팅 실행"""
    # 간단한 백테스팅 구현
    returns = market_data.pct_change().dropna()
    
    # 포트폴리오 수익률 계산
    portfolio_returns = returns @ pd.Series(weights)
    
    # 성과 지표 계산
    total_return = (1 + portfolio_returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # 최대 낙폭 계산
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "period_start": market_data.index[0],
        "period_end": market_data.index[-1],
        "total_days": len(portfolio_returns),
        "win_rate": (portfolio_returns > 0).mean()
    }

def explain_portfolio_decision(self, 
                             portfolio: Dict[str, Any], 
                             top_n: int = 5) -> Dict[str, Any]:
    """포트폴리오 구성 근거 설명"""
    
    explanation = {
        "strategy_rationale": self._explain_strategy(portfolio["strategy"]),
        "master_contributions": self._explain_master_contributions(portfolio),
        "top_picks": self._explain_top_picks(portfolio, top_n),
        "risk_profile": self._explain_risk_profile(portfolio),
        "expected_scenarios": self._generate_scenarios(portfolio)
    }
    
    return explanation

def _explain_strategy(self, strategy_type: str) -> str:
    """전략 선택 근거 설명"""
    strategy = self.strategies[strategy_type]
    return {
        "name": strategy.name,
        "description": strategy.description,
        "key_features": {
            "expected_return": f"{strategy.expected_return:.1%}",
            "max_volatility": f"{strategy.max_volatility:.1%}",
            "max_drawdown": f"{strategy.max_drawdown:.1%}"
        }
    }

def _explain_master_contributions(self, portfolio: Dict[str, Any]) -> Dict[str, str]:
    """각 거장의 기여도 설명"""
    weights = portfolio["master_weights"]
    
    return {
        "buffett": f"가치 투자({weights['buffett']:.1%}): 저평가된 우량기업 발굴",
        "dalio": f"거시경제 분석({weights['dalio']:.1%}): 경제 사이클 대응",
        "feynman": f"과학적 검증({weights['feynman']:.1%}): 리스크 정량화",
        "simons": f"퀀트 분석({weights['simons']:.1%}): 수학적 패턴 활용"
    }

def _explain_top_picks(self, portfolio: Dict[str, Any], top_n: int) -> List[Dict]:
    """상위 종목 선택 근거"""
    positions = portfolio["positions"]
    sorted_positions = sorted(
        positions.items(), 
        key=lambda x: x[1]["weight"], 
        reverse=True
    )[:top_n]
    
    explanations = []
    for ticker, position in sorted_positions:
        fusion_score = portfolio["fusion_scores"].get(ticker, 0)
        analysis = portfolio["analysis_details"].get(ticker, {})
        
        explanations.append({
            "ticker": ticker,
            "weight": position["weight"],
            "fusion_score": fusion_score,
            "key_strengths": self._get_key_strengths(analysis),
            "rationale": f"융합점수 {fusion_score:.2f}점으로 상위 선정"
        })
    
    return explanations

def _get_key_strengths(self, analysis: Dict) -> List[str]:
    """종목별 핵심 강점 추출"""
    strengths = []
    
    # 각 거장별 강점 요약
    if analysis.get("buffett", {}).get("intrinsic_value_score", 0) > 0.7:
        strengths.append("저평가 우량주")
    
    if analysis.get("dalio", {}).get("macro_score", 0) > 0.7:
        strengths.append("거시경제 수혜")
    
    if analysis.get("feynman", {}).get("uncertainty_score", 0) > 0.7:
        strengths.append("예측 가능성 높음")
    
    if analysis.get("simons", {}).get("momentum_score", 0) > 0.7:
        strengths.append("기술적 모멘텀")
    
    return strengths or ["종합 점수 우수"]

def _explain_risk_profile(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """리스크 프로필 설명"""
    backtest = portfolio["backtest_results"]
    
    return {
        "volatility_level": self._categorize_volatility(backtest["volatility"]),
        "drawdown_tolerance": self._categorize_drawdown(backtest["max_drawdown"]),
        "return_expectation": self._categorize_return(backtest["annual_return"]),
        "overall_risk": self._assess_overall_risk(portfolio)
    }

def _categorize_volatility(self, vol: float) -> str:
    """변동성 수준 분류"""
    if vol < 0.15:
        return "낮음 (안정적)"
    elif vol < 0.25:
        return "보통 (표준적)"
    else:
        return "높음 (적극적)"

def _categorize_drawdown(self, dd: float) -> str:
    """낙폭 허용도 분류"""
    dd = abs(dd)
    if dd < 0.10:
        return "보수적 (10% 이하)"
    elif dd < 0.20:
        return "보통 (20% 이하)"
    else:
        return "적극적 (20% 초과)"

def _categorize_return(self, ret: float) -> str:
    """수익률 기대치 분류"""
    if ret < 0.08:
        return "안정적 (8% 이하)"
    elif ret < 0.12:
        return "균형적 (8-12%)"
    else:
        return "성장 지향 (12% 이상)"

def _assess_overall_risk(self, portfolio: Dict[str, Any]) -> str:
    """종합 리스크 평가"""
    strategy = portfolio["strategy"]
    return self.strategies[strategy].description

def _generate_scenarios(self, portfolio: Dict[str, Any]) -> Dict[str, Dict]:
    """시나리오별 예상 성과"""
    base_return = portfolio["expected_metrics"]["return"]
    volatility = portfolio["expected_metrics"]["volatility"]
    
    return {
        "bull_market": {
            "probability": 0.3,
            "expected_return": base_return + volatility,
            "description": "호황장 (금리 하락, 경기 회복)"
        },
        "normal_market": {
            "probability": 0.4,
            "expected_return": base_return,
            "description": "평상장 (현재 추세 지속)"
        },
        "bear_market": {
            "probability": 0.3,
            "expected_return": base_return - volatility,
            "description": "약세장 (금리 상승, 경기 둔화)"
        }
    }
```