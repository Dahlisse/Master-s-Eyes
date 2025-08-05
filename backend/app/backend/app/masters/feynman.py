“””
리처드 파인만 과학적 사고 알고리즘 구현

- First Principle 사고법
- 몬테카를로 시뮬레이션
- 베이지안 추론
- 불확실성 정량화
- 지적 정직성 (Intellectual Honesty)
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
import warnings
from concurrent.futures import ProcessPoolExecutor
import numba

from app.masters.base import BaseMaster, MasterScore, PortfolioRecommendation
from app.core.logging import get_logger

logger = get_logger(**name**)

@dataclass
class FeynmanScore:
“”“파인만 평가 점수 데이터 클래스”””
total_score: float
understanding_score: float
uncertainty_score: float
simplicity_score: float
probability_score: float
monte_carlo_confidence: float
bayesian_probability: float
expected_scenarios: Dict[str, float]
confidence_interval: Tuple[float, float]
intellectual_honesty: float

```
def to_dict(self) -> Dict:
    return {
        'total_score': self.total_score,
        'understanding_score': self.understanding_score,
        'uncertainty_score': self.uncertainty_score,
        'simplicity_score': self.simplicity_score,
        'probability_score': self.probability_score,
        'monte_carlo_confidence': self.monte_carlo_confidence,
        'bayesian_probability': self.bayesian_probability,
        'expected_scenarios': self.expected_scenarios,
        'confidence_interval': self.confidence_interval,
        'intellectual_honesty': self.intellectual_honesty
    }
```

@dataclass
class MonteCarloResult:
“”“몬테카를로 시뮬레이션 결과”””
scenarios: np.ndarray
mean_return: float
volatility: float
var_95: float
var_99: float
expected_shortfall: float
probability_positive: float
confidence_interval: Tuple[float, float]
simulation_count: int

```
def to_dict(self) -> Dict:
    return {
        'mean_return': self.mean_return,
        'volatility': self.volatility,
        'var_95': self.var_95,
        'var_99': self.var_99,
        'expected_shortfall': self.expected_shortfall,
        'probability_positive': self.probability_positive,
        'confidence_interval': self.confidence_interval,
        'simulation_count': self.simulation_count
    }
```

class FeynmanScientificInvestor(BaseMaster):
“””
리처드 파인만의 과학적 사고를 구현한 투자 클래스

```
핵심 원칙:
1. First Principle 사고법 - 근본부터 이해
2. 지적 정직성 - 모르는 것은 모른다고 인정
3. 불확실성 정량화 - 확률적 사고
4. 단순함 추구 - 복잡함을 단순하게 분해
"""

def __init__(self, db: Session):
    super().__init__(db, "Richard Feynman")
    self.simulation_count = 100_000  # 몬테카를로 시뮬레이션 횟수
    self.confidence_level = 0.95     # 신뢰수준
    self.min_understanding_threshold = 0.7  # 최소 이해도 임계값
    
async def evaluate_stock(self, ticker: str) -> Optional[FeynmanScore]:
    """
    파인만 스타일 종목 평가
    
    Args:
        ticker: 종목 코드
        
    Returns:
        FeynmanScore 객체 또는 None
    """
    try:
        if not await self.validate_ticker(ticker):
            return None
        
        logger.info(f"Starting Feynman evaluation for {ticker}")
        
        # 1. First Principle 이해도 평가
        understanding_score = await self._evaluate_first_principle_understanding(ticker)
        
        # 2. 지적 정직성 평가 (우리가 정말 이해하고 있는가?)
        intellectual_honesty = self._evaluate_intellectual_honesty(ticker)
        
        # 3. 불확실성 정량화
        uncertainty_metrics = await self._quantify_uncertainty(ticker)
        
        # 4. 단순성 평가 (복잡한 것을 단순하게 설명할 수 있는가?)
        simplicity_score = await self._evaluate_simplicity(ticker)
        
        # 5. 몬테카를로 시뮬레이션
        monte_carlo_result = await self._run_monte_carlo_simulation(ticker)
        
        # 6. 베이지안 추론
        bayesian_probability = await self._bayesian_inference(ticker)
        
        # 7. 시나리오 분석
        scenario_analysis = await self._scenario_analysis(ticker)
        
        # 8. 종합 평가
        total_score = self._calculate_feynman_total_score(
            understanding_score, intellectual_honesty, uncertainty_metrics,
            simplicity_score, monte_carlo_result, bayesian_probability
        )
        
        return FeynmanScore(
            total_score=total_score,
            understanding_score=understanding_score,
            uncertainty_score=uncertainty_metrics['uncertainty_score'],
            simplicity_score=simplicity_score,
            probability_score=uncertainty_metrics['probability_score'],
            monte_carlo_confidence=monte_carlo_result.confidence_interval[1] - monte_carlo_result.confidence_interval[0],
            bayesian_probability=bayesian_probability,
            expected_scenarios=scenario_analysis,
            confidence_interval=monte_carlo_result.confidence_interval,
            intellectual_honesty=intellectual_honesty
        )
        
    except Exception as e:
        logger.error(f"Error in Feynman evaluation for {ticker}: {str(e)}")
        return None

async def _evaluate_first_principle_understanding(self, ticker: str) -> float:
    """
    First Principle 이해도 평가
    
    핵심 질문:
    - 이 기업이 돈을 버는 근본적인 이유는 무엇인가?
    - 가치 창출의 물리적/경제적 메커니즘은 무엇인가?
    - 모든 가정을 제거하면 남는 것은 무엇인가?
    """
    try:
        # 기업 기본 정보 수집
        business_model_clarity = await self._analyze_business_model_clarity(ticker)
        value_creation_mechanism = await self._analyze_value_creation(ticker)
        fundamental_drivers = await self._identify_fundamental_drivers(ticker)
        
        # First Principle 체크리스트
        understanding_factors = {
            'business_model_clarity': business_model_clarity,      # 사업모델 명확성
            'value_creation_mechanism': value_creation_mechanism, # 가치 창출 메커니즘
            'fundamental_drivers': fundamental_drivers,           # 근본적 동인
            'predictability': await self._assess_predictability(ticker),  # 예측 가능성
            'moat_sustainability': await self._assess_moat_sustainability(ticker)  # 해자 지속성
        }
        
        # 가중 평균 계산
        weights = {
            'business_model_clarity': 0.25,
            'value_creation_mechanism': 0.25,
            'fundamental_drivers': 0.20,
            'predictability': 0.15,
            'moat_sustainability': 0.15
        }
        
        weighted_score = sum(
            understanding_factors[factor] * weight 
            for factor, weight in weights.items()
        )
        
        logger.info(f"First Principle understanding for {ticker}: {weighted_score:.2f}")
        return min(100.0, weighted_score)
        
    except Exception as e:
        logger.warning(f"Error in first principle evaluation for {ticker}: {str(e)}")
        return 50.0  # 중간값

async def _analyze_business_model_clarity(self, ticker: str) -> float:
    """사업 모델 명확성 분석"""
    try:
        # 매출 구조 분석
        revenue_concentration = await self._get_revenue_concentration(ticker)
        product_complexity = await self._assess_product_complexity(ticker)
        customer_dependency = await self._analyze_customer_dependency(ticker)
        
        # 명확성 점수 계산
        clarity_score = 0.0
        
        # 매출 집중도 (높을수록 단순함)
        if revenue_concentration > 0.7:
            clarity_score += 30
        elif revenue_concentration > 0.5:
            clarity_score += 20
        else:
            clarity_score += 10
        
        # 제품 복잡도 (낮을수록 이해하기 쉬움)
        if product_complexity < 0.3:
            clarity_score += 25
        elif product_complexity < 0.6:
            clarity_score += 15
        else:
            clarity_score += 5
        
        # 고객 의존도 (적당한 수준이 좋음)
        if 0.1 <= customer_dependency <= 0.3:
            clarity_score += 25
        elif customer_dependency < 0.5:
            clarity_score += 15
        else:
            clarity_score += 5
        
        # 업종별 보정
        sector_clarity = await self._get_sector_clarity_score(ticker)
        clarity_score += sector_clarity * 20
        
        return min(100.0, clarity_score)
        
    except Exception as e:
        logger.warning(f"Business model clarity analysis failed for {ticker}: {str(e)}")
        return 50.0

async def _analyze_value_creation(self, ticker: str) -> float:
    """가치 창출 메커니즘 분석"""
    try:
        # 경제적 부가가치 분석
        roic = await self._calculate_roic(ticker)
        wacc = await self._calculate_wacc(ticker)
        
        # EVA (Economic Value Added)
        eva_spread = roic - wacc
        
        # 가치 창출 지속성
        eva_consistency = await self._analyze_eva_consistency(ticker)
        
        # 자본 효율성
        capital_efficiency = await self._analyze_capital_efficiency(ticker)
        
        # 현금 창출 능력
        cash_conversion = await self._analyze_cash_conversion(ticker)
        
        # 종합 점수
        value_score = 0.0
        
        # EVA 스프레드 평가
        if eva_spread > 0.05:  # 5% 이상
            value_score += 30
        elif eva_spread > 0:
            value_score += 20
        else:
            value_score += 5
        
        # 일관성 평가
        value_score += eva_consistency * 25
        
        # 자본 효율성
        value_score += capital_efficiency * 25
        
        # 현금 전환 능력
        value_score += cash_conversion * 20
        
        return min(100.0, value_score)
        
    except Exception as e:
        logger.warning(f"Value creation analysis failed for {ticker}: {str(e)}")
        return 50.0

async def _quantify_uncertainty(self, ticker: str) -> Dict[str, float]:
    """불확실성 정량화"""
    try:
        # 다양한 불확실성 요소 측정
        earnings_volatility = await self._calculate_earnings_volatility(ticker)
        revenue_predictability = await self._calculate_revenue_predictability(ticker)
        margin_stability = await self._calculate_margin_stability(ticker)
        external_dependency = await self._assess_external_dependency(ticker)
        
        # 불확실성 점수 계산 (낮을수록 좋음)
        uncertainty_factors = [
            earnings_volatility,
            1 - revenue_predictability,  # 예측가능성의 역수
            1 - margin_stability,
            external_dependency
        ]
        
        avg_uncertainty = np.mean(uncertainty_factors)
        uncertainty_score = max(0, 100 - avg_uncertainty * 100)
        
        # 확률적 평가
        probability_of_success = await self._calculate_success_probability(ticker)
        
        return {
            'uncertainty_score': uncertainty_score,
            'probability_score': probability_of_success * 100,
            'earnings_volatility': earnings_volatility,
            'revenue_predictability': revenue_predictability,
            'margin_stability': margin_stability
        }
        
    except Exception as e:
        logger.warning(f"Uncertainty quantification failed for {ticker}: {str(e)}")
        return {
            'uncertainty_score': 50.0,
            'probability_score': 50.0,
            'earnings_volatility': 0.5,
            'revenue_predictability': 0.5,
            'margin_stability': 0.5
        }

def _evaluate_intellectual_honesty(self, ticker: str) -> float:
    """
    지적 정직성 평가
    
    파인만의 핵심 원칙: "나는 이것을 정말 이해하고 있는가?"
    """
    try:
        honesty_score = 100.0
        
        # 복잡성 페널티 (너무 복잡하면 이해하기 어려움)
        complexity_penalty = self._calculate_complexity_penalty(ticker)
        honesty_score -= complexity_penalty
        
        # 데이터 품질 확인
        data_quality = self._assess_data_quality(ticker)
        if data_quality < 0.8:
            honesty_score -= (0.8 - data_quality) * 50
        
        # 예측 한계 인정
        prediction_confidence = self._assess_prediction_confidence(ticker)
        if prediction_confidence > 0.9:  # 과신은 위험
            honesty_score -= (prediction_confidence - 0.9) * 100
        
        # 가정의 명시성
        assumptions_clarity = self._evaluate_assumptions_clarity(ticker)
        honesty_score += assumptions_clarity * 20
        
        return max(0.0, min(100.0, honesty_score))
        
    except Exception as e:
        logger.warning(f"Intellectual honesty evaluation failed for {ticker}: {str(e)}")
        return 70.0  # 보수적 추정

async def _run_monte_carlo_simulation(self, ticker: str) -> MonteCarloResult:
    """
    몬테카를로 시뮬레이션 실행
    
    다양한 시나리오에서의 수익률 분포 계산
    """
    try:
        logger.info(f"Running Monte Carlo simulation for {ticker} ({self.simulation_count:,} iterations)")
        
        # 기본 파라미터 수집
        historical_returns = await self._get_historical_returns(ticker)
        volatility_estimate = await self._estimate_future_volatility(ticker)
        expected_return = await self._estimate_expected_return(ticker)
        
        # 시뮬레이션 실행 (병렬 처리)
        scenarios = await self._monte_carlo_parallel(
            expected_return, volatility_estimate, self.simulation_count
        )
        
        # 결과 분석
        mean_return = np.mean(scenarios)
        volatility = np.std(scenarios)
        
        # 리스크 지표 계산
        var_95 = np.percentile(scenarios, 5)
        var_99 = np.percentile(scenarios, 1)
        
        # Expected Shortfall (CVaR)
        expected_shortfall = np.mean(scenarios[scenarios <= var_95])
        
        # 양의 수익률 확률
        probability_positive = np.sum(scenarios > 0) / len(scenarios)
        
        # 신뢰구간
        confidence_interval = (
            np.percentile(scenarios, (1 - self.confidence_level) / 2 * 100),
            np.percentile(scenarios, (1 + self.confidence_level) / 2 * 100)
        )
        
        return MonteCarloResult(
            scenarios=scenarios,
            mean_return=mean_return,
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            probability_positive=probability_positive,
            confidence_interval=confidence_interval,
            simulation_count=self.simulation_count
        )
        
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed for {ticker}: {str(e)}")
        # 기본값 반환
        return MonteCarloResult(
            scenarios=np.array([0.0]),
            mean_return=0.0,
            volatility=0.2,
            var_95=-0.15,
            var_99=-0.25,
            expected_shortfall=-0.20,
            probability_positive=0.5,
            confidence_interval=(-0.15, 0.15),
            simulation_count=0
        )

@numba.jit(nopython=True)
def _monte_carlo_simulation_numba(self, expected_return: float, volatility: float, 
                                n_simulations: int, random_seed: int = None) -> np.ndarray:
    """Numba 최적화된 몬테카를로 시뮬레이션"""
    if random_seed:
        np.random.seed(random_seed)
    
    # 정규분포 난수 생성
    random_shocks = np.random.normal(0, 1, n_simulations)
    
    # 수익률 시나리오 계산
    scenarios = expected_return + volatility * random_shocks
    
    return scenarios

async def _monte_carlo_parallel(self, expected_return: float, volatility: float, 
                              n_simulations: int) -> np.ndarray:
    """병렬 처리를 통한 몬테카를로 시뮬레이션"""
    try:
        # CPU 코어 수에 따라 작업 분할
        import multiprocessing
        n_cores = min(4, multiprocessing.cpu_count())  # 최대 4코어 사용
        
        chunk_size = n_simulations // n_cores
        
        # 각 프로세스별 시드 설정
        seeds = [42 + i for i in range(n_cores)]
        
        # 병렬 실행
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [
                executor.submit(
                    self._monte_carlo_simulation_numba,
                    expected_return, volatility, chunk_size, seed
                )
                for seed in seeds
            ]
            
            # 결과 수집
            results = []
            for future in futures:
                results.append(future.result())
        
        # 결과 병합
        all_scenarios = np.concatenate(results)
        
        return all_scenarios
        
    except Exception as e:
        logger.warning(f"Parallel Monte Carlo failed, using single thread: {str(e)}")
        # 단일 스레드로 fallback
        return self._monte_carlo_simulation_numba(expected_return, volatility, n_simulations)

async def _bayesian_inference(self, ticker: str) -> float:
    """
    베이지안 추론을 통한 투자 확률 계산
    
    사전 확률 + 새로운 증거 = 사후 확률
    """
    try:
        # 사전 확률 (Prior) - 업종 평균 성공률
        sector_prior = await self._get_sector_success_rate(ticker)
        
        # 우도 (Likelihood) - 현재 기업의 증거들
        fundamental_evidence = await self._collect_fundamental_evidence(ticker)
        technical_evidence = await self._collect_technical_evidence(ticker)
        market_evidence = await self._collect_market_evidence(ticker)
        
        # 베이지안 업데이트
        posterior_probability = self._bayesian_update(
            sector_prior,
            [fundamental_evidence, technical_evidence, market_evidence]
        )
        
        logger.info(f"Bayesian probability for {ticker}: {posterior_probability:.3f}")
        return posterior_probability
        
    except Exception as e:
        logger.warning(f"Bayesian inference failed for {ticker}: {str(e)}")
        return 0.5  # 중립 확률

def _bayesian_update(self, prior: float, evidences: List[float]) -> float:
    """베이지안 업데이트 수행"""
    try:
        # 연속적 베이지안 업데이트
        posterior = prior
        
        for evidence in evidences:
            # 베이지안 정리: P(H|E) = P(E|H) * P(H) / P(E)
            # 단순화된 형태로 구현
            likelihood_ratio = evidence / (1 - evidence) if evidence != 1 else 10
            prior_odds = posterior / (1 - posterior) if posterior != 1 else 10
            
            posterior_odds = likelihood_ratio * prior_odds
            posterior = posterior_odds / (1 + posterior_odds)
            
            # 극값 방지
            posterior = max(0.01, min(0.99, posterior))
        
        return posterior
        
    except Exception as e:
        logger.warning(f"Bayesian update failed: {str(e)}")
        return prior

async def _scenario_analysis(self, ticker: str) -> Dict[str, float]:
    """시나리오 분석"""
    try:
        scenarios = {
            'optimistic': await self._calculate_optimistic_scenario(ticker),
            'base_case': await self._calculate_base_case_scenario(ticker),
            'pessimistic': await self._calculate_pessimistic_scenario(ticker),
            'black_swan': await self._calculate_black_swan_scenario(ticker)
        }
        
        # 시나리오별 확률 가중치
        probabilities = {
            'optimistic': 0.2,
            'base_case': 0.5,
            'pessimistic': 0.25,
            'black_swan': 0.05
        }
        
        # 기댓값 계산
        expected_value = sum(
            scenarios[scenario] * probabilities[scenario]
            for scenario in scenarios
        )
        
        scenarios['expected_value'] = expected_value
        scenarios['probabilities'] = probabilities
        
        return scenarios
        
    except Exception as e:
        logger.warning(f"Scenario analysis failed for {ticker}: {str(e)}")
        return {
            'optimistic': 0.15,
            'base_case': 0.08,
            'pessimistic': -0.05,
            'black_swan': -0.30,
            'expected_value': 0.06
        }

def _calculate_feynman_total_score(self, understanding: float, honesty: float,
                                 uncertainty: Dict, simplicity: float,
                                 monte_carlo: MonteCarloResult, bayesian: float) -> float:
    """파인만 스타일 종합 점수 계산"""
    try:
        # 가중치 설정
        weights = {
            'understanding': 0.25,      # 이해도
            'honesty': 0.20,           # 지적 정직성
            'uncertainty': 0.20,       # 불확실성 관리
            'simplicity': 0.15,        # 단순성
            'monte_carlo': 0.10,       # 몬테카를로 신뢰도
            'bayesian': 0.10          # 베이지안 확률
        }
        
        # 각 요소별 점수
        scores = {
            'understanding': understanding,
            'honesty': honesty,
            'uncertainty': uncertainty['uncertainty_score'],
            'simplicity': simplicity,
            'monte_carlo': min(100, monte_carlo.probability_positive * 100),
            'bayesian': bayesian * 100
        }
        
        # 가중 평균
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        # 이해도가 임계값 미만이면 페널티
        if understanding < self.min_understanding_threshold * 100:
            understanding_penalty = (self.min_understanding_threshold * 100 - understanding) * 0.5
            total_score -= understanding_penalty
        
        return max(0.0, min(100.0, total_score))
        
    except Exception as e:
        logger.warning(f"Total score calculation failed: {str(e)}")
        return 50.0

# 헬퍼 메서드들 (간소화된 구현)
async def _get_revenue_concentration(self, ticker: str) -> float:
    """매출 집중도 계산"""
    # 실제 구현에서는 사업보고서 분석
    return 0.7  # 70% 집중도

async def _assess_product_complexity(self, ticker: str) -> float:
    """제품 복잡도 평가"""
    # 업종별 복잡도 매핑
    complexity_map = {
        '반도체': 0.8,
        'IT서비스': 0.6,
        '금융': 0.4,
        '식품': 0.2,
        '유틸리티': 0.1
    }
    sector = await self._get_ticker_sector(ticker)
    return complexity_map.get(sector, 0.5)

async def _analyze_customer_dependency(self, ticker: str) -> float:
    """고객 의존도 분석"""
    # 간단한 추정치
    return 0.25

async def _get_sector_clarity_score(self, ticker: str) -> float:
    """업종별 명확성 점수"""
    clarity_map = {
        '유틸리티': 1.0,
        '금융': 0.8,
        '식품': 0.8,
        'IT서비스': 0.6,
        '반도체': 0.4
    }
    sector = await self._get_ticker_sector(ticker)
    return clarity_map.get(sector, 0.6)

async def _calculate_roic(self, ticker: str) -> float:
    """투하자본수익률 계산"""
    return 0.12  # 12%

async def _calculate_wacc(self, ticker: str) -> float:
    """가중평균자본비용 계산"""
    return 0.08  # 8%

async def _get_ticker_sector(self, ticker: str) -> str:
    """종목 섹터 조회"""
    sector_map = {
        '005930': '반도체',
        '055550': '금융',
        '035420': 'IT서비스'
    }
    return sector_map.get(ticker, '기타')

# 추가 헬퍼 메서드들은 실제 데이터에 맞게 구현
```

async def create_feynman_portfolio(available_tickers: List[str], db: Session,
target_allocation: float = 0.2) -> Dict:
“””
파인만 스타일 포트폴리오 생성

```
Args:
    available_tickers: 사용 가능한 종목 리스트
    db: 데이터베이스 세션
    target_allocation: 목표 비중
    
Returns:
    포트폴리오 구성 결과
"""
try:
    feynman_investor = FeynmanScientificInvestor(db)
    
    # 모든 종목에 대해 파인만 평가 수행
    evaluations = []
    
    for ticker in available_tickers:
        try:
            score = await feynman_investor.evaluate_stock(ticker)
            
            if score and score.total_score > 65:  # 65점 이상만 선택
                evaluations.append({
                    'ticker': ticker,
                    'score': score.total_score,
                    'understanding': score.understanding_score,
                    'uncertainty': score.uncertainty_score,
                    'monte_carlo_confidence': score.monte_carlo_confidence,
                    'bayesian_probability': score.bayesian_probability,
                    'details': score.to_dict()
                })
                
        except Exception as e:
            logger.warning(f"Feynman evaluation failed for {ticker}: {str(e)}")
            continue
    
    if not evaluations:
        logger.warning("No stocks passed Feynman evaluation")
        return {}
    
    # 이해도와 불확실성을 고려한 정렬
    evaluations.sort(
        key=lambda x: (x['understanding'] * 0.6 + (100 - x['uncertainty']) * 0.4),
        reverse=True
    )
    
    # 상위 종목 선택 (최대 12개)
    max_holdings = min(12, len(evaluations))
    selected_stocks = evaluations[:max_holdings]
    
    # 불확실성 기반 가중치 계산
    portfolio = {}
    total_weight = 0.0
    
    for stock in selected_stocks:
        # 이해도가 높고 불확실성이 낮을수록 높은 비중
        understanding_factor = stock['understanding'] / 100
        uncertainty_factor = (100 - stock['uncertainty']) / 100
        confidence_factor = (1 - stock['monte_carlo_confidence']) if stock['monte_carlo_confidence'] < 1 else 0.5
        
        # 복합 점수
        composite_score = (
            understanding_factor * 0.4 +
            uncertainty_factor * 0.3 +
            confidence_factor * 0.2 +
            stock['bayesian_probability'] * 0.1
        )
        
        weight = composite_score * target_allocation
        total_weight += weight
        
        portfolio[stock['ticker']] = {
            'weight': weight,
            'feynman_score': stock['score'],
            'understanding_score': stock['understanding'],
            'uncertainty_score': stock['uncertainty'],
            'monte_carlo_confidence': stock['monte_carlo_confidence'],
            'bayesian_probability': stock['bayesian_probability'],
            'reasoning': f"파인만 {stock['score']:.1f}점 - 이해도 {stock['understanding']:.1f}, 불확실성 관리 {100-stock['uncertainty']:.1f}"
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
        'average_understanding': np.mean([s['understanding'] for s in selected_stocks]),
        'average_uncertainty': np.mean([s['uncertainty'] for s in selected_stocks]),
        'strategy': 'Richard Feynman Scientific Thinking',
        'philosophy': '과학적 사고와 불확실성 정량화를 통한 투자'
    }
    
except Exception as e:
    logger.error(f"Error creating Feynman portfolio: {str(e)}")
    return {}
```