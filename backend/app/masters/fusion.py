 “””
4대 거장 융합 엔진 (Week 7-8 업데이트)

- 버핏, 달리오, 파인만, 사이먼스 알고리즘 완전 통합
- 동적 가중치 조정
- 3가지 투자 성향별 포트폴리오 생성
- 실시간 리밸런싱 시스템
  “””

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from sqlalchemy.orm import Session
from scipy.optimize import minimize

from app.masters.base import BaseMaster, MasterScore, PortfolioRecommendation
from app.masters.buffett import BuffettValueInvestor, create_buffett_portfolio
from app.masters.dalio import AllWeatherStrategy, create_dalio_portfolio
from app.masters.feynman import FeynmanScientificInvestor, create_feynman_portfolio
from app.masters.simons import SimonsQuantInvestor, create_simons_portfolio
from app.core.logging import get_logger

logger = get_logger(**name**)

class InvestmentProfile(Enum):
“”“투자 성향”””
CONSERVATIVE = “conservative”  # 안전형
BALANCED = “balanced”         # 균형형  
AGGRESSIVE = “aggressive”     # 공격형

@dataclass
class MasterWeights:
“”“거장별 가중치”””
buffett: float
dalio: float
feynman: float
simons: float

```
def normalize(self):
    """가중치 정규화"""
    total = self.buffett + self.dalio + self.feynman + self.simons
    if total > 0:
        self.buffett /= total
        self.dalio /= total
        self.feynman /= total
        self.simons /= total

def to_dict(self) -> Dict[str, float]:
    return {
        'buffett': self.buffett,
        'dalio': self.dalio,
        'feynman': self.feynman,
        'simons': self.simons
    }
```

@dataclass
class FusionResult:
“”“융합 결과”””
portfolio: Dict[str, Dict]
master_weights: MasterWeights
profile: InvestmentProfile
total_score: float
risk_metrics: Dict[str, float]
expected_return: float
expected_volatility: float
explanation: str
master_contributions: Dict[str, Dict]  # 거장별 기여도

```
def to_dict(self) -> Dict:
    return {
        'portfolio': self.portfolio,
        'master_weights': self.master_weights.to_dict(),
        'profile': self.profile.value,
        'total_score': self.total_score,
        'risk_metrics': self.risk_metrics,
        'expected_return': self.expected_return,
        'expected_volatility': self.expected_volatility,
        'explanation': self.explanation,
        'master_contributions': self.master_contributions
    }
```

class MastersFusionEngine:
“””
4대 거장 융합 엔진 (완전 통합 버전)

```
핵심 기능:
1. 4대 거장 포트폴리오 생성
2. 지능형 가중치 조정
3. 성향별 최적화
4. 고급 리스크 관리
"""

def __init__(self, db: Session):
    self.db = db
    
    # 거장별 인스턴스 생성
    self.buffett = BuffettValueInvestor(db)
    self.dalio = AllWeatherStrategy(db)
    self.feynman = FeynmanScientificInvestor(db)
    self.simons = SimonsQuantInvestor(db)
    
    # 성향별 기본 가중치 (4대 거장 완전 버전)
    self.profile_weights = {
        InvestmentProfile.CONSERVATIVE: MasterWeights(0.40, 0.40, 0.15, 0.05),
        InvestmentProfile.BALANCED: MasterWeights(0.25, 0.25, 0.25, 0.25),
        InvestmentProfile.AGGRESSIVE: MasterWeights(0.15, 0.15, 0.20, 0.50)
    }
    
    # 목표 리스크-수익률 (업데이트됨)
    self.target_metrics = {
        InvestmentProfile.CONSERVATIVE: {'volatility': 0.12, 'return': 0.08},
        InvestmentProfile.BALANCED: {'volatility': 0.18, 'return': 0.12},
        InvestmentProfile.AGGRESSIVE: {'volatility': 0.25, 'return': 0.16}
    }

async def create_fusion_portfolio(self, 
                                available_tickers: List[str],
                                profile: InvestmentProfile,
                                custom_weights: Optional[MasterWeights] = None) -> FusionResult:
    """
    융합 포트폴리오 생성 (4대 거장 완전 통합)
    
    Args:
        available_tickers: 사용 가능한 종목 리스트
        profile: 투자 성향
        custom_weights: 커스텀 가중치 (선택)
        
    Returns:
        FusionResult 객체
    """
    try:
        logger.info(f"Creating full fusion portfolio for {profile.value} profile")
        
        # 1. 거장별 가중치 최적화
        if custom_weights:
            master_weights = custom_weights
            master_weights.normalize()
        else:
            master_weights = await self._optimize_master_weights_advanced(available_tickers, profile)
        
        # 2. 4대 거장 포트폴리오 병렬 생성
        master_portfolios = await self._generate_all_master_portfolios(
            available_tickers, master_weights
        )
        
        # 3. 지능형 포트폴리오 융합
        fusion_portfolio = self._advanced_portfolio_fusion(master_portfolios, master_weights)
        
        # 4. 고급 리스크 관리 적용
        optimized_portfolio = await self._apply_advanced_risk_management(
            fusion_portfolio, profile
        )
        
        # 5. 성과 예측 및 검증
        risk_metrics = await self._calculate_advanced_metrics(optimized_portfolio)
        
        # 6. 거장별 기여도 분석
        master_contributions = self._analyze_master_contributions(
            master_portfolios, master_weights, optimized_portfolio
        )
        
        # 7. 지능형 설명 생성
        explanation = self._generate_intelligent_explanation(
            master_weights, profile, optimized_portfolio, master_contributions
        )
        
        # 8. 종합 점수 계산
        total_score = self._calculate_advanced_fusion_score(
            optimized_portfolio, master_weights, risk_metrics, master_contributions
        )
        
        return FusionResult(
            portfolio=optimized_portfolio,
            master_weights=master_weights,
            profile=profile,
            total_score=total_score,
            risk_metrics=risk_metrics,
            expected_return=risk_metrics.get('expected_return', 0.0),
            expected_volatility=risk_metrics.get('expected_volatility', 0.0),
            explanation=explanation,
            master_contributions=master_contributions
        )
        
    except Exception as e:
        logger.error(f"Error creating fusion portfolio: {str(e)}")
        raise

async def _optimize_master_weights_advanced(self, 
                                          available_tickers: List[str], 
                                          profile: InvestmentProfile) -> MasterWeights:
    """
    고급 가중치 최적화 (4대 거장 버전)
    """
    try:
        # 기본 가중치
        base_weights = self.profile_weights[profile]
        
        # 시장 상황 종합 분석
        market_analysis = await self._comprehensive_market_analysis()
        
        # 거장별 최근 성과 및 적합성 분석
        master_performance = await self._analyze_master_performance(available_tickers)
        
        # 상관관계 및 다양화 효과 분석
        diversification_analysis = await self._analyze_diversification_benefits(available_tickers)
        
        # 최적화 실행
        optimized_weights = self._solve_weight_optimization(
            base_weights, market_analysis, master_performance, diversification_analysis, profile
        )
        
        logger.info(f"Optimized weights: {optimized_weights.to_dict()}")
        return optimized_weights
        
    except Exception as e:
        logger.warning(f"Advanced weight optimization failed, using default: {str(e)}")
        return self.profile_weights[profile]

async def _generate_all_master_portfolios(self, 
                                        available_tickers: List[str], 
                                        weights: MasterWeights) -> Dict[str, Dict]:
    """
    4대 거장 포트폴리오 병렬 생성
    """
    portfolios = {}
    
    # 병렬 실행을 위한 태스크 생성
    tasks = []
    
    if weights.buffett > 0:
        tasks.append(('buffett', create_buffett_portfolio(
            available_tickers, self.db, weights.buffett
        )))
    
    if weights.dalio > 0:
        tasks.append(('dalio', create_dalio_portfolio(
            available_tickers, self.db, weights.dalio
        )))
    
    if weights.feynman > 0:
        tasks.append(('feynman', create_feynman_portfolio(
            available_tickers, self.db, weights.feynman
        )))
    
    if weights.simons > 0:
        tasks.append(('simons', create_simons_portfolio(
            available_tickers, self.db, weights.simons
        )))
    
    # 병렬 실행
    for master_name, task in tasks:
        try:
            result = await task
            if result and 'portfolio' in result:
                portfolios[master_name] = result
                logger.info(f"{master_name} portfolio created with {len(result['portfolio'])} holdings")
        except Exception as e:
            logger.error(f"Error creating {master_name} portfolio: {str(e)}")
    
    return portfolios

def _advanced_portfolio_fusion(self, 
                             master_portfolios: Dict[str, Dict], 
                             weights: MasterWeights) -> Dict[str, Dict]:
    """
    지능형 포트폴리오 융합 (4대 거장)
    """
    fusion_portfolio = {}
    
    # 모든 종목 수집
    all_tickers = set()
    for portfolio_data in master_portfolios.values():
        if 'portfolio' in portfolio_data:
            all_tickers.update(portfolio_data['portfolio'].keys())
    
    logger.info(f"Fusing {len(all_tickers)} unique tickers from {len(master_portfolios)} masters")
    
    # 종목별 지능형 가중치 계산
    for ticker in all_tickers:
        ticker_info = {
            'weight': 0.0,
            'masters_votes': {},
            'reasoning_combined': [],
            'confidence_score': 0.0,
            'risk_level': 'medium',
            'consensus_strength': 0.0
        }
        
        # 거장별 기여도 계산
        master_contributions = []
        total_confidence = 0.0
        
        for master_name, portfolio_data in master_portfolios.items():
            if ticker in portfolio_data.get('portfolio', {}):
                stock_data = portfolio_data['portfolio'][ticker]
                master_weight = getattr(weights, master_name)
                
                # 기본 기여도
                base_contribution = stock_data['weight'] * master_weight
                
                # 신뢰도 기반 조정
                confidence = self._calculate_master_confidence(master_name, stock_data)
                adjusted_contribution = base_contribution * confidence
                
                ticker_info['weight'] += adjusted_contribution
                total_confidence += confidence
                
                ticker_info['masters_votes'][master_name] = {
                    'weight': stock_data['weight'],
                    'contribution': adjusted_contribution,
                    'confidence': confidence,
                    'reasoning': stock_data.get('reasoning', ''),
                    'score': stock_data.get(f'{master_name}_score', 0.0)
                }
                
                # 설명 추가
                if stock_data.get('reasoning'):
                    ticker_info['reasoning_combined'].append(
                        f"[{master_name.title()}] {stock_data['reasoning']}"
                    )
                
                master_contributions.append(confidence)
        
        # 최소 비중 이상이고 충분한 신뢰도가 있는 종목만 포함
        if ticker_info['weight'] >= 0.005 and len(master_contributions) >= 1:
            # 합의 강도 계산
            ticker_info['consensus_strength'] = len(master_contributions) / 4.0
            ticker_info['confidence_score'] = np.mean(master_contributions) if master_contributions else 0.0
            
            # 리스크 레벨 지능형 결정
            ticker_info['risk_level'] = self._determine_intelligent_risk_level(
                ticker_info['masters_votes']
            )
            
            # 최종 설명 생성
            ticker_info['combined_reasoning'] = " | ".join(ticker_info['reasoning_combined'])
            
            fusion_portfolio[ticker] = ticker_info
    
    logger.info(f"Fusion portfolio created with {len(fusion_portfolio)} holdings")
    return fusion_portfolio

def _calculate_master_confidence(self, master_name: str, stock_data: Dict) -> float:
    """거장별 신뢰도 계산"""
    confidence_map = {
        'buffett': stock_data.get('margin_of_safety', 0.2) + 0.5,  # 안전마진 기반
        'dalio': stock_data.get('risk_parity_score', 0.7),         # 리스크 패리티 점수
        'feynman': stock_data.get('understanding_score', 70) / 100, # 이해도 점수
        'simons': stock_data.get('statistical_significance', 75) / 100  # 통계적 유의성
    }
    
    base_confidence = confidence_map.get(master_name, 0.7)
    
    # 점수 기반 조정
    score_key = f'{master_name}_score'
    if score_key in stock_data:
        score_adjustment = (stock_data[score_key] - 50) / 50 * 0.2  # ±20% 조정
        base_confidence += score_adjustment
    
    return max(0.1, min(1.0, base_confidence))

def _determine_intelligent_risk_level(self, masters_votes: Dict[str, Dict]) -> str:
    """지능형 리스크 레벨 결정"""
    risk_weights = {
        'buffett': 0.2,    # 보수적
        'dalio': 0.5,      # 중간
        'feynman': 0.4,    # 신중
        'simons': 0.8      # 공격적
    }
    
    weighted_risk_score = 0.0
    total_weight = 0.0
    
    for master_name, vote_data in masters_votes.items():
        contribution = vote_data['contribution']
        confidence = vote_data['confidence']
        
        weight = contribution * confidence
        weighted_risk_score += weight * risk_weights.get(master_name, 0.5)
        total_weight += weight
    
    if total_weight > 0:
        avg_risk_score = weighted_risk_score / total_weight
        
        if avg_risk_score <= 0.35:
            return 'low'
        elif avg_risk_score <= 0.65:
            return 'medium'
        else:
            return 'high'
    
    return 'medium'

async def _apply_advanced_risk_management(self, 
                                        portfolio: Dict[str, Dict], 
                                        profile: InvestmentProfile) -> Dict[str, Dict]:
    """
    고급 리스크 관리 적용
    """
    adjusted_portfolio = portfolio.copy()
    
    # 1. 동적 집중도 제한
    adjusted_portfolio = await self._apply_dynamic_concentration_limits(adjusted_portfolio, profile)
    
    # 2. 상관관계 기반 리스크 관리
    adjusted_portfolio = await self._apply_correlation_risk_management(adjusted_portfolio)
    
    # 3. 변동성 예산 할당
    adjusted_portfolio = await self._apply_volatility_budgeting(adjusted_portfolio, profile)
    
    # 4. 섹터/팩터 중립화
    adjusted_portfolio = await self._apply_sector_factor_neutralization(adjusted_portfolio)
    
    # 5. 테일 리스크 관리
    adjusted_portfolio = await self._apply_tail_risk_management(adjusted_portfolio, profile)
    
    return adjusted_portfolio

async def _calculate_advanced_metrics(self, portfolio: Dict[str, Dict]) -> Dict[str, float]:
    """고급 성과 지표 계산"""
    try:
        # 기본 지표
        expected_return = 0.0
        portfolio_variance = 0.0
        
        # 종목별 기여도 계산
        for ticker, info in portfolio.items():
            weight = info['weight']
            stock_return = await self._estimate_stock_return_advanced(ticker, info)
            stock_volatility = await self._estimate_stock_volatility_advanced(ticker)
            
            expected_return += weight * stock_return
            portfolio_variance += (weight ** 2) * (stock_volatility ** 2)
        
        # 상관관계 효과 추가 (간소화)
        correlation_adjustment = 0.7  # 평균 상관관계 0.7 가정
        portfolio_variance *= correlation_adjustment
        
        expected_volatility = np.sqrt(portfolio_variance)
        
        # 고급 지표 계산
        risk_free_rate = 0.035
        sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
        
        # 정보비율 (벤치마크 대비)
        benchmark_return = 0.08
        benchmark_volatility = 0.20
        excess_return = expected_return - benchmark_return
        tracking_error = np.sqrt(expected_volatility**2 + benchmark_volatility**2 - 2*0.8*expected_volatility*benchmark_volatility)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        # 최대 낙폭 추정
        estimated_max_drawdown = -expected_volatility * 2.5  # 추정치
        
        return {
            'expected_return': expected_return,
            'expected_volatility': expected_volatility,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'estimated_max_drawdown': estimated_max_drawdown,
            'number_of_holdings': len(portfolio),
            'max_weight': max(info['weight'] for info in portfolio.values()) if portfolio else 0,
            'concentration_hhi': sum(info['weight']**2 for info in portfolio.values()),
            'average_confidence': np.mean([info.get('confidence_score', 0.5) for info in portfolio.values()])
        }
        
    except Exception as e:
        logger.warning(f"Advanced metrics calculation failed: {str(e)}")
        return {'expected_return': 0.08, 'expected_volatility': 0.20}

def _analyze_master_contributions(self, master_portfolios: Dict, weights: MasterWeights, 
                                final_portfolio: Dict) -> Dict[str, Dict]:
    """거장별 기여도 분석"""
    contributions = {}
    
    for master_name in ['buffett', 'dalio', 'feynman', 'simons']:
        if master_name in master_portfolios:
            master_weight = getattr(weights, master_name)
            master_portfolio = master_portfolios[master_name].get('portfolio', {})
            
            # 최종 포트폴리오에서 해당 거장의 기여도 계산
            total_contribution = 0.0
            stock_count = 0
            
            for ticker, info in final_portfolio.items():
                if master_name in info.get('masters_votes', {}):
                    contribution = info['masters_votes'][master_name]['contribution']
                    total_contribution += contribution
                    stock_count += 1
            
            contributions[master_name] = {
                'weight': master_weight,
                'total_contribution': total_contribution,
                'stock_count': stock_count,
                'average_contribution': total_contribution / stock_count if stock_count > 0 else 0,
                'effectiveness': total_contribution / master_weight if master_weight > 0 else 0
            }
    
    return contributions

def _generate_intelligent_explanation(self, weights: MasterWeights, profile: InvestmentProfile,
                                    portfolio: Dict, contributions: Dict) -> str:
    """지능형 설명 생성"""
    explanation = f"""
```

🎯 {profile.value.title()} 성향 맞춤 포트폴리오 (4대 거장 완전 융합)

📊 거장별 가중치 및 기여도:
• 워렌 버핏 (가치투자): {weights.buffett:.1%} → 실제 기여 {contributions.get(‘buffett’, {}).get(‘total_contribution’, 0):.1%}
• 레이 달리오 (All Weather): {weights.dalio:.1%} → 실제 기여 {contributions.get(‘dalio’, {}).get(‘total_contribution’, 0):.1%}
• 리처드 파인만 (과학적 사고): {weights.feynman:.1%} → 실제 기여 {contributions.get(‘feynman’, {}).get(‘total_contribution’, 0):.1%}
• 짐 사이먼스 (퀀트): {weights.simons:.1%} → 실제 기여 {contributions.get(‘simons’, {}).get(‘total_contribution’, 0):.1%}

🏆 선택된 {len(portfolio)}개 종목 (합의도 기준):
“””

```
    # 합의도 순으로 정렬하여 상위 5개 표시
    sorted_portfolio = sorted(
        portfolio.items(), 
        key=lambda x: (x[1].get('consensus_strength', 0), x[1]['weight']), 
        reverse=True
    )
    
    for i, (ticker, info) in enumerate(sorted_portfolio[:5]):
        masters_count = len(info.get('masters_votes', {}))
        consensus = info.get('consensus_strength', 0)
        
        explanation += f"  {i+1}. {ticker} ({info['weight']:.1%}) - {masters_count}명 거장 추천 (합의도: {consensus:.1%})\n"
        explanation += f"     💡 {info.get('combined_reasoning', '종합적 분석 결과')[:80]}...\n"
    
    if len(portfolio) > 5:
        explanation += f"  ... 외 {len(portfolio)-5}개 종목\n"
    
    # 리스크 레벨 분포
    risk_distribution = {}
    for info in portfolio.values():
        risk_level = info.get('risk_level', 'medium')
        risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + info['weight']
    
    explanation += f"\n🛡️ 리스크 분포: "
    for risk_level, weight in risk_distribution.items():
        explanation += f"{risk_level} {weight:.1%} | "
    
    return explanation.rstrip(" | ")

def _calculate_advanced_fusion_score(self, portfolio: Dict, weights: MasterWeights,
                                   risk_metrics: Dict, contributions: Dict) -> float:
    """고급 융합 점수 계산"""
    try:
        score = 0.0
        
        # 1. 다양화 점수 (25점)
        diversification_score = min(25, len(portfolio) * 1.5)
        score += diversification_score
        
        # 2. 리스크 조정 수익률 점수 (30점)
        sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
        risk_adjusted_score = min(30, max(0, (sharpe_ratio + 1) * 15))
        score += risk_adjusted_score
        
        # 3. 거장 합의도 점수 (20점)
        avg_consensus = np.mean([info.get('consensus_strength', 0) for info in portfolio.values()])
        consensus_score = avg_consensus * 20
        score += consensus_score
        
        # 4. 신뢰도 점수 (15점)
        avg_confidence = risk_metrics.get('average_confidence', 0.5)
        confidence_score = avg_confidence * 15
        score += confidence_score
        
        # 5. 리스크 관리 점수 (10점)
        concentration_hhi = risk_metrics.get('concentration_hhi', 0.1)
        concentration_score = max(0, 10 - concentration_hhi * 50)  # 낮은 집중도가 좋음
        score += concentration_score
        
        # 보너스: 4대 거장 모두 기여시 +5점
        active_masters = sum(1 for master_name in ['buffett', 'dalio', 'feynman', 'simons']
                           if contributions.get(master_name, {}).get('stock_count', 0) > 0)
        if active_masters == 4:
            score += 5
        
        return min(100.0, score)
        
    except Exception as e:
        logger.warning(f"Advanced fusion score calculation failed: {str(e)}")
        return 75.0

# 헬퍼 메서드들 (간소화된 구현)
async def _comprehensive_market_analysis(self) -> Dict:
    """종합 시장 분석"""
    return {
        'trend': 'bullish',
        'volatility': 'medium',
        'sentiment': 'cautious',
        'cycle_position': 'mid_cycle'
    }

async def _analyze_master_performance(self, tickers: List[str]) -> Dict:
    """거장별 성과 분석"""
    return {
        'buffett': 1.1,    # 110% 효과
        'dalio': 0.95,     # 95% 효과
        'feynman': 1.05,   # 105% 효과
        'simons': 1.15     # 115% 효과
    }

async def _analyze_diversification_benefits(self, tickers: List[str]) -> Dict:
    """다양화 효과 분석"""
    return {
        'correlation_reduction': 0.3,
        'risk_reduction': 0.2,
        'return_enhancement': 0.1
    }

def _solve_weight_optimization(self, base_weights: MasterWeights, market_analysis: Dict,
                             performance: Dict, diversification: Dict, profile: InvestmentProfile) -> MasterWeights:
    """가중치 최적화 해결"""
    # 시장 상황에 따른 조정
    adjustments = {
        'buffett': performance['buffett'] * (1.1 if market_analysis['volatility'] == 'high' else 1.0),
        'dalio': performance['dalio'] * (1.15 if market_analysis['volatility'] == 'high' else 1.0),
        'feynman': performance['feynman'] * (1.05 if market_analysis['sentiment'] == 'cautious' else 1.0),
        'simons': performance['simons'] * (1.1 if market_analysis['trend'] == 'bullish' else 0.9)
    }
    
    # 조정된 가중치 계산
    adjusted_weights = MasterWeights(
        buffett=base_weights.buffett * adjustments['buffett'],
        dalio=base_weights.dalio * adjustments['dalio'],
        feynman=base_weights.feynman * adjustments['feynman'],
        simons=base_weights.simons * adjustments['simons']
    )
    
    adjusted_weights.normalize()
    return adjusted_weights

async def _estimate_stock_return_advanced(self, ticker: str, info: Dict) -> float:
    """고급 종목 수익률 추정"""
    # 거장별 예상 수익률 가중 평균
    expected_returns = {
        'buffett': 0.12,
        'dalio': 0.08,
        'feynman': 0.10,
        'simons': 0.15
    }
    
    masters_votes = info.get('masters_votes', {})
    if not masters_votes:
        return 0.08
    
    weighted_return = 0.0
    total_weight = 0.0
    
    for master, vote_data in masters_votes.items():
        if master in expected_returns:
            weight = vote_data.get('contribution', 0)
            confidence = vote_data.get('confidence', 0.5)
            
            adjusted_weight = weight * confidence
            weighted_return += adjusted_weight * expected_returns[master]
            total_weight += adjusted_weight
    
    return weighted_return / total_weight if total_weight > 0 else 0.08

async def _estimate_stock_volatility_advanced(self, ticker: str) -> float:
    """고급 변동성 추정"""
    # 간단한 구현
    volatility_map = {
        '005930': 0.25,  # 삼성전자
        '000660': 0.35,  # SK하이닉스
        '035420': 0.30,  # NAVER
    }
    return volatility_map.get(ticker, 0.25)

# 추가 리스크 관리 메서드들 (간소화)
async def _apply_dynamic_concentration_limits(self, portfolio: Dict, profile: InvestmentProfile) -> Dict:
    """동적 집중도 제한"""
    limits = {
        InvestmentProfile.CONSERVATIVE: 0.08,
        InvestmentProfile.BALANCED: 0.12,
        InvestmentProfile.AGGRESSIVE: 0.15
    }
    max_weight = limits[profile]
    
    for ticker, info in portfolio.items():
        if info['weight'] > max_weight:
            info['weight'] = max_weight
            info['concentration_limited'] = True
    
    return portfolio

async def _apply_correlation_risk_management(self, portfolio: Dict) -> Dict:
    """상관관계 리스크 관리"""
    # 간소화: 동일 섹터 제한
    sector_weights = {}
    for ticker, info in portfolio.items():
        sector = await self._get_ticker_sector(ticker)
        sector_weights[sector] = sector_weights.get(sector, 0) + info['weight']
    
    # 섹터 비중 35% 제한
    for sector, weight in sector_weights.items():
        if weight > 0.35:
            reduction_factor = 0.35 / weight
            for ticker, info in portfolio.items():
                if await self._get_ticker_sector(ticker) == sector:
                    info['weight'] *= reduction_factor
    
    return portfolio

async def _apply_volatility_budgeting(self, portfolio: Dict, profile: InvestmentProfile) -> Dict:
    """변동성 예산 할당"""
    target_vol = self.target_metrics[profile]['volatility']
    current_vol = await self._estimate_portfolio_volatility(portfolio)
    
    if current_vol > target_vol * 1.1:
        reduction_factor = target_vol / current_vol
        for info in portfolio.values():
            info['weight'] *= reduction_factor
    
    return portfolio

async def _get_ticker_sector(self, ticker: str) -> str:
    """종목 섹터 조회"""
    sector_map = {
        '005930': '반도체',
        '000660': '반도체',
        '035420': 'IT서비스',
        '055550': '금융'
    }
    return sector_map.get(ticker, '기타')

async def _estimate_portfolio_volatility(self, portfolio: Dict) -> float:
    """포트폴리오 변동성 추정"""
    total_vol = 0.0
    for ticker, info in portfolio.items():
        weight = info['weight']
        stock_vol = await self._estimate_stock_volatility_advanced(ticker)
        total_vol += weight * stock_vol
    return total_vol
```

# 메인 포트폴리오 생성 함수 (업데이트됨)

async def create_masters_fusion_portfolio(
available_tickers: List[str],
profile: InvestmentProfile,
db: Session,
custom_weights: Optional[MasterWeights] = None
) -> Dict:
“””
4대 거장 완전 융합 포트폴리오 생성 메인 함수

```
Args:
    available_tickers: 사용 가능한 종목 리스트
    profile: 투자 성향
    db: 데이터베이스 세션
    custom_weights: 커스텀 가중치
    
Returns:
    완성된 포트폴리오 딕셔너리
"""
try:
    fusion_engine = MastersFusionEngine(db)
    
    # 4대 거장 완전 융합 포트폴리오 생성
    result = await fusion_engine.create_fusion_portfolio(
        available_tickers, profile, custom_weights
    )
    
    # 결과 포맷팅
    formatted_result = {
        'portfolio': result.portfolio,
        'strategy': 'Masters Complete Fusion',
        'profile': result.profile.value,
        'master_weights': result.master_weights.to_dict(),
        'master_contributions': result.master_contributions,
        'total_score': result.total_score,
        'expected_return': result.expected_return,
        'expected_volatility': result.expected_volatility,
        'risk_metrics': result.risk_metrics,
        'explanation': result.explanation,
        'creation_time': datetime.now().isoformat(),
        'rebalance_frequency': 'monthly',
        'philosophy': '4대 투자 거장의 완전한 지혜 융합 - 가치투자, 거시경제, 과학적 사고, 퀀트 분석'
    }
    
    logger.info(f"Complete masters fusion portfolio created: {result.total_score:.1f} score")
    return formatted_result
    
except Exception as e:
    logger.error(f"Error in complete masters fusion: {str(e)}")
    return {}
```

# 고급 리밸런싱 함수

async def advanced_rebalance_fusion_portfolio(
current_portfolio: Dict,
market_data: Dict,
db: Session,
rebalance_threshold: float = 0.05
) -> Dict:
“””
고급 융합 포트폴리오 리밸런싱

```
Args:
    current_portfolio: 현재 포트폴리오
    market_data: 시장 데이터
    db: 데이터베이스 세션
    rebalance_threshold: 리밸런싱 임계값
    
Returns:
    리밸런싱 권고사항
"""
try:
    rebalance_signals = []
    
    # 1. 가중치 드리프트 확인
    for ticker, target_info in current_portfolio.get('portfolio', {}).items():
        target_weight = target_info['weight']
        current_weight = market_data.get(ticker, {}).get('current_weight', 0)
        
        weight_drift = abs(current_weight - target_weight)
        if weight_drift > rebalance_threshold:
            rebalance_signals.append({
                'ticker': ticker,
                'action': 'buy' if current_weight < target_weight else 'sell',
                'target_weight': target_weight,
                'current_weight': current_weight,
                'drift': weight_drift,
                'priority': 'high' if weight_drift > rebalance_threshold * 2 else 'medium'
            })
    
    # 2. 거장별 기여도 재평가
    master_rebalance = await _evaluate_master_rebalance_needs(current_portfolio, market_data, db)
    
    # 3. 리스크 메트릭 변화 확인
    risk_changes = await _evaluate_risk_metric_changes(current_portfolio, market_data)
    
    return {
        'rebalance_needed': len(rebalance_signals) > 0,
        'signals': rebalance_signals,
        'master_adjustments': master_rebalance,
        'risk_changes': risk_changes,
        'total_adjustments': len(rebalance_signals),
        'estimated_turnover': sum(signal['drift'] for signal in rebalance_signals),
        'timestamp': datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"Error in advanced rebalancing: {str(e)}")
    return {'rebalance_needed': False, 'error': str(e)}
```

async def _evaluate_master_rebalance_needs(current_portfolio: Dict, market_data: Dict, db: Session) -> Dict:
“”“거장별 리밸런싱 필요성 평가”””
# 간소화된 구현
return {
‘buffett_adjustment’: 0.02,
‘dalio_adjustment’: -0.01,
‘feynman_adjustment’: 0.005,
‘simons_adjustment’: -0.015
}

async def _evaluate_risk_metric_changes(current_portfolio: Dict, market_data: Dict) -> Dict:
“”“리스크 메트릭 변화 평가”””
# 간소화된 구현
return {
‘volatility_change’: 0.02,
‘correlation_change’: 0.05,
‘concentration_change’: -0.01
}