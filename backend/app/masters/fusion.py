“””
4대 거장 융합 엔진

- 버핏, 달리오, 파인만, 사이먼스 알고리즘 통합
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

# from app.masters.feynman import FeynmanScientificInvestor, create_feynman_portfolio

# from app.masters.simons import SimonsQuantInvestor, create_simons_portfolio

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
        'explanation': self.explanation
    }
```

class MastersFusionEngine:
“””
4대 거장 융합 엔진

```
핵심 기능:
1. 거장별 포트폴리오 생성
2. 동적 가중치 조정
3. 성향별 최적화
4. 리스크 관리
"""

def __init__(self, db: Session):
    self.db = db
    
    # 거장별 인스턴스 생성
    self.buffett = BuffettValueInvestor(db)
    self.dalio = AllWeatherStrategy(db)
    # self.feynman = FeynmanScientificInvestor(db)  # Week 7-8에서 구현
    # self.simons = SimonsQuantInvestor(db)         # Week 7-8에서 구현
    
    # 성향별 기본 가중치
    self.profile_weights = {
        InvestmentProfile.CONSERVATIVE: MasterWeights(0.40, 0.40, 0.15, 0.05),
        InvestmentProfile.BALANCED: MasterWeights(0.30, 0.30, 0.20, 0.20),
        InvestmentProfile.AGGRESSIVE: MasterWeights(0.20, 0.20, 0.20, 0.40)
    }
    
    # 목표 리스크-수익률
    self.target_metrics = {
        InvestmentProfile.CONSERVATIVE: {'volatility': 0.15, 'return': 0.08},
        InvestmentProfile.BALANCED: {'volatility': 0.20, 'return': 0.10},
        InvestmentProfile.AGGRESSIVE: {'volatility': 0.25, 'return': 0.12}
    }

async def create_fusion_portfolio(self, 
                                available_tickers: List[str],
                                profile: InvestmentProfile,
                                custom_weights: Optional[MasterWeights] = None) -> FusionResult:
    """
    융합 포트폴리오 생성
    
    Args:
        available_tickers: 사용 가능한 종목 리스트
        profile: 투자 성향
        custom_weights: 커스텀 가중치 (선택)
        
    Returns:
        FusionResult 객체
    """
    try:
        logger.info(f"Creating fusion portfolio for {profile.value} profile")
        
        # 1. 거장별 가중치 결정
        if custom_weights:
            master_weights = custom_weights
            master_weights.normalize()
        else:
            master_weights = await self._optimize_master_weights(available_tickers, profile)
        
        # 2. 거장별 포트폴리오 생성
        master_portfolios = await self._generate_master_portfolios(
            available_tickers, master_weights
        )
        
        # 3. 포트폴리오 융합
        fusion_portfolio = self._fuse_portfolios(master_portfolios, master_weights)
        
        # 4. 리스크 조정
        adjusted_portfolio = await self._apply_risk_management(
            fusion_portfolio, profile
        )
        
        # 5. 성과 예측
        risk_metrics = self._calculate_expected_metrics(adjusted_portfolio)
        
        # 6. 설명 생성
        explanation = self._generate_fusion_explanation(
            master_weights, profile, adjusted_portfolio
        )
        
        # 7. 종합 점수 계산
        total_score = self._calculate_fusion_score(
            adjusted_portfolio, master_weights, risk_metrics
        )
        
        return FusionResult(
            portfolio=adjusted_portfolio,
            master_weights=master_weights,
            profile=profile,
            total_score=total_score,
            risk_metrics=risk_metrics,
            expected_return=risk_metrics.get('expected_return', 0.0),
            expected_volatility=risk_metrics.get('expected_volatility', 0.0),
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Error creating fusion portfolio: {str(e)}")
        raise

async def _optimize_master_weights(self, 
                                 available_tickers: List[str], 
                                 profile: InvestmentProfile) -> MasterWeights:
    """
    시장 상황에 따른 거장별 가중치 최적화
    """
    try:
        # 기본 가중치
        base_weights = self.profile_weights[profile]
        
        # 시장 상황 분석
        market_condition = await self._analyze_market_condition()
        
        # 조건별 가중치 조정
        adjusted_weights = self._adjust_weights_by_market(base_weights, market_condition)
        
        # 거장별 최근 성과 고려
        performance_adjustment = await self._calculate_performance_adjustment(available_tickers)
        
        # 최종 가중치 계산
        final_weights = MasterWeights(
            buffett=adjusted_weights.buffett * performance_adjustment.get('buffett', 1.0),
            dalio=adjusted_weights.dalio * performance_adjustment.get('dalio', 1.0),
            feynman=adjusted_weights.feynman * performance_adjustment.get('feynman', 1.0),
            simons=adjusted_weights.simons * performance_adjustment.get('simons', 1.0)
        )
        
        final_weights.normalize()
        
        logger.info(f"Optimized weights: {final_weights.to_dict()}")
        return final_weights
        
    except Exception as e:
        logger.warning(f"Weight optimization failed, using default: {str(e)}")
        return self.profile_weights[profile]

async def _generate_master_portfolios(self, 
                                    available_tickers: List[str], 
                                    weights: MasterWeights) -> Dict[str, Dict]:
    """
    거장별 포트폴리오 생성
    """
    portfolios = {}
    
    # 병렬로 포트폴리오 생성
    tasks = []
    
    if weights.buffett > 0:
        tasks.append(('buffett', create_buffett_portfolio(
            available_tickers, self.db, weights.buffett
        )))
    
    if weights.dalio > 0:
        tasks.append(('dalio', create_dalio_portfolio(
            available_tickers, self.db, weights.dalio
        )))
    
    # Week 7-8에서 추가
    # if weights.feynman > 0:
    #     tasks.append(('feynman', create_feynman_portfolio(...)))
    
    # if weights.simons > 0:
    #     tasks.append(('simons', create_simons_portfolio(...)))
    
    # 비동기 실행
    for master_name, task in tasks:
        try:
            result = await task
            if result and 'portfolio' in result:
                portfolios[master_name] = result
                logger.info(f"{master_name} portfolio created with {len(result['portfolio'])} holdings")
        except Exception as e:
            logger.error(f"Error creating {master_name} portfolio: {str(e)}")
    
    return portfolios

def _fuse_portfolios(self, 
                    master_portfolios: Dict[str, Dict], 
                    weights: MasterWeights) -> Dict[str, Dict]:
    """
    거장별 포트폴리오 융합
    """
    fusion_portfolio = {}
    
    # 모든 종목 수집
    all_tickers = set()
    for portfolio_data in master_portfolios.values():
        if 'portfolio' in portfolio_data:
            all_tickers.update(portfolio_data['portfolio'].keys())
    
    # 종목별 가중치 합산
    for ticker in all_tickers:
        ticker_weight = 0.0
        ticker_info = {
            'weight': 0.0,
            'masters_votes': {},
            'reasoning_combined': [],
            'risk_level': 'medium',
            'expected_return': 0.0,
            'volatility': 0.0
        }
        
        # 거장별 비중 합산
        for master_name, portfolio_data in master_portfolios.items():
            if ticker in portfolio_data.get('portfolio', {}):
                stock_data = portfolio_data['portfolio'][ticker]
                master_weight = getattr(weights, master_name)
                
                contribution = stock_data['weight'] * master_weight
                ticker_weight += contribution
                
                ticker_info['masters_votes'][master_name] = {
                    'weight': stock_data['weight'],
                    'contribution': contribution,
                    'reasoning': stock_data.get('reasoning', ''),
                    'score': stock_data.get('score', 0.0)
                }
                
                # 설명 추가
                if stock_data.get('reasoning'):
                    ticker_info['reasoning_combined'].append(
                        f"[{master_name.title()}] {stock_data['reasoning']}"
                    )
        
        # 최소 비중 이상인 종목만 포함
        if ticker_weight >= 0.005:  # 0.5% 이상
            ticker_info['weight'] = ticker_weight
            ticker_info['combined_reasoning'] = " | ".join(ticker_info['reasoning_combined'])
            
            # 리스크 레벨 결정 (거장들의 합의)
            ticker_info['risk_level'] = self._determine_consensus_risk_level(
                ticker_info['masters_votes']
            )
            
            fusion_portfolio[ticker] = ticker_info
    
    logger.info(f"Fusion portfolio created with {len(fusion_portfolio)} holdings")
    return fusion_portfolio

def _determine_consensus_risk_level(self, masters_votes: Dict[str, Dict]) -> str:
    """
    거장들의 합의를 통한 리스크 레벨 결정
    """
    risk_mapping = {
        'buffett': 'low',      # 버핏은 보수적
        'dalio': 'medium',     # 달리오는 균형
        'feynman': 'medium',   # 파인만은 신중
        'simons': 'high'       # 사이먼스는 공격적
    }
    
    risk_scores = {'low': 1, 'medium': 2, 'high': 3}
    
    total_weight = 0.0
    weighted_risk_score = 0.0
    
    for master_name, vote_data in masters_votes.items():
        weight = vote_data['contribution']
        risk_level = risk_mapping.get(master_name, 'medium')
        
        total_weight += weight
        weighted_risk_score += weight * risk_scores[risk_level]
    
    if total_weight > 0:
        avg_risk_score = weighted_risk_score / total_weight
        
        if avg_risk_score <= 1.5:
            return 'low'
        elif avg_risk_score <= 2.5:
            return 'medium'
        else:
            return 'high'
    
    return 'medium'

async def _apply_risk_management(self, 
                               portfolio: Dict[str, Dict], 
                               profile: InvestmentProfile) -> Dict[str, Dict]:
    """
    리스크 관리 적용
    """
    adjusted_portfolio = portfolio.copy()
    
    # 1. 집중도 제한
    adjusted_portfolio = self._apply_concentration_limits(adjusted_portfolio, profile)
    
    # 2. 섹터 분산 확인
    adjusted_portfolio = await self._ensure_sector_diversification(adjusted_portfolio)
    
    # 3. 변동성 조정
    adjusted_portfolio = self._adjust_for_volatility_target(adjusted_portfolio, profile)
    
    # 4. 유동성 확인
    adjusted_portfolio = await self._apply_liquidity_filters(adjusted_portfolio)
    
    return adjusted_portfolio

def _apply_concentration_limits(self, 
                              portfolio: Dict[str, Dict], 
                              profile: InvestmentProfile) -> Dict[str, Dict]:
    """
    집중도 제한 적용
    """
    # 성향별 최대 개별 종목 비중
    max_position_limits = {
        InvestmentProfile.CONSERVATIVE: 0.10,  # 10%
        InvestmentProfile.BALANCED: 0.15,      # 15%
        InvestmentProfile.AGGRESSIVE: 0.20     # 20%
    }
    
    max_position = max_position_limits[profile]
    adjusted_portfolio = {}
    
    # 비중 조정
    for ticker, info in portfolio.items():
        weight = info['weight']
        
        if weight > max_position:
            # 초과 비중을 다른 종목들에 재분배
            excess_weight = weight - max_position
            info['weight'] = max_position
            info['concentration_adjusted'] = True
            
            # 초과 비중은 나중에 재분배 (간단한 구현)
            logger.info(f"Concentration limit applied to {ticker}: {weight:.2%} -> {max_position:.2%}")
        
        adjusted_portfolio[ticker] = info
    
    return adjusted_portfolio

async def _ensure_sector_diversification(self, portfolio: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    섹터 분산 확보
    """
    # 섹터별 비중 계산
    sector_weights = {}
    
    for ticker, info in portfolio.items():
        sector = await self._get_ticker_sector(ticker)
        
        if sector in sector_weights:
            sector_weights[sector] += info['weight']
        else:
            sector_weights[sector] = info['weight']
    
    # 섹터 집중도 확인 (최대 40%)
    max_sector_weight = 0.40
    
    for sector, weight in sector_weights.items():
        if weight > max_sector_weight:
            logger.warning(f"Sector concentration in {sector}: {weight:.2%}")
            # 실제 구현에서는 섹터 내 종목들의 비중 조정
    
    return portfolio

def _adjust_for_volatility_target(self, 
                                portfolio: Dict[str, Dict], 
                                profile: InvestmentProfile) -> Dict[str, Dict]:
    """
    변동성 목표에 따른 조정
    """
    target_vol = self.target_metrics[profile]['volatility']
    
    # 포트폴리오 예상 변동성 계산 (간단한 구현)
    portfolio_vol = self._estimate_portfolio_volatility(portfolio)
    
    if portfolio_vol > target_vol * 1.1:  # 10% 여유
        # 변동성이 높으면 현금 비중 증가
        cash_allocation = min(0.20, (portfolio_vol - target_vol) * 2)
        
        # 모든 종목 비중을 줄이고 현금 추가
        scaling_factor = 1 - cash_allocation
        
        for ticker, info in portfolio.items():
            info['weight'] *= scaling_factor
        
        # 현금 추가 (실제로는 MMF나 단기채)
        portfolio['069500'] = {  # KODEX 200 (현금 대용)
            'weight': cash_allocation,
            'masters_votes': {'risk_management': {'weight': cash_allocation}},
            'combined_reasoning': f'변동성 조정을 위한 현금 비중 {cash_allocation:.1%}',
            'risk_level': 'low'
        }
        
        logger.info(f"Volatility adjustment: added {cash_allocation:.1%} cash allocation")
    
    return portfolio

async def _apply_liquidity_filters(self, portfolio: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    유동성 필터 적용
    """
    filtered_portfolio = {}
    
    for ticker, info in portfolio.items():
        # 최소 거래대금 확인 (일 1억원)
        avg_volume = await self._get_average_trading_volume(ticker)
        
        if avg_volume >= 1e8:  # 1억원 이상
            filtered_portfolio[ticker] = info
        else:
            logger.warning(f"Liquidity filter: removed {ticker} (volume: {avg_volume:,.0f})")
    
    # 제거된 종목의 비중을 다른 종목에 재분배
    if len(filtered_portfolio) < len(portfolio):
        self._redistribute_weights(filtered_portfolio)
    
    return filtered_portfolio

def _redistribute_weights(self, portfolio: Dict[str, Dict]):
    """
    비중 재분배
    """
    total_weight = sum(info['weight'] for info in portfolio.values())
    
    if total_weight > 0 and total_weight != 1.0:
        scaling_factor = 1.0 / total_weight
        
        for info in portfolio.values():
            info['weight'] *= scaling_factor

def _calculate_expected_metrics(self, portfolio: Dict[str, Dict]) -> Dict[str, float]:
    """
    포트폴리오 예상 성과 지표 계산
    """
    # 간단한 구현 (실제로는 더 정교한 모델 필요)
    expected_return = 0.0
    expected_volatility = 0.0
    
    for ticker, info in portfolio.items():
        weight = info['weight']
        
        # 종목별 예상 수익률 (거장들의 예측 평균)
        stock_expected_return = self._estimate_stock_return(ticker, info)
        expected_return += weight * stock_expected_return
        
        # 종목별 변동성
        stock_volatility = self._estimate_stock_volatility(ticker)
        expected_volatility += (weight ** 2) * (stock_volatility ** 2)  # 단순화된 계산
    
    expected_volatility = np.sqrt(expected_volatility)
    
    # 샤프 비율
    risk_free_rate = 0.035  # 3.5%
    sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
    
    return {
        'expected_return': expected_return,
        'expected_volatility': expected_volatility,
        'sharpe_ratio': sharpe_ratio,
        'number_of_holdings': len(portfolio),
        'max_weight': max(info['weight'] for info in portfolio.values()) if portfolio else 0
    }

def _generate_fusion_explanation(self, 
                               weights: MasterWeights, 
                               profile: InvestmentProfile, 
                               portfolio: Dict[str, Dict]) -> str:
    """
    융합 포트폴리오 설명 생성
    """
    explanation = f"""
```

🎯 {profile.value.title()} 성향 맞춤 포트폴리오

📊 4대 거장 가중치:
• 워렌 버핏 (가치투자): {weights.buffett:.1%}
• 레이 달리오 (All Weather): {weights.dalio:.1%}
• 리처드 파인만 (과학적 사고): {weights.feynman:.1%}
• 짐 사이먼스 (퀀트): {weights.simons:.1%}

🏆 선택된 {len(portfolio)}개 종목:
“””

```
    # 상위 5개 종목 설명
    sorted_portfolio = sorted(portfolio.items(), key=lambda x: x[1]['weight'], reverse=True)
    
    for i, (ticker, info) in enumerate(sorted_portfolio[:5]):
        explanation += f"  {i+1}. {ticker} ({info['weight']:.1%}) - {info['risk_level']} 리스크\n"
        if 'combined_reasoning' in info:
            explanation += f"     💡 {info['combined_reasoning'][:100]}...\n"
    
    if len(portfolio) > 5:
        explanation += f"  ... 외 {len(portfolio)-5}개 종목\n"
    
    return explanation

def _calculate_fusion_score(self, 
                          portfolio: Dict[str, Dict], 
                          weights: MasterWeights, 
                          risk_metrics: Dict[str, float]) -> float:
    """
    융합 포트폴리오 종합 점수 계산
    """
    score = 0.0
    
    # 1. 분산투자 점수 (30점)
    diversification_score = min(30, len(portfolio) * 2)  # 종목 수 x 2점
    score += diversification_score
    
    # 2. 리스크 조정 수익률 점수 (40점)
    sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
    risk_adjusted_score = min(40, max(0, sharpe_ratio * 20))  # 샤프비율 x 20점
    score += risk_adjusted_score
    
    # 3. 거장들의 합의도 점수 (20점)
    consensus_score = self._calculate_consensus_score(portfolio)
    score += consensus_score
    
    # 4. 리스크 관리 점수 (10점)
    risk_management_score = self._calculate_risk_management_score(portfolio, risk_metrics)
    score += risk_management_score
    
    return min(100.0, score)

def _calculate_consensus_score(self, portfolio: Dict[str, Dict]) -> float:
    """
    거장들의 합의도 점수 계산
    """
    consensus_scores = []
    
    for ticker, info in portfolio.items():
        masters_votes = info.get('masters_votes', {})
        
        if len(masters_votes) >= 2:  # 2명 이상이 추천한 종목
            consensus_scores.append(min(20, len(masters_votes) * 5))  # 추천자 수 x 5점
        else:
            consensus_scores.append(5)  # 기본 점수
    
    return np.mean(consensus_scores) if consensus_scores else 0.0

def _calculate_risk_management_score(self, 
                                   portfolio: Dict[str, Dict], 
                                   risk_metrics: Dict[str, float]) -> float:
    """
    리스크 관리 점수 계산
    """
    score = 0.0
    
    # 최대 집중도 확인
    max_weight = risk_metrics.get('max_weight', 0)
    if max_weight <= 0.15:
        score += 5
    elif max_weight <= 0.20:
        score += 3
    
    # 종목 수 적정성
    num_holdings = len(portfolio)
    if 10 <= num_holdings <= 25:
        score += 5
    elif 5 <= num_holdings <= 30:
        score += 3
    
    return score

# 헬퍼 메서드들
async def _analyze_market_condition(self) -> Dict[str, str]:
    """시장 상황 분석"""
    # 실제 구현에서는 시장 지표들을 분석
    return {
        'trend': 'bullish',      # 상승장
        'volatility': 'medium',  # 중간 변동성
        'sentiment': 'neutral'   # 중립 심리
    }

def _adjust_weights_by_market(self, base_weights: MasterWeights, 
                            market_condition: Dict[str, str]) -> MasterWeights:
    """시장 상황에 따른 가중치 조정"""
    adjusted = MasterWeights(
        buffett=base_weights.buffett,
        dalio=base_weights.dalio,
        feynman=base_weights.feynman,
        simons=base_weights.simons
    )
    
    # 시장 상황별 조정
    if market_condition['trend'] == 'bearish':  # 하락장
        adjusted.buffett *= 1.2  # 가치투자 비중 증가
        adjusted.dalio *= 1.1    # 리스크 패리티 증가
        adjusted.simons *= 0.8   # 퀀트 비중 감소
    elif market_condition['trend'] == 'bullish':  # 상승장
        adjusted.simons *= 1.2   # 퀀트 비중 증가
        adjusted.buffett *= 0.9  # 가치투자 비중 감소
    
    if market_condition['volatility'] == 'high':  # 고변동성
        adjusted.dalio *= 1.2    # All Weather 비중 증가
        adjusted.feynman *= 1.1  # 과학적 사고 증가
    
    adjusted.normalize()
    return adjusted

async def _calculate_performance_adjustment(self, tickers: List[str]) -> Dict[str, float]:
    """최근 성과 기반 조정"""
    # 실제 구현에서는 각 거장의 최근 성과를 계산
    return {
        'buffett': 1.0,
        'dalio': 1.0,
        'feynman': 1.0,
        'simons': 1.0
    }

def _estimate_portfolio_volatility(self, portfolio: Dict[str, Dict]) -> float:
    """포트폴리오 변동성 추정"""
    # 간단한 구현 (가중평균)
    total_vol = 0.0
    
    for ticker, info in portfolio.items():
        weight = info['weight']
        stock_vol = self._estimate_stock_volatility(ticker)
        total_vol += weight * stock_vol
    
    return total_vol

def _estimate_stock_return(self, ticker: str, info: Dict) -> float:
    """종목별 예상 수익률 추정"""
    # 거장들의 예측 평균
    masters_votes = info.get('masters_votes', {})
    
    if not masters_votes:
        return 0.08  # 기본값 8%
    
    # 각 거장의 예상 수익률 (간단한 매핑)
    expected_returns = {
        'buffett': 0.12,   # 가치투자는 높은 수익률 기대
        'dalio': 0.08,     # All Weather는 안정적 수익률
        'feynman': 0.10,   # 과학적 사고는 중간 수익률
        'simons': 0.15     # 퀀트는 높은 수익률 추구
    }
    
    weighted_return = 0.0
    total_weight = 0.0
    
    for master, vote_data in masters_votes.items():
        if master in expected_returns:
            contribution = vote_data.get('contribution', 0)
            weighted_return += contribution * expected_returns[master]
            total_weight += contribution
    
    return weighted_return / total_weight if total_weight > 0 else 0.08

def _estimate_stock_volatility(self, ticker: str) -> float:
    """종목별 변동성 추정"""
    # 실제 구현에서는 과거 데이터 기반 계산
    volatility_map = {
        '005930': 0.25,  # 삼성전자
        '000660': 0.35,  # SK하이닉스
        '035420': 0.30,  # NAVER
        # ... 더 많은 종목
    }
    
    return volatility_map.get(ticker, 0.25)  # 기본값 25%

async def _get_ticker_sector(self, ticker: str) -> str:
    """종목 섹터 조회"""
    # 실제 구현에서는 데이터베이스에서 조회
    sector_map = {
        '005930': '반도체',
        '000660': '반도체',
        '035420': 'IT서비스',
        '055550': '금융',
        # ... 더 많은 매핑
    }
    
    return sector_map.get(ticker, '기타')

async def _get_average_trading_volume(self, ticker: str) -> float:
    """평균 거래대금 조회"""
    # 실제 구현에서는 최근 거래량 데이터 계산
    return 5e8  # 기본값 5억원
```

# 메인 포트폴리오 생성 함수

async def create_masters_fusion_portfolio(
available_tickers: List[str],
profile: InvestmentProfile,
db: Session,
custom_weights: Optional[MasterWeights] = None
) -> Dict:
“””
4대 거장 융합 포트폴리오 생성 메인 함수

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
    
    # 융합 포트폴리오 생성
    result = await fusion_engine.create_fusion_portfolio(
        available_tickers, profile, custom_weights
    )
    
    # 결과 포맷팅
    formatted_result = {
        'portfolio': result.portfolio,
        'strategy': 'Masters Fusion',
        'profile': result.profile.value,
        'master_weights': result.master_weights.to_dict(),
        'total_score': result.total_score,
        'expected_return': result.expected_return,
        'expected_volatility': result.expected_volatility,
        'risk_metrics': result.risk_metrics,
        'explanation': result.explanation,
        'creation_time': datetime.now().isoformat(),
        'rebalance_frequency': 'monthly',
        'philosophy': '4대 투자 거장의 지혜 융합'
    }
    
    logger.info(f"Masters fusion portfolio created: {result.total_score:.1f} score")
    return formatted_result
    
except Exception as e:
    logger.error(f"Error in masters fusion: {str(e)}")
    return {}
```

# 실시간 리밸런싱 함수

async def rebalance_fusion_portfolio(
current_portfolio: Dict,
market_data: Dict,
db: Session
) -> Dict:
“””
융합 포트폴리오 리밸런싱

```
Args:
    current_portfolio: 현재 포트폴리오
    market_data: 시장 데이터
    db: 데이터베이스 세션
    
Returns:
    리밸런싱 권고사항
"""
try:
    # 현재 비중과 목표 비중 비교
    rebalance_signals = []
    
    for ticker, target_info in current_portfolio.get('portfolio', {}).items():
        target_weight = target_info['weight']
        current_weight = market_data.get(ticker, {}).get('current_weight', 0)
        
        # 5% 이상 차이나면 리밸런싱 신호
        weight_diff = abs(current_weight - target_weight)
        if weight_diff > 0.05:
            rebalance_signals.append({
                'ticker': ticker,
                'action': 'buy' if current_weight < target_weight else 'sell',
                'target_weight': target_weight,
                'current_weight': current_weight,
                'adjustment_needed': weight_diff
            })
    
    return {
        'rebalance_needed': len(rebalance_signals) > 0,
        'signals': rebalance_signals,
        'total_adjustments': len(rebalance_signals),
        'timestamp': datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"Error in rebalancing: {str(e)}")
    return {'rebalance_needed': False, 'error': str(e)}
```