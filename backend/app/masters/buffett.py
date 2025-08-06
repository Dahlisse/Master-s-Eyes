“””
워렌 버핏 가치투자 알고리즘 구현

- DCF 모델 및 내재가치 계산
- 경제적 해자 평가 시스템
- 안전마진 계산
- 사업 이해도 평가
  “””

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
import asyncio
from sqlalchemy.orm import Session

from app.models.company import Company
from app.models.market import MarketData
from app.utils.calculations import calculate_wacc, calculate_fcf_growth, calculate_roe
from app.core.logging import get_logger

logger = get_logger(**name**)

@dataclass
class BuffettScore:
“”“버핏 평가 점수 데이터 클래스”””
total_score: float
intrinsic_value: float
current_price: float
margin_of_safety: float
moat_score: float
management_score: float
understanding_score: float
valuation_score: float
quality_score: float

```
def to_dict(self) -> Dict:
    return {
        'total_score': self.total_score,
        'intrinsic_value': self.intrinsic_value,
        'current_price': self.current_price,
        'margin_of_safety': self.margin_of_safety,
        'moat_score': self.moat_score,
        'management_score': self.management_score,
        'understanding_score': self.understanding_score,
        'valuation_score': self.valuation_score,
        'quality_score': self.quality_score
    }
```

class BuffettValueInvestor:
“””
워렌 버핏의 가치투자 철학을 구현한 클래스

```
핵심 원칙:
1. Circle of Competence - 이해 가능한 사업
2. Economic Moat - 경제적 해자
3. Excellent Management - 뛰어난 경영진
4. Reasonable Price - 합리적 가격
"""

def __init__(self, db: Session):
    self.db = db
    self.min_roe = 0.15  # 최소 ROE 15%
    self.min_debt_ratio = 0.5  # 최대 부채비율 50%
    self.min_safety_margin = 0.2  # 최소 안전마진 20%
    self.min_years_data = 5  # 최소 5년 데이터
    
async def evaluate_company(self, ticker: str) -> Optional[BuffettScore]:
    """
    기업에 대한 버핏 스타일 종합 평가
    
    Args:
        ticker: 종목 코드
        
    Returns:
        BuffettScore 객체 또는 None
    """
    try:
        # 기업 기본 정보 가져오기
        company = self.db.query(Company).filter(Company.ticker == ticker).first()
        if not company:
            logger.warning(f"Company not found: {ticker}")
            return None
        
        # 재무 데이터 수집
        financial_data = await self._get_financial_data(ticker)
        if not financial_data:
            return None
        
        # 시장 데이터 수집
        market_data = await self._get_market_data(ticker)
        if not market_data:
            return None
        
        # 1. 사업 이해도 평가 (Circle of Competence)
        understanding_score = self._evaluate_business_understanding(company, financial_data)
        
        # 2. 경제적 해자 평가 (Economic Moat)
        moat_score = self._evaluate_economic_moat(financial_data)
        
        # 3. 경영진 평가 (Management Quality)
        management_score = self._evaluate_management(financial_data)
        
        # 4. 사업 품질 평가 (Business Quality)
        quality_score = self._evaluate_business_quality(financial_data)
        
        # 5. 내재가치 계산 (Intrinsic Value)
        intrinsic_value = self._calculate_intrinsic_value(financial_data, ticker)
        
        # 6. 밸류에이션 평가
        current_price = market_data['current_price']
        valuation_score, margin_of_safety = self._evaluate_valuation(
            intrinsic_value, current_price
        )
        
        # 7. 종합 점수 계산
        total_score = self._calculate_total_score(
            understanding_score, moat_score, management_score,
            quality_score, valuation_score
        )
        
        return BuffettScore(
            total_score=total_score,
            intrinsic_value=intrinsic_value,
            current_price=current_price,
            margin_of_safety=margin_of_safety,
            moat_score=moat_score,
            management_score=management_score,
            understanding_score=understanding_score,
            valuation_score=valuation_score,
            quality_score=quality_score
        )
        
    except Exception as e:
        logger.error(f"Error evaluating company {ticker}: {str(e)}")
        return None

def _evaluate_business_understanding(self, company: Company, financial_data: Dict) -> float:
    """
    사업 이해도 평가 (Circle of Competence)
    
    평가 기준:
    - 사업 모델의 단순성
    - 업종의 예측 가능성
    - 주력 사업 집중도
    """
    score = 0.0
    
    # 업종별 이해도 점수 (한국 시장 특화)
    sector_scores = {
        '은행': 0.9,           # 단순한 사업 모델
        '보험': 0.8,           # 이해 가능
        '유통': 0.8,           # 소비자 중심
        '식품': 0.9,           # 필수재
        '제약': 0.7,           # 복잡하지만 필수재
        '화학': 0.6,           # 복잡한 사업
        '반도체': 0.4,         # 매우 복잡하고 변동성 큼
        '조선': 0.5,           # 사이클리컬
        '자동차': 0.6,         # 복잡한 공급망
        '통신': 0.8,           # 안정적 사업
        '전력': 0.9,           # 유틸리티
    }
    
    sector_score = sector_scores.get(company.sector, 0.5)
    score += sector_score * 30  # 30점 만점
    
    # 매출 집중도 평가
    # 주력 사업 비중이 높을수록 좋음
    if 'revenue_concentration' in financial_data:
        concentration = financial_data['revenue_concentration']
        if concentration > 0.7:
            score += 25  # 25점 추가
        elif concentration > 0.5:
            score += 15
        else:
            score += 5
    else:
        score += 15  # 기본 점수
    
    # 사업 지속성 평가
    revenue_stability = self._calculate_revenue_stability(financial_data)
    score += revenue_stability * 25  # 25점 만점
    
    # 경쟁 구조 평가
    competition_score = self._evaluate_competition_structure(company.sector)
    score += competition_score * 20  # 20점 만점
    
    return min(score, 100.0)

def _evaluate_economic_moat(self, financial_data: Dict) -> float:
    """
    경제적 해자 평가
    
    평가 기준:
    1. 브랜드 파워 (가격 결정력)
    2. 네트워크 효과
    3. 높은 전환 비용
    4. 비용 우위
    5. 규제 장벽
    """
    score = 0.0
    
    # 1. 수익성 지속성 (ROE 트렌드)
    roe_trend = self._calculate_roe_trend(financial_data)
    if roe_trend > 0.15 and self._is_roe_stable(financial_data):
        score += 25  # 지속적 고수익성
    elif roe_trend > 0.10:
        score += 15
    else:
        score += 5
    
    # 2. 마진 안정성 (Operating Margin)
    margin_stability = self._calculate_margin_stability(financial_data)
    score += margin_stability * 20  # 20점 만점
    
    # 3. 시장 점유율 및 경쟁 우위
    market_position_score = self._evaluate_market_position(financial_data)
    score += market_position_score * 20  # 20점 만점
    
    # 4. 가격 결정력 (Price Power)
    price_power = self._evaluate_price_power(financial_data)
    score += price_power * 15  # 15점 만점
    
    # 5. 자본 효율성 (Asset Turnover)
    asset_efficiency = self._evaluate_asset_efficiency(financial_data)
    score += asset_efficiency * 10  # 10점 만점
    
    # 6. 진입 장벽 평가
    entry_barrier_score = self._evaluate_entry_barriers(financial_data)
    score += entry_barrier_score * 10  # 10점 만점
    
    return min(score, 100.0)

def _evaluate_management(self, financial_data: Dict) -> float:
    """
    경영진 평가
    
    평가 기준:
    1. 주주 친화적 정책
    2. 자본 배분 능력
    3. 정직성과 투명성
    4. 장기적 사고
    """
    score = 0.0
    
    # 1. 배당 정책 평가
    dividend_score = self._evaluate_dividend_policy(financial_data)
    score += dividend_score * 25  # 25점 만점
    
    # 2. 자사주 매입 정책
    buyback_score = self._evaluate_buyback_policy(financial_data)
    score += buyback_score * 20  # 20점 만점
    
    # 3. 부채 관리 능력
    debt_management_score = self._evaluate_debt_management(financial_data)
    score += debt_management_score * 25  # 25점 만점
    
    # 4. 수익성 개선 트렌드
    profitability_trend = self._calculate_profitability_trend(financial_data)
    score += profitability_trend * 15  # 15점 만점
    
    # 5. 자본 배분 효율성
    capital_allocation_score = self._evaluate_capital_allocation(financial_data)
    score += capital_allocation_score * 15  # 15점 만점
    
    return min(score, 100.0)

def _evaluate_business_quality(self, financial_data: Dict) -> float:
    """
    사업 품질 평가
    
    평가 기준:
    1. 수익성 지표
    2. 재무 안정성
    3. 성장성
    4. 현금 창출 능력
    """
    score = 0.0
    
    # 1. ROE 평가 (5년 평균)
    avg_roe = np.mean(financial_data.get('roe_history', [0]))
    if avg_roe > 0.20:
        score += 30
    elif avg_roe > 0.15:
        score += 25
    elif avg_roe > 0.10:
        score += 15
    else:
        score += 5
    
    # 2. 부채비율 평가
    debt_ratio = financial_data.get('debt_ratio', 1.0)
    if debt_ratio < 0.3:
        score += 25
    elif debt_ratio < 0.5:
        score += 20
    elif debt_ratio < 0.7:
        score += 10
    else:
        score += 0
    
    # 3. 현금흐름 품질
    fcf_quality = self._evaluate_fcf_quality(financial_data)
    score += fcf_quality * 25  # 25점 만점
    
    # 4. 매출 성장 품질
    revenue_growth_quality = self._evaluate_revenue_growth_quality(financial_data)
    score += revenue_growth_quality * 20  # 20점 만점
    
    return min(score, 100.0)

def _calculate_intrinsic_value(self, financial_data: Dict, ticker: str) -> float:
    """
    DCF 모델을 사용한 내재가치 계산
    
    단계:
    1. 과거 FCF 분석
    2. 성장률 추정
    3. 할인율 계산 (WACC)
    4. 터미널 가치 계산
    5. 현재가치 할인
    """
    try:
        # 기본 데이터 확인
        fcf_history = financial_data.get('fcf_history', [])
        if len(fcf_history) < 3:
            logger.warning(f"Insufficient FCF data for {ticker}")
            return 0.0
        
        # 1. FCF 성장률 계산
        fcf_growth_rate = self._calculate_fcf_growth_rate(fcf_history)
        
        # 2. 할인율 계산 (WACC)
        wacc = self._calculate_wacc(financial_data, ticker)
        
        # 3. 미래 FCF 예측 (10년)
        current_fcf = fcf_history[-1]
        projected_fcf = []
        
        # 첫 5년: 추정 성장률
        conservative_growth = min(fcf_growth_rate, 0.15)  # 최대 15%로 제한
        for year in range(1, 6):
            projected_fcf.append(current_fcf * ((1 + conservative_growth) ** year))
        
        # 다음 5년: 성장률 점진적 감소
        terminal_growth = 0.03  # 장기 성장률 3%
        for year in range(6, 11):
            fade_rate = conservative_growth * (1 - (year - 5) * 0.2)
            fade_rate = max(fade_rate, terminal_growth)
            projected_fcf.append(current_fcf * ((1 + fade_rate) ** year))
        
        # 4. 터미널 가치 계산
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        
        # 5. 현재가치 할인
        pv_fcf = sum([fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(projected_fcf)])
        pv_terminal = terminal_value / ((1 + wacc) ** 10)
        
        enterprise_value = pv_fcf + pv_terminal
        
        # 6. 주식가치 계산
        net_cash = financial_data.get('net_cash', 0)
        shares_outstanding = financial_data.get('shares_outstanding', 1)
        
        equity_value = enterprise_value + net_cash
        intrinsic_value_per_share = equity_value / shares_outstanding
        
        logger.info(f"DCF calculation for {ticker}: {intrinsic_value_per_share:.2f}")
        return intrinsic_value_per_share
        
    except Exception as e:
        logger.error(f"Error calculating intrinsic value for {ticker}: {str(e)}")
        return 0.0

def _evaluate_valuation(self, intrinsic_value: float, current_price: float) -> Tuple[float, float]:
    """
    밸류에이션 평가 및 안전마진 계산
    """
    if intrinsic_value <= 0 or current_price <= 0:
        return 0.0, 0.0
    
    margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
    
    # 밸류에이션 점수 계산
    if margin_of_safety >= 0.3:  # 30% 이상 할인
        valuation_score = 100
    elif margin_of_safety >= 0.2:  # 20% 이상 할인
        valuation_score = 80
    elif margin_of_safety >= 0.1:  # 10% 이상 할인
        valuation_score = 60
    elif margin_of_safety >= 0:  # 공정가치
        valuation_score = 40
    else:  # 고평가
        valuation_score = max(0, 40 + margin_of_safety * 100)
    
    return valuation_score, margin_of_safety

def _calculate_total_score(self, understanding: float, moat: float, 
                         management: float, quality: float, valuation: float) -> float:
    """
    버핏 스타일 종합 점수 계산
    
    가중치:
    - 사업 이해도: 20%
    - 경제적 해자: 25%
    - 경영진: 20%
    - 사업 품질: 25%
    - 밸류에이션: 10%
    """
    weights = {
        'understanding': 0.20,
        'moat': 0.25,
        'management': 0.20,
        'quality': 0.25,
        'valuation': 0.10
    }
    
    total_score = (
        understanding * weights['understanding'] +
        moat * weights['moat'] +
        management * weights['management'] +
        quality * weights['quality'] +
        valuation * weights['valuation']
    )
    
    return min(total_score, 100.0)

# 헬퍼 메서드들 (상세 구현)
async def _get_financial_data(self, ticker: str) -> Optional[Dict]:
    """재무 데이터 수집"""
    # 실제 구현에서는 데이터베이스나 API에서 가져옴
    pass

async def _get_market_data(self, ticker: str) -> Optional[Dict]:
    """시장 데이터 수집"""
    # 실제 구현에서는 실시간 API에서 가져옴
    pass

def _calculate_revenue_stability(self, financial_data: Dict) -> float:
    """매출 안정성 계산"""
    revenue_history = financial_data.get('revenue_history', [])
    if len(revenue_history) < 5:
        return 0.5
    
    # 매출 변동계수 계산 (CV = std/mean)
    cv = np.std(revenue_history) / np.mean(revenue_history)
    
    # CV가 낮을수록 안정적
    if cv < 0.1:
        return 1.0
    elif cv < 0.2:
        return 0.8
    elif cv < 0.3:
        return 0.6
    else:
        return 0.3

def _evaluate_competition_structure(self, sector: str) -> float:
    """업종 경쟁 구조 평가"""
    # 업종별 경쟁 강도 점수
    competition_scores = {
        '은행': 0.7,           # 진입장벽 높음
        '보험': 0.8,           # 규제 장벽
        '유통': 0.5,           # 경쟁 치열
        '식품': 0.7,           # 브랜드 중요
        '제약': 0.8,           # 특허 보호
        '화학': 0.6,           # 원료 의존
        '반도체': 0.4,         # 기술 변화 빠름
        '조선': 0.3,           # 과잉 공급
        '자동차': 0.5,         # 글로벌 경쟁
        '통신': 0.8,           # 네트워크 효과
        '전력': 0.9,           # 독점적 지위
    }
    
    return competition_scores.get(sector, 0.5)

def _calculate_roe_trend(self, financial_data: Dict) -> float:
    """ROE 트렌드 계산"""
    roe_history = financial_data.get('roe_history', [])
    if len(roe_history) < 3:
        return 0.0
    
    return np.mean(roe_history[-3:])  # 최근 3년 평균

def _is_roe_stable(self, financial_data: Dict) -> bool:
    """ROE 안정성 확인"""
    roe_history = financial_data.get('roe_history', [])
    if len(roe_history) < 5:
        return False
    
    # 5년 중 3년 이상 15% 이상이면 안정적
    high_roe_years = sum(1 for roe in roe_history if roe > 0.15)
    return high_roe_years >= 3

def _calculate_margin_stability(self, financial_data: Dict) -> float:
    """마진 안정성 계산"""
    margin_history = financial_data.get('operating_margin_history', [])
    if len(margin_history) < 3:
        return 0.5
    
    # 마진 변동계수 계산
    cv = np.std(margin_history) / np.mean(margin_history)
    
    if cv < 0.2:
        return 1.0
    elif cv < 0.4:
        return 0.7
    else:
        return 0.3

def _evaluate_market_position(self, financial_data: Dict) -> float:
    """시장 지위 평가"""
    # 시장 점유율, 매출 규모 등을 고려
    market_cap_rank = financial_data.get('market_cap_rank', 100)
    
    if market_cap_rank <= 10:
        return 1.0
    elif market_cap_rank <= 30:
        return 0.8
    elif market_cap_rank <= 100:
        return 0.6
    else:
        return 0.3

def _evaluate_price_power(self, financial_data: Dict) -> float:
    """가격 결정력 평가"""
    # 마진 개선 능력으로 평가
    margin_trend = financial_data.get('margin_trend', 0)
    
    if margin_trend > 0.02:  # 2%p 이상 개선
        return 1.0
    elif margin_trend > 0:
        return 0.7
    elif margin_trend > -0.01:
        return 0.5
    else:
        return 0.2

def _evaluate_asset_efficiency(self, financial_data: Dict) -> float:
    """자산 효율성 평가"""
    asset_turnover = financial_data.get('asset_turnover', 0)
    
    if asset_turnover > 1.5:
        return 1.0
    elif asset_turnover > 1.0:
        return 0.8
    elif asset_turnover > 0.5:
        return 0.6
    else:
        return 0.3

def _evaluate_entry_barriers(self, financial_data: Dict) -> float:
    """진입 장벽 평가"""
    # 자본 집약도, 규제 등을 고려
    capex_ratio = financial_data.get('capex_to_revenue', 0)
    
    if capex_ratio > 0.1:  # 자본 집약적
        return 0.8
    else:
        return 0.5

def _evaluate_dividend_policy(self, financial_data: Dict) -> float:
    """배당 정책 평가"""
    dividend_history = financial_data.get('dividend_history', [])
    if len(dividend_history) < 5:
        return 0.5
    
    # 배당 성장성과 지속성
    dividend_growth = (dividend_history[-1] - dividend_history[0]) / dividend_history[0]
    dividend_cuts = sum(1 for i in range(1, len(dividend_history)) 
                      if dividend_history[i] < dividend_history[i-1])
    
    if dividend_growth > 0.05 and dividend_cuts == 0:
        return 1.0
    elif dividend_growth > 0 and dividend_cuts <= 1:
        return 0.7
    else:
        return 0.3

def _evaluate_buyback_policy(self, financial_data: Dict) -> float:
    """자사주 매입 정책 평가"""
    buyback_history = financial_data.get('buyback_history', [])
    
    if not buyback_history:
        return 0.5
    
    # 주식 수 감소 여부
    shares_reduction = sum(buyback_history) / financial_data.get('shares_outstanding', 1)
    
    if shares_reduction > 0.02:  # 2% 이상 감소
        return 1.0
    elif shares_reduction > 0:
        return 0.7
    else:
        return 0.3

def _evaluate_debt_management(self, financial_data: Dict) -> float:
    """부채 관리 평가"""
    debt_trend = financial_data.get('debt_trend', 0)
    interest_coverage = financial_data.get('interest_coverage', 0)
    
    score = 0.0
    
    # 부채 감소 트렌드
    if debt_trend < -0.05:
        score += 0.5
    elif debt_trend < 0:
        score += 0.3
    
    # 이자보상배율
    if interest_coverage > 10:
        score += 0.5
    elif interest_coverage > 5:
        score += 0.3
    elif interest_coverage > 2:
        score += 0.1
    
    return score

def _calculate_profitability_trend(self, financial_data: Dict) -> float:
    """수익성 트렌드 계산"""
    roe_history = financial_data.get('roe_history', [])
    if len(roe_history) < 3:
        return 0.5
    
    # 최근 3년 ROE 트렌드
    recent_trend = np.polyfit(range(len(roe_history[-3:])), roe_history[-3:], 1)[0]
    
    if recent_trend > 0.02:
        return 1.0
    elif recent_trend > 0:
        return 0.7
    elif recent_trend > -0.02:
        return 0.5
    else:
        return 0.2

def _evaluate_capital_allocation(self, financial_data: Dict) -> float:
    """자본 배분 효율성 평가"""
    roic = financial_data.get('roic', 0)
    wacc = financial_data.get('wacc', 0.1)
    
    if roic > wacc * 1.5:
        return 1.0
    elif roic > wacc:
        return 0.7
    else:
        return 0.3

def _evaluate_fcf_quality(self, financial_data: Dict) -> float:
    """현금흐름 품질 평가"""
    fcf_history = financial_data.get('fcf_history', [])
    net_income_history = financial_data.get('net_income_history', [])
    
    if len(fcf_history) < 3 or len(net_income_history) < 3:
        return 0.5
    
    # FCF/순이익 비율
    fcf_to_ni_ratio = np.mean([fcf/ni for fcf, ni in 
                              zip(fcf_history[-3:], net_income_history[-3:]) if ni > 0])
    
    if fcf_to_ni_ratio > 1.2:
        return 1.0
    elif fcf_to_ni_ratio > 1.0:
        return 0.8
    elif fcf_to_ni_ratio > 0.8:
        return 0.6
    else:
        return 0.3

def _evaluate_revenue_growth_quality(self, financial_data: Dict) -> float:
    """매출 성장 품질 평가"""
    revenue_history = financial_data.get('revenue_history', [])
    if len(revenue_history) < 5:
        return 0.5
    
    # 지속적인 성장 여부
    growth_years = sum(1 for i in range(1, len(revenue_history)) 
                      if revenue_history[i] > revenue_history[i-1])
    
    growth_rate = growth_years / (len(revenue_history) - 1)
    
    if growth_rate > 0.8:
```