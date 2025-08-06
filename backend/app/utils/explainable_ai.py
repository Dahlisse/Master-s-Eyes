“””
설명 가능한 AI (XAI) 시스템 - Master’s Eye
파일 위치: backend/app/utils/explainable_ai.py

모든 AI 투자 결정에 대한 완전 투명성과 설명 가능성 제공
“””

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import logging

logger = logging.getLogger(**name**)

class DecisionType(Enum):
“”“의사결정 유형”””
STOCK_SELECTION = “종목선택”
WEIGHT_ALLOCATION = “비중배분”
RISK_MANAGEMENT = “리스크관리”
TIMING = “타이밍”
REBALANCING = “리밸런싱”

class ExplanationLevel(Enum):
“”“설명 수준”””
SIMPLE = “간단함”      # 초보자용
DETAILED = “상세함”    # 중급자용
TECHNICAL = “기술적”   # 고급자용

@dataclass
class DecisionFactor:
“”“의사결정 요인”””
name: str
weight: float           # 의사결정에서의 가중치
value: float           # 실제 값
impact: float          # 결정에 미친 영향도
master_source: str     # 어느 거장의 관점인지
confidence: float      # 신뢰도 (0~1)
explanation: str       # 텍스트 설명

@dataclass
class DecisionExplanation:
“”“의사결정 설명”””
decision_id: str
decision_type: DecisionType
final_decision: Any
confidence_score: float
primary_factors: List[DecisionFactor]
alternative_options: List[Dict[str, Any]]
risk_factors: List[str]
assumptions: List[str]
sensitivity_analysis: Dict[str, float]
timestamp: str

class MasterContributionAnalyzer:
“”“4대 거장 기여도 분석기”””

```
def __init__(self):
    self.master_weights = {
        "buffett": 0.25,
        "dalio": 0.25, 
        "feynman": 0.25,
        "simons": 0.25
    }

def analyze_decision_contribution(self, 
                               stock_analysis: Dict[str, Any],
                               decision_type: DecisionType) -> List[DecisionFactor]:
    """각 거장의 의사결정 기여도 분석"""
    factors = []
    
    # 버핏 기여도 분석
    buffett_factors = self._analyze_buffett_contribution(
        stock_analysis.get("buffett", {}), decision_type
    )
    factors.extend(buffett_factors)
    
    # 달리오 기여도 분석
    dalio_factors = self._analyze_dalio_contribution(
        stock_analysis.get("dalio", {}), decision_type
    )
    factors.extend(dalio_factors)
    
    # 파인만 기여도 분석
    feynman_factors = self._analyze_feynman_contribution(
        stock_analysis.get("feynman", {}), decision_type
    )
    factors.extend(feynman_factors)
    
    # 사이먼스 기여도 분석
    simons_factors = self._analyze_simons_contribution(
        stock_analysis.get("simons", {}), decision_type
    )
    factors.extend(simons_factors)
    
    # 기여도 순으로 정렬
    factors.sort(key=lambda x: abs(x.impact), reverse=True)
    
    return factors

def _analyze_buffett_contribution(self, 
                                buffett_analysis: Dict,
                                decision_type: DecisionType) -> List[DecisionFactor]:
    """버핏 관점 기여도 분석"""
    factors = []
    
    # 내재가치 분석
    intrinsic_value_score = buffett_analysis.get("intrinsic_value_score", 0)
    if intrinsic_value_score != 0:
        factors.append(DecisionFactor(
            name="내재가치 평가",
            weight=0.3,
            value=intrinsic_value_score,
            impact=intrinsic_value_score * 0.3 * self.master_weights["buffett"],
            master_source="워렌 버핏",
            confidence=buffett_analysis.get("confidence", 0.5),
            explanation=self._get_intrinsic_value_explanation(intrinsic_value_score)
        ))
    
    # 경제적 해자
    moat_score = buffett_analysis.get("moat_score", 0)
    if moat_score != 0:
        factors.append(DecisionFactor(
            name="경제적 해자",
            weight=0.25,
            value=moat_score,
            impact=moat_score * 0.25 * self.master_weights["buffett"],
            master_source="워렌 버핏", 
            confidence=buffett_analysis.get("confidence", 0.5),
            explanation=self._get_moat_explanation(moat_score)
        ))
    
    # 재무 건전성
    financial_health = buffett_analysis.get("financial_health_score", 0)
    if financial_health != 0:
        factors.append(DecisionFactor(
            name="재무 건전성",
            weight=0.25,
            value=financial_health,
            impact=financial_health * 0.25 * self.master_weights["buffett"],
            master_source="워렌 버핏",
            confidence=buffett_analysis.get("confidence", 0.5),
            explanation=self._get_financial_health_explanation(financial_health)
        ))
    
    # 경영진 평가
    management_score = buffett_analysis.get("management_score", 0)
    if management_score != 0:
        factors.append(DecisionFactor(
            name="경영진 역량",
            weight=0.2,
            value=management_score,
            impact=management_score * 0.2 * self.master_weights["buffett"],
            master_source="워렌 버핏",
            confidence=buffett_analysis.get("confidence", 0.5),
            explanation=self._get_management_explanation(management_score)
        ))
    
    return factors

def _analyze_dalio_contribution(self, 
                              dalio_analysis: Dict,
                              decision_type: DecisionType) -> List[DecisionFactor]:
    """달리오 관점 기여도 분석"""
    factors = []
    
    # 경제 사이클 분석
    cycle_score = dalio_analysis.get("economic_cycle_score", 0)
    if cycle_score != 0:
        factors.append(DecisionFactor(
            name="경제 사이클 적합성",
            weight=0.35,
            value=cycle_score,
            impact=cycle_score * 0.35 * self.master_weights["dalio"],
            master_source="레이 달리오",
            confidence=dalio_analysis.get("confidence", 0.5),
            explanation=self._get_cycle_explanation(cycle_score)
        ))
    
    # 거시경제 요인
    macro_score = dalio_analysis.get("macro_score", 0)
    if macro_score != 0:
        factors.append(DecisionFactor(
            name="거시경제 환경",
            weight=0.3,
            value=macro_score,
            impact=macro_score * 0.3 * self.master_weights["dalio"],
            master_source="레이 달리오",
            confidence=dalio_analysis.get("confidence", 0.5),
            explanation=self._get_macro_explanation(macro_score)
        ))
    
    # 상관관계 분석
    correlation_score = dalio_analysis.get("correlation_score", 0)
    if correlation_score != 0:
        factors.append(DecisionFactor(
            name="포트폴리오 분산 효과",
            weight=0.2,
            value=correlation_score,
            impact=correlation_score * 0.2 * self.master_weights["dalio"],
            master_source="레이 달리오",
            confidence=dalio_analysis.get("confidence", 0.5),
            explanation=self._get_correlation_explanation(correlation_score)
        ))
    
    # 테일 리스크
    tail_risk_score = dalio_analysis.get("tail_risk_score", 0)
    if tail_risk_score != 0:
        factors.append(DecisionFactor(
            name="극한 위험 관리",
            weight=0.15,
            value=tail_risk_score,
            impact=tail_risk_score * 0.15 * self.master_weights["dalio"],
            master_source="레이 달리오",
            confidence=dalio_analysis.get("confidence", 0.5),
            explanation=self._get_tail_risk_explanation(tail_risk_score)
        ))
    
    return factors

def _analyze_feynman_contribution(self, 
                                feynman_analysis: Dict,
                                decision_type: DecisionType) -> List[DecisionFactor]:
    """파인만 관점 기여도 분석"""
    factors = []
    
    # 불확실성 정량화
    uncertainty_score = feynman_analysis.get("uncertainty_score", 0)
    if uncertainty_score != 0:
        factors.append(DecisionFactor(
            name="불확실성 수준",
```