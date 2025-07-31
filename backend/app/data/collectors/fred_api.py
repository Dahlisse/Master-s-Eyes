# backend/app/data/collectors/fred_api.py

“””
FRED (Federal Reserve Economic Data) API 연동 모듈
거시경제 지표 수집 및 레이 달리오 All Weather 전략용 데이터 제공
“””
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
import json
import numpy as np

from app.core.config import get_settings
from app.core.redis import get_redis_client
from app.data.processors.data_validator import DataValidator

settings = get_settings()
logger = logging.getLogger(**name**)

@dataclass
class EconomicIndicator:
“”“경제 지표 데이터 구조”””
series_id: str
title: str
value: float
date: datetime
units: str
frequency: str
seasonal_adjustment: str
last_updated: datetime

@dataclass
class EconomicCycle:
“”“경제 사이클 분석 결과”””
growth_trend: str  # ‘expanding’, ‘peak’, ‘contracting’, ‘trough’
inflation_trend: str  # ‘rising’, ‘falling’, ‘stable’
cycle_phase: str  # ‘recovery’, ‘expansion’, ‘peak’, ‘contraction’
confidence_score: float  # 0-1
key_indicators: Dict[str, float]

class FREDCollector:
“”“FRED API 데이터 수집기 - 레이 달리오 스타일”””

```
def __init__(self):
    self.api_key = settings.FRED_API_KEY if hasattr(settings, 'FRED_API_KEY') else "your_fred_api_key"
    self.base_url = "https://api.stlouisfed.org/fred"
    self.redis_client = get_redis_client()
    self.validator = DataValidator()
    self.session = None
    
    # 레이 달리오 Economic Machine 핵심 지표들
    self.core_indicators = {
        # GDP 및 성장 지표
        'GDP': {
            'series_id': 'GDP',
            'title': 'Gross Domestic Product',
            'category': 'growth',
            'weight': 1.0,
            'dalio_importance': 'high'
        },
        'GDPC1': {
            'series_id': 'GDPC1', 
            'title': 'Real GDP',
            'category': 'growth',
            'weight': 1.0,
            'dalio_importance': 'high'
        },
        'NYGDPMKTPCDWLD': {
            'series_id': 'NYGDPMKTPCDWLD',
            'title': 'World GDP',
            'category': 'growth',
            'weight': 0.7,
            'dalio_importance': 'medium'
        },
        
        # 인플레이션 지표 (핵심!)
        'CPIAUCNS': {
            'series_id': 'CPIAUCNS',
            'title': 'Consumer Price Index',
            'category': 'inflation',
            'weight': 1.0,
            'dalio_importance': 'high'
        },
        'CPILFESL': {
            'series_id': 'CPILFESL',
            'title': 'Core CPI',
            'category': 'inflation', 
            'weight': 1.0,
            'dalio_importance': 'high'
        },
        'PCEPI': {
            'series_id': 'PCEPI',
            'title': 'PCE Price Index',
            'category': 'inflation',
            'weight': 0.9,
            'dalio_importance': 'high'
        },
        'PCEPILFE': {
            'series_id': 'PCEPILFE',
            'title': 'Core PCE Price Index',
            'category': 'inflation',
            'weight': 0.9,
            'dalio_importance': 'high'
        },
        
        # 고용 지표
        'UNRATE': {
            'series_id': 'UNRATE',
            'title': 'Unemployment Rate',
            'category': 'employment',
            'weight': 0.9,
            'dalio_importance': 'high'
        },
        'PAYEMS': {
            'series_id': 'PAYEMS',
            'title': 'Nonfarm Payrolls',
            'category': 'employment',
            'weight': 0.8,
            'dalio_importance': 'medium'
        },
        'CIVPART': {
            'series_id': 'CIVPART',
            'title': 'Labor Force Participation Rate',
            'category': 'employment',
            'weight': 0.6,
            'dalio_importance': 'medium'
        },
        
        # 금리 및 통화정책 (핵심!)
        'FEDFUNDS': {
            'series_id': 'FEDFUNDS',
            'title': 'Federal Funds Rate',
            'category': 'monetary_policy',
            'weight': 1.0,
            'dalio_importance': 'high'
        },
        'DGS10': {
            'series_id': 'DGS10',
            'title': '10-Year Treasury Rate',
            'category': 'monetary_policy',
            'weight': 1.0,
            'dalio_importance': 'high'
        },
        'DGS2': {
            'series_id': 'DGS2',
            'title': '2-Year Treasury Rate',
            'category': 'monetary_policy',
            'weight': 0.8,
            'dalio_importance': 'medium'
        },
        'T10Y2Y': {
            'series_id': 'T10Y2Y',
            'title': '10Y-2Y Treasury Spread',
            'category': 'monetary_policy',
            'weight': 0.9,
            'dalio_importance': 'high'  # 경기침체 예측
        },
        
        # 신용 및 부채 (달리오 핵심 관심사)
        'TOTLL': {
            'series_id': 'TOTLL',
            'title': 'Total Consumer Loans',
            'category': 'credit',
            'weight': 0.7,
            'dalio_importance': 'high'
        },
        'GFDEBTN': {
            'series_id': 'GFDEBTN',
            'title': 'Federal Debt Total',
            'category': 'debt',
            'weight': 0.8,
            'dalio_importance': 'high'
        },
        'GFDEGDQ188S': {
            'series_id': 'GFDEGDQ188S',
            'title': 'Federal Debt to GDP Ratio',
            'category': 'debt',
            'weight': 0.9,
            'dalio_importance': 'high'
        },
        
        # 생산성 및 혁신
        'OPHNFB': {
            'series_id': 'OPHNFB',
            'title': 'Nonfarm Business Sector: Output Per Hour',
            'category': 'productivity',
            'weight': 0.8,
            'dalio_importance': 'medium'
        },
        'PRS85006092': {
            'series_id': 'PRS85006092',
            'title': 'Labor Productivity',
            'category': 'productivity',
            'weight': 0.7,
            'dalio_importance': 'medium'
        },
        
        # 국제 무역 및 경상수지
        'NETEXP': {
            'series_id': 'NETEXP',
            'title': 'Net Exports',
            'category': 'trade',
            'weight': 0.6,
            'dalio_importance': 'medium'
        },
        'IMPGS': {
            'series_id': 'IMPGS',
            'title': 'Imports of Goods and Services',
            'category': 'trade',
            'weight': 0.5,
            'dalio_importance': 'low'
        },
        'EXPGS': {
            'series_id': 'EXPGS',
            'title': 'Exports of Goods and Services',
            'category': 'trade',
            'weight': 0.5,
            'dalio_importance': 'low'
        },
        
        # 소비자 신뢰 및 심리
        'UMCSENT': {
            'series_id': 'UMCSENT',
            'title': 'University of Michigan Consumer Sentiment',
            'category': 'sentiment',
            'weight': 0.6,
            'dalio_importance': 'medium'
        },
        
        # 주택 시장 (자산 버블 모니터링)
        'CSUSHPISA': {
            'series_id': 'CSUSHPISA',
            'title': 'Case-Shiller Home Price Index',
            'category': 'housing',
            'weight': 0.7,
            'dalio_importance': 'medium'
        },
        'HOUST': {
            'series_id': 'HOUST',
            'title': 'Housing Starts',
            'category': 'housing',
            'weight': 0.6,
            'dalio_importance': 'low'
        },
        
        # 통화량 (QE 모니터링)
        'BOGMBASE': {
            'series_id': 'BOGMBASE',
            'title': 'Monetary Base',
            'category': 'money_supply',
            'weight': 0.8,
            'dalio_importance': 'high'
        },
        'M2SL': {
            'series_id': 'M2SL',
            'title': 'M2 Money Stock',
            'category': 'money_supply',
            'weight': 0.7,
            'dalio_importance': 'medium'
        }
    }
    
    # 국제 경제 지표 (글로벌 연관성)
    self.international_indicators = {
        'CHNGDPNQDSMEI': {
            'series_id': 'CHNGDPNQDSMEI',
            'title': 'China GDP',
            'country': 'China',
            'weight': 0.9
        },
        'JPNRGDPEXP': {
            'series_id': 'JPNRGDPEXP',
            'title': 'Japan Real GDP',
            'country': 'Japan', 
            'weight': 0.6
        },
        'CLVMNACSCAB1GQEA19': {
            'series_id': 'CLVMNACSCAB1GQEA19',
            'title': 'Euro Area Real GDP',
            'country': 'Euro Area',
            'weight': 0.7
        }
    }

async def __aenter__(self):
    """비동기 컨텍스트 매니저 진입"""
    self.session = aiohttp.ClientSession()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    """비동기 컨텍스트 매니저 종료"""
    if self.session:
        await self.session.close()

async def get_economic_indicator(
    self, 
    series_id: str, 
    limit: int = 100,
    start_date: Optional[str] = None
) -> Optional[EconomicIndicator]:
    """개별 경제 지표 데이터 수집"""
    try:
        # 캐시 확인
        cache_key = f"fred:{series_id}:{limit}"
        cached_data = await self.redis_client.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            return EconomicIndicator(**data)
        
        # FRED API 호출
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'limit': limit,
            'sort_order': 'desc'
        }
        
        if start_date:
            params['start_date'] = start_date
        
        # 시리즈 메타데이터 조회
        series_url = f"{self.base_url}/series"
        observations_url = f"{self.base_url}/series/observations"
        
        async with self.session.get(series_url, params={'series_id': series_id, 'api_key': self.api_key, 'file_type': 'json'}) as response:
            if response.status != 200:
                logger.error(f"Failed to get series metadata for {series_id}: {response.status}")
                return None
            
            series_data = await response.json()
            series_info = series_data['seriess'][0]
        
        # 관측값 조회
        async with self.session.get(observations_url, params=params) as response:
            if response.status != 200:
                logger.error(f"Failed to get observations for {series_id}: {response.status}")
                return None
            
            obs_data = await response.json()
            observations = obs_data['observations']
            
            if not observations:
                logger.warning(f"No observations found for {series_id}")
                return None
            
            # 최신 유효한 데이터 찾기
            latest_obs = None
            for obs in observations:
                if obs['value'] != '.':  # FRED에서 '.'는 결측값
                    latest_obs = obs
                    break
            
            if not latest_obs:
                logger.warning(f"No valid observations found for {series_id}")
                return None
            
            # EconomicIndicator 객체 생성
            indicator = EconomicIndicator(
                series_id=series_id,
                title=series_info['title'],
                value=float(latest_obs['value']),
                date=datetime.strptime(latest_obs['date'], '%Y-%m-%d'),
                units=series_info['units'],
                frequency=series_info['frequency'],
                seasonal_adjustment=series_info.get('seasonal_adjustment', 'Not Seasonally Adjusted'),
                last_updated=datetime.strptime(series_info['last_updated'], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
            )
            
            # Redis 캐시 (1시간 TTL)
            await self.redis_client.setex(
                cache_key,
                3600,
                json.dumps({
                    'series_id': indicator.series_id,
                    'title': indicator.title,  
                    'value': indicator.value,
                    'date': indicator.date.isoformat(),
                    'units': indicator.units,
                    'frequency': indicator.frequency,
                    'seasonal_adjustment': indicator.seasonal_adjustment,
                    'last_updated': indicator.last_updated.isoformat()
                })
            )
            
            logger.info(f"Collected {series_id}: {indicator.value} {indicator.units}")
            return indicator
            
    except Exception as e:
        logger.error(f"Error collecting {series_id}: {e}")
        return None

async def collect_all_core_indicators(self) -> Dict[str, EconomicIndicator]:
    """모든 핵심 경제 지표 수집"""
    try:
        logger.info("Collecting all core economic indicators")
        
        indicators = {}
        
        # 배치 처리 (API 제한 고려)
        semaphore = asyncio.Semaphore(5)  # 동시 요청 5개로 제한
        
        async def collect_single_indicator(series_id: str):
            async with semaphore:
                indicator = await self.get_economic_indicator(series_id)
                if indicator:
                    indicators[series_id] = indicator
                await asyncio.sleep(0.2)  # API 제한 준수
        
        # 모든 지표 병렬 수집
        tasks = [
            collect_single_indicator(series_id)
            for series_id in self.core_indicators.keys()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Collected {len(indicators)} core indicators")
        return indicators
        
    except Exception as e:
        logger.error(f"Error collecting core indicators: {e}")
        return {}

async def analyze_economic_cycle(self) -> EconomicCycle:
    """레이 달리오 방식 경제 사이클 분석"""
    try:
        logger.info("Analyzing economic cycle (Dalio style)")
        
        # 핵심 지표들 수집
        indicators = await self.collect_all_core_indicators()
        
        if len(indicators) < 5:
            logger.warning("Insufficient data for cycle analysis")
            return self._default_cycle_analysis()
        
        # 성장 추세 분석
        growth_score = self._analyze_growth_trend(indicators)
        inflation_score = self._analyze_inflation_trend(indicators)
        
        # 4가지 경제 환경 매트릭스
        if growth_score > 0 and inflation_score < 0:
            cycle_phase = "recovery"  # 성장↑, 인플레↓
        elif growth_score > 0 and inflation_score > 0:
            cycle_phase = "expansion"  # 성장↑, 인플레↑
        elif growth_score < 0 and inflation_score > 0:
            cycle_phase = "stagflation"  # 성장↓, 인플레↑  
        else:
            cycle_phase = "contraction"  # 성장↓, 인플레↓
        
        # 신뢰도 계산
        confidence_score = self._calculate_confidence(indicators, growth_score, inflation_score)
        
        # 주요 지표값 추출
        key_indicators = {
            'gdp_growth': indicators.get('GDPC1', EconomicIndicator('', '', 0, datetime.now(), '', '', '', datetime.now())).value,
            'cpi_inflation': indicators.get('CPIAUCNS', EconomicIndicator('', '', 0, datetime.now(), '', '', '', datetime.now())).value,
            'unemployment': indicators.get('UNRATE', EconomicIndicator('', '', 0, datetime.now(), '', '', '', datetime.now())).value,
            'fed_funds_rate': indicators.get('FEDFUNDS', EconomicIndicator('', '', 0, datetime.now(), '', '', '', datetime.now())).value,
            'yield_curve': indicators.get('T10Y2Y', EconomicIndicator('', '', 0, datetime.now(), '', '', '', datetime.now())).value
        }
        
        result = EconomicCycle(
            growth_trend="expanding" if growth_score > 0.3 else "contracting" if growth_score < -0.3 else "stable",
            inflation_trend="rising" if inflation_score > 0.3 else "falling" if inflation_score < -0.3 else "stable", 
            cycle_phase=cycle_phase,
            confidence_score=confidence_score,
            key_indicators=key_indicators
        )
        
        # 결과 캐시
        await self.redis_client.setex(
            'fred_economic_cycle',
            1800,  # 30분 TTL
            json.dumps({
                'growth_trend': result.growth_trend,
                'inflation_trend': result.inflation_trend,
                'cycle_phase': result.cycle_phase,
                'confidence_score': result.confidence_score,
                'key_indicators': result.key_indicators,
                'timestamp': datetime.now().isoformat()
            })
        )
        
        logger.info(f"Economic cycle analysis: {cycle_phase} (confidence: {confidence_score:.2f})")
        return result
        
    except Exception as e:
        logger.error(f"Error in economic cycle analysis: {e}")
        return self._default_cycle_analysis()

def _analyze_growth_trend(self, indicators: Dict[str, EconomicIndicator]) -> float:
    """성장 추세 분석 (-1 to 1)"""
    score = 0.0
    weight_sum = 0.0
    
    # GDP 성장률 (가장 중요)
    if 'GDPC1' in indicators:
        # 실제 구현에서는 YoY 성장률 계산 필요
        gdp_growth = 2.0  # 임시값 - 실제로는 계산 필요
        score += gdp_growth * 0.4
        weight_sum += 0.4
    
    # 고용 상황
    if 'UNRATE' in indicators:
        unemployment = indicators['UNRATE'].value
        # 실업률이 낮을수록 성장 긍정적
        unemployment_score = max(-1, min(1, (5.0 - unemployment) / 3.0))
        score += unemployment_score * 0.3
        weight_sum += 0.3
    
    # 소비자 심리
    if 'UMCSENT' in indicators:
        sentiment = indicators['UMCSENT'].value
        sentiment_score = max(-1, min(1, (sentiment - 85) / 15))
        score += sentiment_score * 0.2
        weight_sum += 0.2
    
    # 생산성
    if 'OPHNFB' in indicators:
        productivity_score = 0.1  # 임시값
        score += productivity_score * 0.1
        weight_sum += 0.1
    
    return score / weight_sum if weight_sum > 0 else 0.0

def _analyze_inflation_trend(self, indicators: Dict[str, EconomicIndicator]) -> float:
    """인플레이션 추세 분석 (-1 to 1)"""
    score = 0.0
    weight_sum = 0.0
    
    # 핵심 CPI
    if 'CPILFESL' in indicators:
        # 실제로는 YoY 변화율 계산 필요
        core_cpi_change = 3.0  # 임시값
        # 2% 타겟 기준
        cpi_score = max(-1, min(1, (core_cpi_change - 2.0) / 3.0))
        score += cpi_score * 0.4
        weight_sum += 0.4
    
    # PCE 
    if 'PCEPILFE' in indicators:
        pce_score = 0.0  # 임시값
        score += pce_score * 0.3
        weight_sum += 0.3
    
    # 임금 상승 압력 (실업률로 대체)
    if 'UNRATE' in indicators:
        unemployment = indicators['UNRATE'].value
        wage_pressure = max(-1, min(1, (4.0 - unemployment) / 2.0))
        score += wage_pressure * 0.3
        weight_sum += 0.3
    
    return score / weight_sum if weight_sum > 0 else 0.0

def _calculate_confidence(
    self, 
    indicators: Dict[str, EconomicIndicator], 
    growth_score: float, 
    inflation_score: float
) -> float:
    """분석 신뢰도 계산"""
    # 데이터 완성도
    data_completeness = len(indicators) / len(self.core_indicators)
    
    # 지표 간 일관성
    consistency_score = 1.0 - abs(growth_score) * abs(inflation_score) * 0.1
    
    # 데이터 신선도 (최근 업데이트 여부)
    freshness_scores = []
    for indicator in indicators.values():
        days_old = (datetime.now() - indicator.last_updated).days
        freshness = max(0, 1 - days_old / 90)  # 90일 기준
        freshness_scores.append(freshness)
    
    avg_freshness = np.mean(freshness_scores) if freshness_scores else 0.5
    
    # 최종 신뢰도
    confidence = (data_completeness * 0.4 + consistency_score * 0.3 + avg_freshness * 0.3)
    
    return max(0.0, min(1.0, confidence))

def _default_cycle_analysis(self) -> EconomicCycle:
    """기본 사이클 분석 (데이터 부족시)"""
    return EconomicCycle(
        growth_trend="unknown",
        inflation_trend="unknown", 
        cycle_phase="uncertain",
        confidence_score=0.0,
        key_indicators={}
    )

async def get_yield_curve_data(self) -> Dict[str, float]:
    """수익률 곡선 데이터 수집"""
    try:
        yield_series = {
            '1mo': 'DGS1MO',
            '3mo': 'DGS3MO', 
            '6mo': 'DGS6MO',
            '1y': 'DGS1',
            '2y': 'DGS2',
            '3y': 'DGS3',
            '5y': 'DGS5',
            '7y': 'DGS7',
            '10y': 'DGS10',
            '20y': 'DGS20',
            '30y': 'DGS30'
        }
        
        yield_curve = {}
        
        for period, series_id in yield_series.items():
            indicator = await self.get_economic_indicator(series_id, limit=1)
            if indicator:
                yield_curve[period] = indicator.value
        
        return yield_curve
        
    except Exception as e:
        logger.error(f"Error collecting yield curve data: {e}")
        return {}

async def get_international_indicators(self) -> Dict[str, EconomicIndicator]:
    """국제 경제 지표 수집"""
    try:
        international_data = {}
        
        for series_id in self.international_indicators.keys():
            indicator = await self.get_economic_indicator(series_id)
            if indicator:
                international_data[series_id] = indicator
            await asyncio.sleep(0.3)  # API 제한 준수
        
        return international_data
        
    except Exception as e:
        logger.error(f"Error collecting international indicators: {e}")
        return {}
```

# 비동기 함수들 (스케줄러/API용)

async def collect_economic_indicators() -> Dict[str, Any]:
“”“모든 경제 지표 수집”””
async with FREDCollector() as collector:
core_indicators = await collector.collect_all_core_indicators()
cycle_analysis = await collector.analyze_economic_cycle()
yield_curve = await collector.get_yield_curve_data()

```
    return {
        'core_indicators': {k: v.__dict__ for k, v in core_indicators.items()},
        'cycle_analysis': cycle_analysis.__dict__,
        'yield_curve': yield_curve,
        'timestamp': datetime.now().isoformat()
    }
```

async def get_dalio_economic_machine() -> EconomicCycle:
“”“달리오 경제 머신 분석”””
async with FREDCollector() as collector:
return await collector.analyze_economic_cycle()

# 동기 래퍼 함수들 (Celery 태스크용)

def sync_collect_economic_indicators():
“”“동기 방식 경제 지표 수집”””
return asyncio.run(collect_economic_indicators())

def sync_get_economic_cycle():
“”“동기 방식 경제 사이클 분석”””
return asyncio.run(get_dalio_economic_machine())

if **name** == “**main**”:
# 테스트 실행
async def main():
async with FREDCollector() as collector:
print(”=== FRED 경제 지표 수집 테스트 ===”)

```
        # 핵심 지표 몇 개만 테스트
        test_indicators = ['FEDFUNDS', 'UNRATE', 'CPIAUCNS']
        
        for series_id in test_indicators:
            indicator = await collector.get_economic_indicator(series_id)
            if indicator:
                print(f"{indicator.title}: {indicator.value} {indicator.units}")
        
        # 경제 사이클 분석
        cycle = await collector.analyze_economic_cycle()
        print(f"\n경제 사이클: {cycle.cycle_phase}")
        print(f"성장 추세: {cycle.growth_trend}")
        print(f"인플레이션 추세: {cycle.inflation_trend}")
        print(f"신뢰도: {cycle.confidence_score:.2f}")

asyncio.run(main())
```