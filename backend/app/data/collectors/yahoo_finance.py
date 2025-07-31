# app/data/collectors/yahoo_finance.py

“””
Yahoo Finance API 연동 모듈
글로벌 시장 데이터, 환율, 원자재 등 수집
“””
import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import json

from app.core.config import get_settings
from app.core.redis import get_redis_client
from app.data.processors.data_validator import DataValidator

settings = get_settings()
logger = logging.getLogger(**name**)

@dataclass
class GlobalMarketData:
“”“글로벌 시장 데이터 구조”””
symbol: str
name: str
price: float
change: float
change_percent: float
volume: int
market_cap: Optional[float]
pe_ratio: Optional[float]
dividend_yield: Optional[float]
timestamp: datetime

@dataclass
class EconomicIndicator:
“”“경제 지표 데이터 구조”””
indicator: str
value: float
date: datetime
source: str

class YahooFinanceCollector:
“”“Yahoo Finance 데이터 수집기”””

```
def __init__(self):
    self.redis_client = get_redis_client()
    self.validator = DataValidator()
    self.session = None
    
    # 수집할 주요 글로벌 자산들
    self.global_assets = {
        # 주요 지수
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^N225': 'Nikkei 225',
        '^FTSE': 'FTSE 100',
        '^HSI': 'Hang Seng',
        
        # 환율
        'USDKRW=X': 'USD/KRW',
        'EURUSD=X': 'EUR/USD',
        'GBPUSD=X': 'GBP/USD',
        'USDJPY=X': 'USD/JPY',
        'USDCNY=X': 'USD/CNY',
        
        # 원자재
        'CL=F': 'WTI Crude Oil',
        'GC=F': 'Gold',
        'SI=F': 'Silver',
        'HG=F': 'Copper',
        'NG=F': 'Natural Gas',
        
        # 채권
        '^TNX': 'US 10Y Treasury',
        '^TYX': 'US 30Y Treasury',
        '^FVX': 'US 5Y Treasury',
        
        # VIX
        '^VIX': 'VIX Fear Index',
        
        # 달러 인덱스
        'DX-Y.NYB': 'US Dollar Index',
        
        # 주요 ETF
        'SPY': 'SPDR S&P 500 ETF',
        'QQQ': 'Invesco QQQ Trust',
        'IWM': 'iShares Russell 2000 ETF',
        'EFA': 'iShares MSCI EAFE ETF',
        'EEM': 'iShares MSCI Emerging Markets ETF'
    }
    
    # 한국 관련 ETF
    self.korea_etfs = {
        'EWY': 'iShares MSCI South Korea ETF',
        'FLKR': 'Franklin FTSE South Korea ETF',
        'FKO': 'First Trust South Korea AlphaDEX Fund'
    }
    
    # 반도체 관련 ETF/주식 (삼성전자와 연관성)
    self.semiconductor_assets = {
        'SOXX': 'iShares Semiconductor ETF',
        'SMH': 'VanEck Semiconductor ETF',
        'NVDA': 'NVIDIA',
        'TSM': 'Taiwan Semiconductor',
        'INTC': 'Intel',
        'AMD': 'Advanced Micro Devices',
        'QCOM': 'Qualcomm',
        'AVGO': 'Broadcom'
    }

async def __aenter__(self):
    """비동기 컨텍스트 매니저 진입"""
    self.session = aiohttp.ClientSession()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    """비동기 컨텍스트 매니저 종료"""
    if self.session:
        await self.session.close()

async def collect_realtime_data(self, symbols: List[str]) -> Dict[str, GlobalMarketData]:
    """실시간 글로벌 시장 데이터 수집"""
    try:
        logger.info(f"Collecting realtime data for {len(symbols)} symbols")
        
        # yfinance를 사용한 실시간 데이터 수집
        data = {}
        
        # 배치로 처리 (API 제한 고려)
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            
            try:
                # yfinance Ticker 객체들 생성
                tickers = yf.Tickers(' '.join(batch_symbols))
                
                for symbol in batch_symbols:
                    try:
                        ticker = tickers.tickers[symbol]
                        info = ticker.info
                        hist = ticker.history(period='1d', interval='1m')
                        
                        if not hist.empty:
                            latest = hist.iloc[-1]
                            
                            market_data = GlobalMarketData(
                                symbol=symbol,
                                name=self._get_asset_name(symbol),
                                price=float(latest['Close']),
                                change=float(latest['Close'] - hist.iloc[0]['Open']),
                                change_percent=float((latest['Close'] - hist.iloc[0]['Open']) / hist.iloc[0]['Open'] * 100),
                                volume=int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
                                market_cap=info.get('marketCap'),
                                pe_ratio=info.get('trailingPE'),
                                dividend_yield=info.get('dividendYield'),
                                timestamp=datetime.now()
                            )
                            
                            # 데이터 검증
                            if self.validator.validate_price_data(market_data.price):
                                data[symbol] = market_data
                                
                                # Redis에 캐시 (5분 TTL)
                                await self._cache_market_data(symbol, market_data)
                                
                    except Exception as e:
                        logger.error(f"Error collecting data for {symbol}: {e}")
                        continue
                
                # API 제한 방지를 위한 지연
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error collecting batch {batch_symbols}: {e}")
                continue
        
        logger.info(f"Successfully collected data for {len(data)} symbols")
        return data
        
    except Exception as e:
        logger.error(f"Error in collect_realtime_data: {e}")
        return {}

async def collect_historical_data(
    self, 
    symbol: str, 
    period: str = '1y',
    interval: str = '1d'
) -> pd.DataFrame:
    """과거 데이터 수집"""
    try:
        logger.info(f"Collecting historical data for {symbol}, period: {period}")
        
        # 캐시 확인
        cache_key = f"yahoo_hist:{symbol}:{period}:{interval}"
        cached_data = await self.redis_client.get(cache_key)
        
        if cached_data:
            return pd.read_json(cached_data)
        
        # yfinance로 과거 데이터 수집
        ticker = yf.Ticker(symbol)
        hist_data = ticker.history(period=period, interval=interval)
        
        if not hist_data.empty:
            # 데이터 정제
            hist_data.reset_index(inplace=True)
            hist_data['Symbol'] = symbol
            
            # Redis에 캐시 (1시간 TTL)
            await self.redis_client.setex(
                cache_key, 
                3600, 
                hist_data.to_json()
            )
            
            logger.info(f"Collected {len(hist_data)} historical records for {symbol}")
            return hist_data
        else:
            logger.warning(f"No historical data found for {symbol}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error collecting historical data for {symbol}: {e}")
        return pd.DataFrame()

async def collect_all_global_data(self) -> Dict[str, Any]:
    """모든 글로벌 데이터 수집"""
    try:
        logger.info("Starting comprehensive global data collection")
        
        all_symbols = list(self.global_assets.keys()) + \
                     list(self.korea_etfs.keys()) + \
                     list(self.semiconductor_assets.keys())
        
        # 실시간 데이터 수집
        realtime_data = await self.collect_realtime_data(all_symbols)
        
        # 카테고리별로 분류
        categorized_data = {
            'indices': {},
            'currencies': {},
            'commodities': {},
            'bonds': {},
            'volatility': {},
            'korea_etfs': {},
            'semiconductors': {}
        }
        
        for symbol, data in realtime_data.items():
            category = self._categorize_symbol(symbol)
            if category:
                categorized_data[category][symbol] = data
        
        # 요약 통계 계산
        summary = self._calculate_market_summary(categorized_data)
        
        result = {
            'data': categorized_data,
            'summary': summary,
            'timestamp': datetime.now(),
            'total_symbols': len(realtime_data)
        }
        
        # 전체 데이터 캐시
        await self.redis_client.setex(
            'yahoo_global_data', 
            300,  # 5분 TTL
            json.dumps(result, default=str)
        )
        
        logger.info(f"Global data collection completed: {len(realtime_data)} symbols")
        return result
        
    except Exception as e:
        logger.error(f"Error in collect_all_global_data: {e}")
        return {}

async def get_sector_performance(self) -> Dict[str, float]:
    """섹터별 성과 데이터 수집"""
    try:
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLC': 'Communication Services',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate'
        }
        
        sector_data = await self.collect_realtime_data(list(sector_etfs.keys()))
        
        performance = {}
        for symbol, data in sector_data.items():
            sector_name = sector_etfs.get(symbol, symbol)
            performance[sector_name] = data.change_percent
        
        return performance
        
    except Exception as e:
        logger.error(f"Error collecting sector performance: {e}")
        return {}

async def get_fear_greed_indicators(self) -> Dict[str, float]:
    """공포/탐욕 지수 관련 지표 수집"""
    try:
        fear_greed_symbols = ['^VIX', '^TNX', 'DX-Y.NYB', 'GC=F']
        
        data = await self.collect_realtime_data(fear_greed_symbols)
        
        indicators = {}
        for symbol, market_data in data.items():
            if symbol == '^VIX':
                indicators['vix'] = market_data.price
            elif symbol == '^TNX':
                indicators['us_10y_yield'] = market_data.price
            elif symbol == 'DX-Y.NYB':
                indicators['dollar_index'] = market_data.price
            elif symbol == 'GC=F':
                indicators['gold_price'] = market_data.price
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error collecting fear/greed indicators: {e}")
        return {}

def _get_asset_name(self, symbol: str) -> str:
    """심볼에서 자산명 가져오기"""
    all_assets = {**self.global_assets, **self.korea_etfs, **self.semiconductor_assets}
    return all_assets.get(symbol, symbol)

def _categorize_symbol(self, symbol: str) -> Optional[str]:
    """심볼을 카테고리별로 분류"""
    if symbol in ['^GSPC', '^DJI', '^IXIC', '^N225', '^FTSE', '^HSI']:
        return 'indices'
    elif '=X' in symbol:
        return 'currencies'
    elif symbol in ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F']:
        return 'commodities'
    elif symbol in ['^TNX', '^TYX', '^FVX']:
        return 'bonds'
    elif symbol == '^VIX':
        return 'volatility'
    elif symbol in self.korea_etfs:
        return 'korea_etfs'
    elif symbol in self.semiconductor_assets:
        return 'semiconductors'
    return None

def _calculate_market_summary(self, categorized_data: Dict) -> Dict[str, Any]:
    """시장 요약 통계 계산"""
    summary = {}
    
    for category, data in categorized_data.items():
        if data:
            changes = [item.change_percent for item in data.values()]
            summary[category] = {
                'avg_change': sum(changes) / len(changes),
                'positive_count': sum(1 for x in changes if x > 0),
                'negative_count': sum(1 for x in changes if x < 0),
                'total_count': len(changes)
            }
    
    return summary

async def _cache_market_data(self, symbol: str, data: GlobalMarketData):
    """시장 데이터 Redis 캐시"""
    try:
        cache_key = f"yahoo_realtime:{symbol}"
        cache_data = {
            'symbol': data.symbol,
            'name': data.name,
            'price': data.price,
            'change': data.change,
            'change_percent': data.change_percent,
            'volume': data.volume,
            'timestamp': data.timestamp.isoformat()
        }
        
        await self.redis_client.setex(
            cache_key, 
            300,  # 5분 TTL
            json.dumps(cache_data)
        )
        
    except Exception as e:
        logger.error(f"Error caching data for {symbol}: {e}")

async def get_cached_data(self, symbol: str) -> Optional[GlobalMarketData]:
    """캐시된 데이터 조회"""
    try:
        cache_key = f"yahoo_realtime:{symbol}"
        cached = await self.redis_client.get(cache_key)
        
        if cached:
            data = json.loads(cached)
            return GlobalMarketData(
                symbol=data['symbol'],
                name=data['name'],
                price=data['price'],
                change=data['change'],
                change_percent=data['change_percent'],
                volume=data['volume'],
                market_cap=None,
                pe_ratio=None,
                dividend_yield=None,
                timestamp=datetime.fromisoformat(data['timestamp'])
            )
        return None
        
    except Exception as e:
        logger.error(f"Error getting cached data for {symbol}: {e}")
        return None
```

# 비동기 함수들

async def collect_global_market_data() -> Dict[str, Any]:
“”“글로벌 시장 데이터 수집 (스케줄러용)”””
async with YahooFinanceCollector() as collector:
return await collector.collect_all_global_data()

async def collect_sector_rotation_data() -> Dict[str, float]:
“”“섹터 로테이션 데이터 수집”””
async with YahooFinanceCollector() as collector:
return await collector.get_sector_performance()

async def collect_market_sentiment_data() -> Dict[str, float]:
“”“시장 심리 지표 수집”””
async with YahooFinanceCollector() as collector:
return await collector.get_fear_greed_indicators()

# 동기 래퍼 함수들 (Celery 태스크용)

def sync_collect_global_data():
“”“동기 방식 글로벌 데이터 수집”””
return asyncio.run(collect_global_market_data())

def sync_collect_sector_data():
“”“동기 방식 섹터 데이터 수집”””
return asyncio.run(collect_sector_rotation_data())

def sync_collect_sentiment_data():
“”“동기 방식 시장 심리 데이터 수집”””
return asyncio.run(collect_market_sentiment_data())

if **name** == “**main**”:
# 테스트 실행 예제
async def main():
async with YahooFinanceCollector() as collector:
# 주요 지수만 테스트
test_symbols = [’^GSPC’, ‘USDKRW=X’, ‘^VIX’]
data = await collector.collect_realtime_data(test_symbols)

```
        print("=== Yahoo Finance 데이터 수집 테스트 ===")
        for symbol, market_data in data.items():
            print(f"{market_data.name} ({symbol}): "
                  f"${market_data.price:.2f} "
                  f"({market_data.change_percent:+.2f}%)")

asyncio.run(main())
```