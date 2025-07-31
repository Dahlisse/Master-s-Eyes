# backend/app/api/v1/stocks.py

“””
주식 관련 기본 REST API 엔드포인트

- 현재가, 호가창, 차트 데이터
- 재무정보, 종목 검색
- 매매동향, 체결강도
  “””

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio

from …data.collectors.kis_api import KISApiClient, KISConfig, MarketType
from …core.config import get_kis_config
from …core.logging import get_main_logger, get_performance_logger, log_execution_time
from …core.redis import get_cached_stock_price, cache_stock_price, get_cached_orderbook, cache_orderbook

logger = get_main_logger()
perf_logger = get_performance_logger()

router = APIRouter(prefix=”/stocks”, tags=[“stocks”])

# KIS API 클라이언트 의존성

async def get_kis_client() -> KISApiClient:
“”“KIS API 클라이언트 의존성”””
config = get_kis_config()
client = KISApiClient(config)
await client.initialize()
return client

@router.get(”/{ticker}/price”)
@log_execution_time()
async def get_stock_price(
ticker: str,
market: str = Query(“KOSPI”, description=“시장 구분 (KOSPI, KOSDAQ, KONEX)”),
use_cache: bool = Query(True, description=“캐시 사용 여부”),
client: KISApiClient = Depends(get_kis_client)
):
“””
현재가 조회

```
- **ticker**: 6자리 종목 코드 (예: 005930)
- **market**: 시장 구분 (KOSPI, KOSDAQ, KONEX)
- **use_cache**: 캐시 사용 여부 (기본: True)
"""
try:
    # 티커 유효성 검증
    if not _validate_ticker(ticker):
        raise HTTPException(status_code=400, detail="유효하지 않은 종목 코드입니다")
    
    # 시장 구분 변환
    market_type = _get_market_type(market)
    
    # 캐시 확인
    if use_cache:
        cached_data = await get_cached_stock_price(ticker)
        if cached_data:
            logger.info(f"캐시에서 주가 조회: {ticker}")
            return {
                "ticker": ticker,
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.now().isoformat()
            }
    
    # 실시간 데이터 조회
    async with client:
        price_data = await client.get_current_price(ticker, market_type)
        
    # 응답 데이터 구성
    response_data = {
        "ticker": price_data.ticker,
        "name": price_data.name,
        "current_price": price_data.current_price,
        "change": price_data.change,
        "change_rate": price_data.change_rate,
        "volume": price_data.volume,
        "trade_value": price_data.trade_value,
        "open_price": price_data.open_price,
        "high_price": price_data.high_price,
        "low_price": price_data.low_price,
        "prev_close": price_data.prev_close,
        "market_cap": price_data.market_cap,
        "timestamp": price_data.timestamp.isoformat()
    }
    
    # 캐시 저장
    if use_cache:
        await cache_stock_price(ticker, response_data, expire=30)
    
    return {
        "ticker": ticker,
        "data": response_data,
        "source": "api",
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"주가 조회 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"주가 조회 실패: {str(e)}")
```

@router.get(”/{ticker}/orderbook”)
@log_execution_time()
async def get_stock_orderbook(
ticker: str,
market: str = Query(“KOSPI”, description=“시장 구분”),
use_cache: bool = Query(True, description=“캐시 사용 여부”),
client: KISApiClient = Depends(get_kis_client)
):
“””
호가창 조회

```
- **ticker**: 6자리 종목 코드
- **market**: 시장 구분
- **use_cache**: 캐시 사용 여부
"""
try:
    if not _validate_ticker(ticker):
        raise HTTPException(status_code=400, detail="유효하지 않은 종목 코드입니다")
    
    market_type = _get_market_type(market)
    
    # 캐시 확인
    if use_cache:
        cached_data = await get_cached_orderbook(ticker)
        if cached_data:
            return {
                "ticker": ticker,
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.now().isoformat()
            }
    
    # 실시간 데이터 조회
    async with client:
        orderbook_data = await client.get_orderbook(ticker, market_type)
        
    response_data = {
        "ticker": orderbook_data.ticker,
        "ask_prices": orderbook_data.ask_prices,
        "ask_volumes": orderbook_data.ask_volumes,
        "bid_prices": orderbook_data.bid_prices,
        "bid_volumes": orderbook_data.bid_volumes,
        "total_ask_volume": orderbook_data.total_ask_volume,
        "total_bid_volume": orderbook_data.total_bid_volume,
        "spread": orderbook_data.ask_prices[0] - orderbook_data.bid_prices[0] if orderbook_data.ask_prices[0] > 0 and orderbook_data.bid_prices[0] > 0 else 0,
        "timestamp": orderbook_data.timestamp.isoformat()
    }
    
    # 캐시 저장 (짧은 만료 시간)
    if use_cache:
        await cache_orderbook(ticker, response_data, expire=10)
    
    return {
        "ticker": ticker,
        "data": response_data,
        "source": "api",
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"호가창 조회 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"호가창 조회 실패: {str(e)}")
```

@router.get(”/{ticker}/ohlcv”)
@log_execution_time()
async def get_stock_ohlcv(
ticker: str,
period: str = Query(“D”, description=“기간 (D:일봉, W:주봉, M:월봉)”),
count: int = Query(100, ge=1, le=1000, description=“조회할 봉 개수”),
market: str = Query(“KOSPI”, description=“시장 구분”),
client: KISApiClient = Depends(get_kis_client)
):
“””
OHLCV 차트 데이터 조회

```
- **ticker**: 6자리 종목 코드
- **period**: 봉 종류 (D:일봉, W:주봉, M:월봉)
- **count**: 조회할 봉 개수 (1~1000)
- **market**: 시장 구분
"""
try:
    if not _validate_ticker(ticker):
        raise HTTPException(status_code=400, detail="유효하지 않은 종목 코드입니다")
    
    if period not in ["D", "W", "M"]:
        raise HTTPException(status_code=400, detail="유효하지 않은 기간입니다 (D, W, M)")
    
    async with client:
        ohlcv_data = await client.get_ohlcv_data(ticker, period, count)
        
    return {
        "ticker": ticker,
        "period": period,
        "count": len(ohlcv_data),
        "data": ohlcv_data,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"OHLCV 조회 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"OHLCV 조회 실패: {str(e)}")
```

@router.get(”/{ticker}/financial”)
@log_execution_time()
async def get_stock_financial(
ticker: str,
market: str = Query(“KOSPI”, description=“시장 구분”),
client: KISApiClient = Depends(get_kis_client)
):
“””
재무정보 조회

```
- **ticker**: 6자리 종목 코드
- **market**: 시장 구분
"""
try:
    if not _validate_ticker(ticker):
        raise HTTPException(status_code=400, detail="유효하지 않은 종목 코드입니다")
    
    market_type = _get_market_type(market)
    
    async with client:
        financial_data = await client.get_financial_info(ticker, market_type)
        
    response_data = {
        "ticker": financial_data.ticker,
        "per": financial_data.per,
        "pbr": financial_data.pbr,
        "eps": financial_data.eps,
        "bps": financial_data.bps,
        "roe": financial_data.roe,
        "debt_ratio": financial_data.debt_ratio,
        "dividend_yield": financial_data.dividend_yield,
        "market_cap": financial_data.market_cap,
        "timestamp": financial_data.timestamp.isoformat()
    }
    
    return {
        "ticker": ticker,
        "data": response_data,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"재무정보 조회 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"재무정보 조회 실패: {str(e)}")
```

@router.get(”/search”)
@log_execution_time()
async def search_stocks(
keyword: str = Query(…, min_length=1, description=“검색 키워드”),
limit: int = Query(20, ge=1, le=100, description=“최대 결과 수”),
client: KISApiClient = Depends(get_kis_client)
):
“””
종목 검색

```
- **keyword**: 종목명 또는 종목 코드
- **limit**: 최대 결과 수 (1~100)
"""
try:
    async with client:
        search_results = await client.search_stocks(keyword)
        
    # 결과 제한
    limited_results = search_results[:limit]
    
    return {
        "keyword": keyword,
        "total_found": len(search_results),
        "returned_count": len(limited_results),
        "results": limited_results,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"종목 검색 실패 ({keyword}): {e}")
    raise HTTPException(status_code=500, detail=f"종목 검색 실패: {str(e)}")
```

@router.get(”/batch/prices”)
@log_execution_time()
async def get_batch_prices(
tickers: List[str] = Query(…, description=“종목 코드 목록”),
market: str = Query(“KOSPI”, description=“시장 구분”),
use_cache: bool = Query(True, description=“캐시 사용 여부”),
client: KISApiClient = Depends(get_kis_client)
):
“””
다수 종목 현재가 일괄 조회

```
- **tickers**: 종목 코드 목록 (최대 50개)
- **market**: 시장 구분
- **use_cache**: 캐시 사용 여부
"""
try:
    if len(tickers) > 50:
        raise HTTPException(status_code=400, detail="한 번에 최대 50개 종목까지 조회 가능합니다")
    
    # 티커 유효성 검증
    invalid_tickers = [t for t in tickers if not _validate_ticker(t)]
    if invalid_tickers:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 종목 코드: {invalid_tickers}")
    
    market_type = _get_market_type(market)
    
    results = {}
    cache_hits = 0
    api_calls = 0
    
    async with client:
        for ticker in tickers:
            try:
                # 캐시 확인
                if use_cache:
                    cached_data = await get_cached_stock_price(ticker)
                    if cached_data:
                        results[ticker] = {
                            "data": cached_data,
                            "source": "cache"
                        }
                        cache_hits += 1
                        continue
                
                # API 호출
                price_data = await client.get_current_price(ticker, market_type)
                
                response_data = {
                    "ticker": price_data.ticker,
                    "name": price_data.name,
                    "current_price": price_data.current_price,
                    "change": price_data.change,
                    "change_rate": price_data.change_rate,
                    "volume": price_data.volume,
                    "timestamp": price_data.timestamp.isoformat()
                }
                
                results[ticker] = {
                    "data": response_data,
                    "source": "api"
                }
                
                # 캐시 저장
                if use_cache:
                    await cache_stock_price(ticker, response_data, expire=30)
                
                api_calls += 1
                
                # API 호출 간격 (속도 제한 고려)
                if api_calls % 10 == 0:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"종목 {ticker} 조회 실패: {e}")
                results[ticker] = {
                    "error": str(e),
                    "source": "error"
                }
    
    return {
        "requested_tickers": tickers,
        "successful_count": len([r for r in results.values() if "data" in r]),
        "cache_hits": cache_hits,
        "api_calls": api_calls,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"일괄 주가 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"일괄 조회 실패: {str(e)}")
```

# 유틸리티 함수들

def _validate_ticker(ticker: str) -> bool:
“”“티커 유효성 검증”””
import re
if not isinstance(ticker, str):
return False
pattern = r’^\d{6}$’
return bool(re.match(pattern, ticker))

def _get_market_type(market: str) -> MarketType:
“”“시장 구분 문자열을 MarketType으로 변환”””
market_map = {
“KOSPI”: MarketType.KOSPI,
“KOSDAQ”: MarketType.KOSDAQ,
“KONEX”: MarketType.KONEX
}

```
if market.upper() not in market_map:
    raise HTTPException(status_code=400, detail=f"유효하지 않은 시장 구분: {market}")

return market_map[market.upper()]
```