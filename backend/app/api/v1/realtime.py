# backend/app/api/v1/realtime.py

“””
실시간 데이터 API 엔드포인트

- 실시간 수집 제어
- 종목 구독 관리
- 실시간 데이터 조회
- 성능 모니터링
  “””

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime

from …data.collectors.realtime_collector import get_realtime_collector, start_realtime_collection, stop_realtime_collection
from …core.logging import get_main_logger, get_performance_logger, get_business_logger
from …core.redis import get_cached_realtime_data, cache_realtime_data

logger = get_main_logger()
perf_logger = get_performance_logger()
business_logger = get_business_logger()

router = APIRouter(prefix=”/realtime”, tags=[“realtime”])

# WebSocket 연결 관리

active_connections: List[WebSocket] = []

@router.post(”/start”)
async def start_collection(background_tasks: BackgroundTasks):
“”“실시간 수집 시작”””
try:
logger.info(“실시간 데이터 수집 시작 요청”)
background_tasks.add_task(start_realtime_collection)

```
    business_logger.log_portfolio_creation(
        user_id=0,  # 시스템 사용자
        portfolio_id=0,
        strategy="realtime_collection",
        action="start"
    )
    
    return {
        "status": "started", 
        "message": "실시간 데이터 수집을 시작했습니다",
        "timestamp": datetime.now().isoformat()
    }
except Exception as e:
    logger.error(f"실시간 수집 시작 실패: {e}")
    raise HTTPException(status_code=500, detail=f"수집 시작 실패: {str(e)}")
```

@router.post(”/stop”)
async def stop_collection():
“”“실시간 수집 정지”””
try:
logger.info(“실시간 데이터 수집 정지 요청”)
await stop_realtime_collection()

```
    business_logger.log_portfolio_creation(
        user_id=0,
        portfolio_id=0,
        strategy="realtime_collection",
        action="stop"
    )
    
    return {
        "status": "stopped", 
        "message": "실시간 데이터 수집을 정지했습니다",
        "timestamp": datetime.now().isoformat()
    }
except Exception as e:
    logger.error(f"실시간 수집 정지 실패: {e}")
    raise HTTPException(status_code=500, detail=f"수집 정지 실패: {str(e)}")
```

@router.post(”/subscribe”)
async def subscribe_tickers(tickers: List[str]):
“”“종목 구독 추가”””
try:
logger.info(f”종목 구독 요청: {tickers}”)

```
    # 티커 유효성 검증
    invalid_tickers = [t for t in tickers if not _validate_ticker(t)]
    if invalid_tickers:
        raise HTTPException(
            status_code=400, 
            detail=f"유효하지 않은 티커: {invalid_tickers}"
        )
    
    collector = await get_realtime_collector()
    results = await collector.subscribe_multiple_tickers(tickers)
    
    # 성공/실패 분리
    successful = [ticker for ticker, success in results.items() if success]
    failed = [ticker for ticker, success in results.items() if not success]
    
    logger.info(f"구독 완료 - 성공: {successful}, 실패: {failed}")
    
    return {
        "status": "completed",
        "successful_subscriptions": successful,
        "failed_subscriptions": failed,
        "total_requested": len(tickers),
        "success_count": len(successful),
        "timestamp": datetime.now().isoformat()
    }
except Exception as e:
    logger.error(f"종목 구독 실패: {e}")
    raise HTTPException(status_code=500, detail=f"구독 실패: {str(e)}")
```

@router.delete(”/subscribe/{ticker}”)
async def unsubscribe_ticker(ticker: str):
“”“종목 구독 해제”””
try:
logger.info(f”종목 구독 해제 요청: {ticker}”)

```
    if not _validate_ticker(ticker):
        raise HTTPException(status_code=400, detail=f"유효하지 않은 티커: {ticker}")
    
    collector = await get_realtime_collector()
    success = await collector.unsubscribe_ticker(ticker)
    
    if success:
        logger.info(f"종목 구독 해제 완료: {ticker}")
        return {
            "status": "unsubscribed",
            "ticker": ticker,
            "timestamp": datetime.now().isoformat()
        }
    else:
        logger.warning(f"종목 구독 해제 실패: {ticker}")
        return {
            "status": "failed",
            "ticker": ticker,
            "message": "구독 해제에 실패했습니다",
            "timestamp": datetime.now().isoformat()
        }
except Exception as e:
    logger.error(f"종목 구독 해제 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"구독 해제 실패: {str(e)}")
```

@router.get(”/subscriptions”)
async def get_subscriptions():
“”“현재 구독 중인 종목 목록 조회”””
try:
collector = await get_realtime_collector()
stats = collector.get_stats()

```
    return {
        "subscribed_tickers": list(stats.get("connection_stats", {}).keys()),
        "total_subscriptions": stats.get("subscribed_tickers", 0),
        "active_connections": stats.get("active_connections", 0),
        "timestamp": datetime.now().isoformat()
    }
except Exception as e:
    logger.error(f"구독 목록 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"구독 목록 조회 실패: {str(e)}")
```

@router.get(”/stats”)
async def get_collection_stats():
“”“수집 통계 조회”””
try:
collector = await get_realtime_collector()
stats = collector.get_stats()

```
    # 성능 로깅
    perf_logger.log_data_processing(
        data_type="realtime_stats",
        record_count=stats.get("subscribed_tickers", 0),
        processing_time=0.001  # 통계 조회는 빠름
    )
    
    return {
        "collection_stats": stats,
        "timestamp": datetime.now().isoformat()
    }
except Exception as e:
    logger.error(f"통계 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")
```

@router.get(”/health”)
async def health_check():
“”“헬스 체크”””
try:
collector = await get_realtime_collector()
stats = collector.get_stats()

```
    is_healthy = (
        stats["is_running"] and 
        stats["active_connections"] > 0 and
        stats["queue_size"] < 8000  # 큐 크기가 80% 미만
    )
    
    status = "healthy" if is_healthy else "degraded" if stats["is_running"] else "stopped"
    
    return {
        "status": status,
        "is_running": stats["is_running"],
        "active_connections": stats["active_connections"],
        "subscribed_tickers": stats["subscribed_tickers"],
        "queue_size": stats["queue_size"],
        "queue_utilization": f"{(stats['queue_size'] / 10000) * 100:.1f}%",
        "timestamp": datetime.now().isoformat()
    }
except Exception as e:
    logger.error(f"헬스 체크 실패: {e}")
    return {
        "status": "unhealthy",
        "error": str(e),
        "timestamp": datetime.now().isoformat()
    }
```

@router.get(”/data/{ticker}/price”)
async def get_realtime_price(ticker: str):
“”“실시간 주가 데이터 조회”””
try:
if not _validate_ticker(ticker):
raise HTTPException(status_code=400, detail=f”유효하지 않은 티커: {ticker}”)

```
    # 캐시에서 먼저 조회
    cached_data = await get_cached_realtime_data(ticker, "price")
    
    if cached_data:
        return {
            "ticker": ticker,
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"실시간 주가 데이터 없음: {ticker}")
        
except HTTPException:
    raise
except Exception as e:
    logger.error(f"실시간 주가 조회 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"주가 조회 실패: {str(e)}")
```

@router.get(”/data/{ticker}/orderbook”)
async def get_realtime_orderbook(ticker: str):
“”“실시간 호가창 데이터 조회”””
try:
if not _validate_ticker(ticker):
raise HTTPException(status_code=400, detail=f”유효하지 않은 티커: {ticker}”)

```
    # 캐시에서 조회
    cached_data = await get_cached_realtime_data(ticker, "orderbook")
    
    if cached_data:
        return {
            "ticker": ticker,
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"실시간 호가 데이터 없음: {ticker}")
        
except HTTPException:
    raise
except Exception as e:
    logger.error(f"실시간 호가 조회 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"호가 조회 실패: {str(e)}")
```

@router.get(”/data/batch”)
async def get_batch_realtime_data(
tickers: List[str] = Query(…, description=“조회할 종목 코드 목록”),
data_types: List[str] = Query([“price”], description=“데이터 타입 (price, orderbook)”)
):
“”“다수 종목 실시간 데이터 일괄 조회”””
try:
# 티커 유효성 검증
invalid_tickers = [t for t in tickers if not _validate_ticker(t)]
if invalid_tickers:
raise HTTPException(
status_code=400,
detail=f”유효하지 않은 티커: {invalid_tickers}”
)

```
    # 데이터 타입 유효성 검증
    valid_types = {"price", "orderbook", "execution"}
    invalid_types = [t for t in data_types if t not in valid_types]
    if invalid_types:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 데이터 타입: {invalid_types}"
        )
    
    result = {}
    
    # 각 종목과 데이터 타입 조합으로 조회
    for ticker in tickers:
        result[ticker] = {}
        for data_type in data_types:
            cached_data = await get_cached_realtime_data(ticker, data_type)
            result[ticker][data_type] = cached_data
    
    return {
        "data": result,
        "requested_tickers": tickers,
        "requested_types": data_types,
        "timestamp": datetime.now().isoformat()
    }
    
except HTTPException:
    raise
except Exception as e:
    logger.error(f"일괄 실시간 데이터 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"일괄 조회 실패: {str(e)}")
```

# WebSocket 엔드포인트

@router.websocket(”/ws/{ticker}”)
async def websocket_realtime_data(websocket: WebSocket, ticker: str):
“”“실시간 데이터 WebSocket 스트림”””
if not _validate_ticker(ticker):
await websocket.close(code=1000, reason=“Invalid ticker”)
return

```
await websocket.accept()
active_connections.append(websocket)

logger.info(f"WebSocket 연결 시작: {ticker}")

try:
    # 실시간 데이터 스트림
    while True:
        # 캐시에서 최신 데이터 조회
        price_data = await get_cached_realtime_data(ticker, "price")
        orderbook_data = await get_cached_realtime_data(ticker, "orderbook")
        
        # 데이터가 있으면 전송
        if price_data or orderbook_data:
            message = {
                "ticker": ticker,
                "price": price_data,
                "orderbook": orderbook_data,
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_text(json.dumps(message, default=str))
        
        # 1초 대기
        await asyncio.sleep(1)
        
except WebSocketDisconnect:
    logger.info(f"WebSocket 연결 종료: {ticker}")
except Exception as e:
    logger.error(f"WebSocket 오류 ({ticker}): {e}")
finally:
    if websocket in active_connections:
        active_connections.remove(websocket)
```

@router.websocket(”/ws/multi”)
async def websocket_multi_ticker(websocket: WebSocket):
“”“다중 종목 실시간 데이터 WebSocket 스트림”””
await websocket.accept()
active_connections.append(websocket)

```
logger.info("다중 종목 WebSocket 연결 시작")

try:
    # 클라이언트로부터 구독할 종목 목록 수신
    data = await websocket.receive_text()
    subscription_request = json.loads(data)
    tickers = subscription_request.get("tickers", [])
    
    # 티커 유효성 검증
    valid_tickers = [t for t in tickers if _validate_ticker(t)]
    
    if not valid_tickers:
        await websocket.send_text(json.dumps({
            "error": "유효한 티커가 없습니다",
            "requested": tickers
        }))
        return
    
    logger.info(f"다중 종목 구독: {valid_tickers}")
    
    # 실시간 데이터 스트림
    while True:
        message = {
            "data": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # 각 종목의 최신 데이터 수집
        for ticker in valid_tickers:
            price_data = await get_cached_realtime_data(ticker, "price")
            orderbook_data = await get_cached_realtime_data(ticker, "orderbook")
            
            if price_data or orderbook_data:
                message["data"][ticker] = {
                    "price": price_data,
                    "orderbook": orderbook_data
                }
        
        # 데이터가 있으면 전송
        if message["data"]:
            await websocket.send_text(json.dumps(message, default=str))
        
        # 1초 대기
        await asyncio.sleep(1)
        
except WebSocketDisconnect:
    logger.info("다중 종목 WebSocket 연결 종료")
except Exception as e:
    logger.error(f"다중 종목 WebSocket 오류: {e}")
finally:
    if websocket in active_connections:
        active_connections.remove(websocket)
```

# 유틸리티 함수들

def _validate_ticker(ticker: str) -> bool:
“”“티커 유효성 검증”””
import re

```
if not isinstance(ticker, str):
    return False

# 한국 주식 티커는 6자리 숫자
pattern = r'^\d{6}$'
return bool(re.match(pattern, ticker))
```

@router.get(”/validate/{ticker}”)
async def validate_ticker(ticker: str):
“”“티커 유효성 검증”””
is_valid = _validate_ticker(ticker)

```
return {
    "ticker": ticker,
    "is_valid": is_valid,
    "format": "6자리 숫자" if not is_valid else "유효함",
    "timestamp": datetime.now().isoformat()
}
```

# 시스템 관리 엔드포인트

@router.post(”/system/cleanup”)
async def cleanup_system():
“”“시스템 정리 (캐시 클리어 등)”””
try:
# Redis 캐시 정리
from …core.redis import cleanup_expired_keys
await cleanup_expired_keys(“realtime:*”)

```
    logger.info("시스템 정리 완료")
    
    return {
        "status": "cleaned",
        "message": "시스템 정리가 완료되었습니다",
        "timestamp": datetime.now().isoformat()
    }
except Exception as e:
    logger.error(f"시스템 정리 실패: {e}")
    raise HTTPException(status_code=500, detail=f"시스템 정리 실패: {str(e)}")
```

@router.get(”/system/connections”)
async def get_websocket_connections():
“”“활성 WebSocket 연결 정보”””
return {
“active_websocket_connections”: len(active_connections),
“connection_details”: [
{
“client_host”: getattr(conn.client, ‘host’, ‘unknown’),
“client_port”: getattr(conn.client, ‘port’, ‘unknown’)
}
for conn in active_connections
],
“timestamp”: datetime.now().isoformat()
}