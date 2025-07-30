# backend/app/core/config.py (실시간 설정 추가 부분)

“””
실시간 데이터 수집 설정
“””

import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class RealtimeConfig:
“”“실시간 데이터 수집 설정”””
max_concurrent_connections: int = int(os.getenv(“REALTIME_MAX_CONNECTIONS”, “5”))
batch_size: int = int(os.getenv(“REALTIME_BATCH_SIZE”, “100”))
flush_interval: int = int(os.getenv(“REALTIME_FLUSH_INTERVAL”, “5”))
cache_expire: int = int(os.getenv(“REALTIME_CACHE_EXPIRE”, “60”))
max_queue_size: int = int(os.getenv(“REALTIME_MAX_QUEUE_SIZE”, “10000”))
enable_monitoring: bool = os.getenv(“REALTIME_ENABLE_MONITORING”, “true”).lower() == “true”

def get_realtime_config() -> RealtimeConfig:
“”“실시간 설정 반환”””
return RealtimeConfig()

# backend/requirements.txt에 추가할 의존성

“””
aioredis==2.0.1
asyncpg==0.28.0
sqlalchemy[asyncio]==2.0.20
alembic==1.12.0
“””

# backend/app/core/redis.py

“””
Redis 연결 설정
“””

import os
import aioredis
from typing import Optional

_redis_client: Optional[aioredis.Redis] = None

def get_redis_url() -> str:
“”“Redis URL 반환”””
return os.getenv(
“REDIS_URL”,
“redis://localhost:6379/0”
)

async def get_redis_client() -> aioredis.Redis:
“”“Redis 클라이언트 반환”””
global _redis_client

```
if _redis_client is None:
    _redis_client = await aioredis.from_url(get_redis_url())
    
return _redis_client
```

async def close_redis_client():
“”“Redis 클라이언트 종료”””
global _redis_client

```
if _redis_client:
    await _redis_client.close()
    _redis_client = None
```

# backend/app/core/logging.py

“””
로깅 설정
“””

import logging
import sys
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
“”“로거 설정”””
logger = logging.getLogger(name)

```
if not logger.handlers:
    # 핸들러 생성
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # 로거에 핸들러 추가
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    
return logger
```

# backend/app/api/v1/realtime.py

“””
실시간 데이터 API 엔드포인트
“””

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import asyncio

from …data.collectors.realtime_collector import get_realtime_collector, start_realtime_collection, stop_realtime_collection

router = APIRouter(prefix=”/realtime”, tags=[“realtime”])

@router.post(”/start”)
async def start_collection(background_tasks: BackgroundTasks):
“”“실시간 수집 시작”””
try:
background_tasks.add_task(start_realtime_collection)
return {“status”: “started”, “message”: “실시간 데이터 수집을 시작했습니다”}
except Exception as e:
raise HTTPException(status_code=500, detail=f”수집 시작 실패: {str(e)}”)

@router.post(”/stop”)
async def stop_collection():
“”“실시간 수집 정지”””
try:
await stop_realtime_collection()
return {“status”: “stopped”, “message”: “실시간 데이터 수집을 정지했습니다”}
except Exception as e:
raise HTTPException(status_code=500, detail=f”수집 정지 실패: {str(e)}”)

@router.post(”/subscribe”)
async def subscribe_tickers(tickers: List[str]):
“”“종목 구독”””
try:
collector = await get_realtime_collector()
results = await collector.subscribe_multiple_tickers(tickers)
return {“results”: results}
except Exception as e:
raise HTTPException(status_code=500, detail=f”구독 실패: {str(e)}”)

@router.delete(”/subscribe/{ticker}”)
async def unsubscribe_ticker(ticker: str):
“”“종목 구독 해제”””
try:
collector = await get_realtime_collector()
success = await collector.unsubscribe_ticker(ticker)
return {“success”: success, “ticker”: ticker}
except Exception as e:
raise HTTPException(status_code=500, detail=f”구독 해제 실패: {str(e)}”)

@router.get(”/stats”)
async def get_collection_stats():
“”“수집 통계 조회”””
try:
collector = await get_realtime_collector()
stats = collector.get_stats()
return stats
except Exception as e:
raise HTTPException(status_code=500, detail=f”통계 조회 실패: {str(e)}”)

@router.get(”/health”)
async def health_check():
“”“헬스 체크”””
try:
collector = await get_realtime_collector()
stats = collector.get_stats()

```
    return {
        "status": "healthy" if stats["is_running"] else "stopped",
        "active_connections": stats["active_connections"],
        "subscribed_tickers": stats["subscribed_tickers"],
        "queue_size": stats["queue_size"]
    }
except Exception as e:
    return {
        "status": "unhealthy",
        "error": str(e)
    }
```