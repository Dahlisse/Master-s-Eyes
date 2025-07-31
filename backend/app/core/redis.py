# backend/app/core/redis.py

“””
Redis 연결 관리

- 비동기 Redis 클라이언트
- 연결 풀 관리
- 캐시 유틸리티 함수
  “””

import os
import json
import logging
from typing import Optional, Any, Dict, List, Union
import aioredis
from datetime import datetime, timedelta

logger = logging.getLogger(**name**)

class RedisManager:
“”“Redis 연결 관리자”””

```
def __init__(self):
    self.client: Optional[aioredis.Redis] = None
    self.is_connected = False
    
async def connect(self):
    """Redis 연결"""
    try:
        redis_url = get_redis_url()
        self.client = await aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
            retry_on_timeout=True
        )
        
        # 연결 테스트
        await self.client.ping()
        self.is_connected = True
        logger.info(f"Redis 연결 성공: {redis_url}")
        
    except Exception as e:
        logger.error(f"Redis 연결 실패: {e}")
        self.is_connected = False
        raise
        
async def disconnect(self):
    """Redis 연결 해제"""
    if self.client:
        await self.client.close()
        self.is_connected = False
        logger.info("Redis 연결 해제")
        
async def get(self, key: str) -> Optional[str]:
    """값 조회"""
    try:
        if not self.is_connected:
            await self.connect()
        return await self.client.get(key)
    except Exception as e:
        logger.error(f"Redis GET 오류 ({key}): {e}")
        return None
        
async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
    """값 저장"""
    try:
        if not self.is_connected:
            await self.connect()
            
        if expire:
            await self.client.setex(key, expire, value)
        else:
            await self.client.set(key, value)
        return True
    except Exception as e:
        logger.error(f"Redis SET 오류 ({key}): {e}")
        return False
        
async def delete(self, key: str) -> bool:
    """키 삭제"""
    try:
        if not self.is_connected:
            await self.connect()
        result = await self.client.delete(key)
        return result > 0
    except Exception as e:
        logger.error(f"Redis DELETE 오류 ({key}): {e}")
        return False
        
async def exists(self, key: str) -> bool:
    """키 존재 확인"""
    try:
        if not self.is_connected:
            await self.connect()
        return await self.client.exists(key) > 0
    except Exception as e:
        logger.error(f"Redis EXISTS 오류 ({key}): {e}")
        return False
        
async def expire(self, key: str, seconds: int) -> bool:
    """만료 시간 설정"""
    try:
        if not self.is_connected:
            await self.connect()
        return await self.client.expire(key, seconds)
    except Exception as e:
        logger.error(f"Redis EXPIRE 오류 ({key}): {e}")
        return False
        
async def keys(self, pattern: str) -> List[str]:
    """패턴으로 키 검색"""
    try:
        if not self.is_connected:
            await self.connect()
        return await self.client.keys(pattern)
    except Exception as e:
        logger.error(f"Redis KEYS 오류 ({pattern}): {e}")
        return []
        
async def hget(self, name: str, key: str) -> Optional[str]:
    """해시 필드 조회"""
    try:
        if not self.is_connected:
            await self.connect()
        return await self.client.hget(name, key)
    except Exception as e:
        logger.error(f"Redis HGET 오류 ({name}, {key}): {e}")
        return None
        
async def hset(self, name: str, key: str, value: str) -> bool:
    """해시 필드 저장"""
    try:
        if not self.is_connected:
            await self.connect()
        result = await self.client.hset(name, key, value)
        return result >= 0
    except Exception as e:
        logger.error(f"Redis HSET 오류 ({name}, {key}): {e}")
        return False
        
async def hgetall(self, name: str) -> Dict[str, str]:
    """해시 전체 조회"""
    try:
        if not self.is_connected:
            await self.connect()
        return await self.client.hgetall(name)
    except Exception as e:
        logger.error(f"Redis HGETALL 오류 ({name}): {e}")
        return {}
```

# 전역 Redis 관리자 인스턴스

_redis_manager: Optional[RedisManager] = None

def get_redis_url() -> str:
“”“Redis URL 반환”””
return os.getenv(
“REDIS_URL”,
“redis://localhost:6379/0”
)

async def get_redis_client() -> RedisManager:
“”“Redis 클라이언트 반환”””
global _redis_manager

```
if _redis_manager is None:
    _redis_manager = RedisManager()
    await _redis_manager.connect()
    
return _redis_manager
```

async def close_redis_client():
“”“Redis 클라이언트 종료”””
global _redis_manager

```
if _redis_manager:
    await _redis_manager.disconnect()
    _redis_manager = None
```

# 캐시 유틸리티 함수들

async def cache_stock_price(ticker: str, price_data: Dict[str, Any], expire: int = 60):
“”“주가 데이터 캐시”””
try:
redis = await get_redis_client()
key = f”price:{ticker}”
value = json.dumps(price_data, default=str)
await redis.set(key, value, expire)
except Exception as e:
logger.error(f”주가 캐시 저장 오류 ({ticker}): {e}”)

async def get_cached_stock_price(ticker: str) -> Optional[Dict[str, Any]]:
“”“캐시된 주가 데이터 조회”””
try:
redis = await get_redis_client()
key = f”price:{ticker}”
cached_data = await redis.get(key)

```
    if cached_data:
        return json.loads(cached_data)
    return None
except Exception as e:
    logger.error(f"주가 캐시 조회 오류 ({ticker}): {e}")
    return None
```

async def cache_orderbook(ticker: str, orderbook_data: Dict[str, Any], expire: int = 30):
“”“호가창 데이터 캐시”””
try:
redis = await get_redis_client()
key = f”orderbook:{ticker}”
value = json.dumps(orderbook_data, default=str)
await redis.set(key, value, expire)
except Exception as e:
logger.error(f”호가창 캐시 저장 오류 ({ticker}): {e}”)

async def get_cached_orderbook(ticker: str) -> Optional[Dict[str, Any]]:
“”“캐시된 호가창 데이터 조회”””
try:
redis = await get_redis_client()
key = f”orderbook:{ticker}”
cached_data = await redis.get(key)

```
    if cached_data:
        return json.loads(cached_data)
    return None
except Exception as e:
    logger.error(f"호가창 캐시 조회 오류 ({ticker}): {e}")
    return None
```

async def cache_realtime_data(ticker: str, data_type: str, data: Dict[str, Any], expire: int = 60):
“”“실시간 데이터 캐시”””
try:
redis = await get_redis_client()
key = f”realtime:{ticker}:{data_type}”
value = json.dumps(data, default=str)
await redis.set(key, value, expire)
except Exception as e:
logger.error(f”실시간 데이터 캐시 저장 오류 ({ticker}, {data_type}): {e}”)

async def get_cached_realtime_data(ticker: str, data_type: str) -> Optional[Dict[str, Any]]:
“”“캐시된 실시간 데이터 조회”””
try:
redis = await get_redis_client()
key = f”realtime:{ticker}:{data_type}”
cached_data = await redis.get(key)

```
    if cached_data:
        return json.loads(cached_data)
    return None
except Exception as e:
    logger.error(f"실시간 데이터 캐시 조회 오류 ({ticker}, {data_type}): {e}")
    return None
```

async def set_rate_limit(key: str, limit: int, window: int) -> bool:
“”“속도 제한 설정”””
try:
redis = await get_redis_client()
rate_key = f”rate_limit:{key}”

```
    # 현재 카운트 조회
    current = await redis.get(rate_key)
    if current is None:
        # 처음 요청
        await redis.set(rate_key, "1", window)
        return True
    elif int(current) < limit:
        # 제한 내
        await redis.incr(rate_key)
        return True
    else:
        # 제한 초과
        return False
except Exception as e:
    logger.error(f"속도 제한 설정 오류 ({key}): {e}")
    return True  # 오류 시 허용
```

async def cleanup_expired_keys(pattern: str):
“”“만료된 키 정리”””
try:
redis = await get_redis_client()
keys = await redis.keys(pattern)

```
    for key in keys:
        ttl = await redis.client.ttl(key)
        if ttl == -1:  # 만료 시간이 없는 키
            await redis.expire(key, 3600)  # 1시간 만료 시간 설정
except Exception as e:
    logger.error(f"키 정리 오류 ({pattern}): {e}")
```

# 컨텍스트 매니저

class RedisContext:
“”“Redis 컨텍스트 매니저”””

```
def __init__(self):
    self.redis = None
    
async def __aenter__(self) -> RedisManager:
    self.redis = await get_redis_client()
    return self.redis
    
async def __aexit__(self, exc_type, exc_val, exc_tb):
    # 연결은 전역 관리자가 관리하므로 여기서는 종료하지 않음
    pass
```