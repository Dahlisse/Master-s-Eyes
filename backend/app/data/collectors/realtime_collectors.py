# backend/app/data/collectors/realtime_collector.py

“””
WebSocket 실시간 데이터 수집 시스템

- 다수 종목 동시 실시간 수집
- Redis 캐싱 및 데이터베이스 저장
- 자동 재연결 및 에러 복구
- 성능 모니터링 및 백프레셔 제어
  “””

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import aioredis
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from .kis_api import KISApiClient, KISConfig, StockPrice, OrderBook
from ..processors.data_validator import RealtimeDataValidator
from …core.database import get_database_url
from …core.redis import get_redis_url
from …core.logging import setup_logger

logger = setup_logger(**name**)

@dataclass
class RealtimeConfig:
“”“실시간 데이터 수집 설정”””
max_concurrent_connections: int = 5  # 동시 WebSocket 연결 수
batch_size: int = 100               # 배치 처리 크기
flush_interval: int = 5             # DB 저장 주기 (초)
cache_expire: int = 60              # Redis 캐시 만료 시간 (초)
max_queue_size: int = 10000         # 최대 큐 크기
enable_monitoring: bool = True       # 성능 모니터링 활성화

@dataclass
class RealtimeDataPoint:
“”“실시간 데이터 포인트”””
ticker: str
data_type: str  # ‘price’, ‘orderbook’, ‘execution’
data: Dict[str, Any]
timestamp: datetime
received_at: datetime

@dataclass
class ConnectionStats:
“”“연결 통계”””
ticker: str
connected_at: datetime
last_message_at: Optional[datetime] = None
message_count: int = 0
error_count: int = 0
reconnect_count: int = 0

class RealtimeDataCollector:
“”“실시간 데이터 수집기”””

```
def __init__(self, kis_config: KISConfig, realtime_config: RealtimeConfig):
    self.kis_config = kis_config
    self.config = realtime_config
    
    # 연결 관리
    self.active_connections: Dict[str, KISApiClient] = {}
    self.connection_stats: Dict[str, ConnectionStats] = {}
    self.subscribed_tickers: Set[str] = set()
    
    # 데이터 큐 및 배치 처리
    self.data_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
    self.batch_buffer: List[RealtimeDataPoint] = []
    self.last_flush_time = time.time()
    
    # 외부 연결
    self.redis: Optional[aioredis.Redis] = None
    self.db_engine = None
    self.db_session = None
    
    # 태스크 관리
    self.running_tasks: Set[asyncio.Task] = set()
    self.is_running = False
    
    # 성능 모니터링
    self.performance_stats = {
        'messages_per_second': deque(maxlen=60),  # 1분간 초당 메시지 수
        'queue_size_history': deque(maxlen=60),   # 큐 크기 히스토리
        'error_rate': deque(maxlen=60),           # 에러율
        'last_stats_time': time.time()
    }
    
    # 데이터 검증기
    self.validator = RealtimeDataValidator()
    
async def initialize(self):
    """수집기 초기화"""
    try:
        # Redis 연결
        self.redis = await aioredis.from_url(get_redis_url())
        logger.info("Redis 연결 성공")
        
        # 데이터베이스 연결
        self.db_engine = create_async_engine(get_database_url())
        async_session = sessionmaker(
            self.db_engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
        self.db_session = async_session
        logger.info("데이터베이스 연결 성공")
        
        logger.info("실시간 데이터 수집기 초기화 완료")
        
    except Exception as e:
        logger.error(f"수집기 초기화 실패: {e}")
        raise
        
async def start(self):
    """수집기 시작"""
    if self.is_running:
        logger.warning("수집기가 이미 실행 중입니다")
        return
        
    self.is_running = True
    logger.info("실시간 데이터 수집기 시작")
    
    # 백그라운드 태스크 시작
    tasks = [
        self._data_processor_task(),
        self._batch_flusher_task(),
        self._connection_monitor_task(),
    ]
    
    if self.config.enable_monitoring:
        tasks.append(self._performance_monitor_task())
        
    for task_coro in tasks:
        task = asyncio.create_task(task_coro)
        self.running_tasks.add(task)
        task.add_done_callback(self.running_tasks.discard)
        
async def stop(self):
    """수집기 정지"""
    self.is_running = False
    logger.info("실시간 데이터 수집기 정지 중...")
    
    # 모든 태스크 취소
    for task in self.running_tasks:
        task.cancel()
        
    await asyncio.gather(*self.running_tasks, return_exceptions=True)
    self.running_tasks.clear()
    
    # 연결 정리
    await self._cleanup_connections()
    
    # 남은 데이터 플러시
    await self._flush_remaining_data()
    
    # 외부 연결 정리
    if self.redis:
        await self.redis.close()
    if self.db_engine:
        await self.db_engine.dispose()
        
    logger.info("실시간 데이터 수집기 정지 완료")
    
async def subscribe_ticker(self, ticker: str) -> bool:
    """종목 구독 추가"""
    try:
        if ticker in self.subscribed_tickers:
            logger.warning(f"종목 {ticker}는 이미 구독 중입니다")
            return True
            
        # 연결 풀에서 사용 가능한 클라이언트 찾기
        client = await self._get_available_client()
        if not client:
            logger.error(f"사용 가능한 WebSocket 연결이 없습니다: {ticker}")
            return False
            
        # 실시간 데이터 구독
        await client.subscribe_realtime_price(
            ticker, 
            lambda data: self._handle_price_data(ticker, data)
        )
        await client.subscribe_realtime_orderbook(
            ticker,
            lambda data: self._handle_orderbook_data(ticker, data)
        )
        
        self.subscribed_tickers.add(ticker)
        self.connection_stats[ticker] = ConnectionStats(
            ticker=ticker,
            connected_at=datetime.now()
        )
        
        logger.info(f"종목 {ticker} 구독 시작")
        return True
        
    except Exception as e:
        logger.error(f"종목 {ticker} 구독 실패: {e}")
        return False
        
async def unsubscribe_ticker(self, ticker: str) -> bool:
    """종목 구독 해제"""
    try:
        if ticker not in self.subscribed_tickers:
            logger.warning(f"종목 {ticker}는 구독 중이 아닙니다")
            return True
            
        # 해당 종목을 구독 중인 클라이언트 찾기
        client = self._find_client_for_ticker(ticker)
        if client:
            await client.unsubscribe_realtime_price(ticker)
            await client.unsubscribe_realtime_orderbook(ticker)
            
        self.subscribed_tickers.discard(ticker)
        if ticker in self.connection_stats:
            del self.connection_stats[ticker]
            
        logger.info(f"종목 {ticker} 구독 해제")
        return True
        
    except Exception as e:
        logger.error(f"종목 {ticker} 구독 해제 실패: {e}")
        return False
        
async def subscribe_multiple_tickers(self, tickers: List[str]) -> Dict[str, bool]:
    """다수 종목 일괄 구독"""
    results = {}
    
    # 배치 크기로 나누어 처리
    batch_size = min(len(tickers), self.config.max_concurrent_connections * 10)
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        tasks = [self.subscribe_ticker(ticker) for ticker in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for ticker, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                results[ticker] = False
                logger.error(f"종목 {ticker} 구독 실패: {result}")
            else:
                results[ticker] = result
                
        # 배치 간 딜레이
        if i + batch_size < len(tickers):
            await asyncio.sleep(0.1)
            
    return results
    
async def _get_available_client(self) -> Optional[KISApiClient]:
    """사용 가능한 WebSocket 클라이언트 반환"""
    # 기존 연결 중 사용 가능한 것 찾기
    for client_id, client in self.active_connections.items():
        if client.is_connected and len(client.price_callbacks) < 50:  # 연결당 최대 50개 종목
            return client
            
    # 새 연결 생성 (최대 연결 수 확인)
    if len(self.active_connections) < self.config.max_concurrent_connections:
        client_id = f"client_{len(self.active_connections)}"
        client = KISApiClient(self.kis_config)
        
        try:
            await client.initialize()
            await client.connect_websocket()
            self.active_connections[client_id] = client
            logger.info(f"새 WebSocket 연결 생성: {client_id}")
            return client
            
        except Exception as e:
            logger.error(f"WebSocket 연결 생성 실패: {e}")
            return None
            
    return None
    
def _find_client_for_ticker(self, ticker: str) -> Optional[KISApiClient]:
    """특정 종목을 구독 중인 클라이언트 찾기"""
    for client in self.active_connections.values():
        if ticker in client.price_callbacks:
            return client
    return None
    
async def _handle_price_data(self, ticker: str, data: Dict):
    """실시간 주가 데이터 처리"""
    try:
        # 데이터 검증
        if not self.validator.validate_price_data(data):
            logger.warning(f"유효하지 않은 주가 데이터: {ticker}")
            return
            
        # 데이터 포인트 생성
        data_point = RealtimeDataPoint(
            ticker=ticker,
            data_type='price',
            data=data,
            timestamp=data.get('timestamp', datetime.now()),
            received_at=datetime.now()
        )
        
        # 큐에 추가 (백프레셔 제어)
        try:
            self.data_queue.put_nowait(data_point)
        except asyncio.QueueFull:
            logger.warning(f"데이터 큐 포화: {ticker} 주가 데이터 드롭")
            self.performance_stats['error_rate'].append(1)
            
        # 통계 업데이트
        if ticker in self.connection_stats:
            stats = self.connection_stats[ticker]
            stats.last_message_at = datetime.now()
            stats.message_count += 1
            
    except Exception as e:
        logger.error(f"주가 데이터 처리 오류 ({ticker}): {e}")
        if ticker in self.connection_stats:
            self.connection_stats[ticker].error_count += 1
            
async def _handle_orderbook_data(self, ticker: str, data: Dict):
    """실시간 호가 데이터 처리"""
    try:
        # 데이터 검증
        if not self.validator.validate_orderbook_data(data):
            logger.warning(f"유효하지 않은 호가 데이터: {ticker}")
            return
            
        # 데이터 포인트 생성
        data_point = RealtimeDataPoint(
            ticker=ticker,
            data_type='orderbook',
            data=data,
            timestamp=data.get('timestamp', datetime.now()),
            received_at=datetime.now()
        )
        
        # 큐에 추가
        try:
            self.data_queue.put_nowait(data_point)
        except asyncio.QueueFull:
            logger.warning(f"데이터 큐 포화: {ticker} 호가 데이터 드롭")
            
        # 통계 업데이트
        if ticker in self.connection_stats:
            stats = self.connection_stats[ticker]
            stats.last_message_at = datetime.now()
            stats.message_count += 1
            
    except Exception as e:
        logger.error(f"호가 데이터 처리 오류 ({ticker}): {e}")
        if ticker in self.connection_stats:
            self.connection_stats[ticker].error_count += 1
            
async def _data_processor_task(self):
    """데이터 처리 태스크"""
    while self.is_running:
        try:
            # 큐에서 데이터 가져오기 (타임아웃 설정)
            try:
                data_point = await asyncio.wait_for(
                    self.data_queue.get(), 
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
                
            # 배치 버퍼에 추가
            self.batch_buffer.append(data_point)
            
            # Redis 캐시 업데이트 (최신 데이터)
            await self._update_redis_cache(data_point)
            
            # 배치 크기 도달 시 플러시
            if len(self.batch_buffer) >= self.config.batch_size:
                await self._flush_batch()
                
        except Exception as e:
            logger.error(f"데이터 처리 태스크 오류: {e}")
            await asyncio.sleep(1)
            
async def _batch_flusher_task(self):
    """배치 플러시 태스크"""
    while self.is_running:
        try:
            current_time = time.time()
            
            # 플러시 주기 도달 시 또는 배치가 있을 때
            if (current_time - self.last_flush_time >= self.config.flush_interval and 
                self.batch_buffer):
                await self._flush_batch()
                
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"배치 플러시 태스크 오류: {e}")
            await asyncio.sleep(1)
            
async def _update_redis_cache(self, data_point: RealtimeDataPoint):
    """Redis 캐시 업데이트"""
    try:
        if not self.redis:
            return
            
        # 캐시 키 생성
        cache_key = f"realtime:{data_point.ticker}:{data_point.data_type}"
        
        # 데이터 직렬화
        cache_data = {
            'data': data_point.data,
            'timestamp': data_point.timestamp.isoformat(),
            'received_at': data_point.received_at.isoformat()
        }
        
        # Redis에 저장 (만료 시간 설정)
        await self.redis.setex(
            cache_key,
            self.config.cache_expire,
            json.dumps(cache_data, default=str)
        )
        
    except Exception as e:
        logger.error(f"Redis 캐시 업데이트 오류: {e}")
        
async def _flush_batch(self):
    """배치 데이터 데이터베이스 저장"""
    if not self.batch_buffer:
        return
        
    try:
        async with self.db_session() as session:
            # 데이터 타입별로 그룹화
            grouped_data = defaultdict(list)
            for data_point in self.batch_buffer:
                grouped_data[data_point.data_type].append(data_point)
                
            # 타입별 배치 삽입
            for data_type, data_points in grouped_data.items():
                await self._insert_batch_by_type(session, data_type, data_points)
                
            await session.commit()
            
        logger.debug(f"배치 저장 완료: {len(self.batch_buffer)}개 레코드")
        
        # 배치 버퍼 초기화
        self.batch_buffer.clear()
        self.last_flush_time = time.time()
        
    except Exception as e:
        logger.error(f"배치 저장 오류: {e}")
        # 에러 발생 시 배치 버퍼 유지 (재시도 가능)
        
async def _insert_batch_by_type(self, session: AsyncSession, data_type: str, data_points: List[RealtimeDataPoint]):
    """데이터 타입별 배치 삽입"""
    try:
        if data_type == 'price':
            await self._insert_price_batch(session, data_points)
        elif data_type == 'orderbook':
            await self._insert_orderbook_batch(session, data_points)
        else:
            logger.warning(f"알 수 없는 데이터 타입: {data_type}")
            
    except Exception as e:
        logger.error(f"배치 삽입 오류 ({data_type}): {e}")
        raise
        
async def _insert_price_batch(self, session: AsyncSession, data_points: List[RealtimeDataPoint]):
    """주가 데이터 배치 삽입"""
    from ...models.market import RealtimePriceData
    
    records = []
    for dp in data_points:
        record = RealtimePriceData(
            ticker=dp.ticker,
            current_price=dp.data.get('current_price', 0),
            change=dp.data.get('change', 0),
            change_rate=dp.data.get('change_rate', 0.0),
            volume=dp.data.get('volume', 0),
            timestamp=dp.timestamp,
            received_at=dp.received_at
        )
        records.append(record)
        
    session.add_all(records)
    
async def _insert_orderbook_batch(self, session: AsyncSession, data_points: List[RealtimeDataPoint]):
    """호가 데이터 배치 삽입"""
    from ...models.market import RealtimeOrderbookData
    
    records = []
    for dp in data_points:
        record = RealtimeOrderbookData(
            ticker=dp.ticker,
            ask_prices=dp.data.get('ask_prices', []),
            ask_volumes=dp.data.get('ask_volumes', []),
            bid_prices=dp.data.get('bid_prices', []),
            bid_volumes=dp.data.get('bid_volumes', []),
            timestamp=dp.timestamp,
            received_at=dp.received_at
        )
        records.append(record)
        
    session.add_all(records)
    
async def _connection_monitor_task(self):
    """연결 모니터링 태스크"""
    while self.is_running:
        try:
            # 연결 상태 체크
            disconnected_clients = []
            for client_id, client in self.active_connections.items():
                if not client.is_connected:
                    disconnected_clients.append(client_id)
                    logger.warning(f"WebSocket 연결 끊어짐: {client_id}")
                    
            # 끊어진 연결 정리 및 재연결
            for client_id in disconnected_clients:
                await self._reconnect_client(client_id)
                
            # 종목별 메시지 수신 상태 체크
            current_time = datetime.now()
            for ticker, stats in self.connection_stats.items():
                if (stats.last_message_at and 
                    current_time - stats.last_message_at > timedelta(minutes=5)):
                    logger.warning(f"종목 {ticker} 메시지 수신 중단 (5분 초과)")
                    
            await asyncio.sleep(10)  # 10초마다 체크
            
        except Exception as e:
            logger.error(f"연결 모니터링 태스크 오류: {e}")
            await asyncio.sleep(10)
            
async def _reconnect_client(self, client_id: str):
    """클라이언트 재연결"""
    try:
        old_client = self.active_connections.get(client_id)
        if old_client:
            await old_client.close()
            
        # 새 클라이언트 생성
        new_client = KISApiClient(self.kis_config)
        await new_client.initialize()
        await new_client.connect_websocket()
        
        self.active_connections[client_id] = new_client
        
        # 기존 구독 복원 (해당 클라이언트가 담당하던 종목들)
        # TODO: 구독 복원 로직 구현
        
        logger.info(f"WebSocket 클라이언트 재연결 완료: {client_id}")
        
    except Exception as e:
        logger.error(f"클라이언트 재연결 실패 ({client_id}): {e}")
        
async def _performance_monitor_task(self):
    """성능 모니터링 태스크"""
    while self.is_running:
        try:
            current_time = time.time()
            
            # 초당 메시지 수 계산
            messages_in_last_second = sum(
                stats.message_count for stats in self.connection_stats.values()
            ) - sum(self.performance_stats['messages_per_second'])
            
            self.performance_stats['messages_per_second'].append(messages_in_last_second)
            self.performance_stats['queue_size_history'].append(self.data_queue.qsize())
            
            # 성능 통계 로깅 (1분마다)
            if current_time - self.performance_stats['last_stats_time'] >= 60:
                await self._log_performance_stats()
                self.performance_stats['last_stats_time'] = current_time
                
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"성능 모니터링 태스크 오류: {e}")
            await asyncio.sleep(1)
            
async def _log_performance_stats(self):
    """성능 통계 로깅"""
    avg_msg_per_sec = sum(self.performance_stats['messages_per_second']) / max(len(self.performance_stats['messages_per_second']), 1)
    avg_queue_size = sum(self.performance_stats['queue_size_history']) / max(len(self.performance_stats['queue_size_history']), 1)
    
    logger.info(f"성능 통계 - 평균 메시지/초: {avg_msg_per_sec:.1f}, 평균 큐 크기: {avg_queue_size:.1f}")
    logger.info(f"활성 연결: {len(self.active_connections)}, 구독 종목: {len(self.subscribed_tickers)}")
    
async def _cleanup_connections(self):
    """연결 정리"""
    for client_id, client in self.active_connections.items():
        try:
            await client.close()
        except Exception as e:
            logger.error(f"연결 정리 오류 ({client_id}): {e}")
            
    self.active_connections.clear()
    self.connection_stats.clear()
    self.subscribed_tickers.clear()
    
async def _flush_remaining_data(self):
    """남은 데이터 플러시"""
    if self.batch_buffer:
        await self._flush_batch()
        
    # 큐에 남은 데이터 처리
    remaining_data = []
    while not self.data_queue.empty():
        try:
            data_point = self.data_queue.get_nowait()
            remaining_data.append(data_point)
        except asyncio.QueueEmpty:
            break
            
    if remaining_data:
        self.batch_buffer.extend(remaining_data)
        await self._flush_batch()
        
def get_stats(self) -> Dict[str, Any]:
    """수집기 통계 반환"""
    return {
        'is_running': self.is_running,
        'active_connections': len(self.active_connections),
        'subscribed_tickers': len(self.subscribed_tickers),
        'queue_size': self.data_queue.qsize(),
        'batch_buffer_size': len(self.batch_buffer),
        'connection_stats': {
            ticker: {
                'message_count': stats.message_count,
                'error_count': stats.error_count,
                'reconnect_count': stats.reconnect_count,
                'last_message_at': stats.last_message_at.isoformat() if stats.last_message_at else None
            }
            for ticker, stats in self.connection_stats.items()
        },
        'performance': {
            'avg_messages_per_second': sum(self.performance_stats['messages_per_second']) / max(len(self.performance_stats['messages_per_second']), 1),
            'avg_queue_size': sum(self.performance_stats['queue_size_history']) / max(len(self.performance_stats['queue_size_history']), 1)
        }
    }
```

# ==================== 전역 수집기 인스턴스 ====================

_realtime_collector: Optional[RealtimeDataCollector] = None

async def get_realtime_collector() -> RealtimeDataCollector:
“”“전역 실시간 수집기 인스턴스 반환”””
global _realtime_collector

```
if _realtime_collector is None:
    from ...core.config import get_kis_config, get_realtime_config
    
    kis_config = get_kis_config()
    realtime_config = get_realtime_config()
    
    _realtime_collector = RealtimeDataCollector(kis_config, realtime_config)
    await _realtime_collector.initialize()
    
return _realtime_collector
```

async def start_realtime_collection():
“”“실시간 수집 시작”””
collector = await get_realtime_collector()
await collector.start()

async def stop_realtime_collection():
“”“실시간 수집 정지”””
global _realtime_collector

```
if _realtime_collector:
    await _realtime_collector.stop()
    _realtime_collector = None
```

# ==================== 사용 예제 ====================

async def example_usage():
“”“사용 예제”””
from …core.config import get_kis_config

```
# 설정
kis_config = get_kis_config()
realtime_config = RealtimeConfig(
    max_concurrent_connections=3,
    batch_size=50,
    flush_interval=3
)

# 수집기 생성 및 시작
collector = RealtimeDataCollector(kis_config, realtime_config)
await collector.initialize()
await collector.start()

try:
    # 종목 구독
    tickers = ["005930", "000660", "035420"]  # 삼성전자, SK하이닉스, 네이버
    results = await collector.subscribe_multiple_tickers(tickers)
    print("구독 결과:", results)
    
    # 30초간 데이터 수집
    await asyncio.sleep(30)
    
    # 통계 확인
    stats = collector.get_stats()
    print("수집 통계:", stats)
    
finally:
    # 수집기 정지
    await collector.stop()
```

if **name** == “**main**”:
asyncio.run(example_usage())