# backend/app/data/processors/api_failure_handler.py

“””
외부 API 장애 대응 로직

- 실시간 API 상태 모니터링
- 자동 장애 감지 및 복구
- 폴백 메커니즘 및 데이터 소스 전환
- Circuit Breaker 패턴 구현
- 지능형 재시도 및 백오프
  “””
  import asyncio
  import aiohttp
  import time
  from datetime import datetime, timedelta
  from typing import Dict, List, Optional, Any, Callable, Union
  import logging
  from dataclasses import dataclass, asdict
  from enum import Enum
  import json
  import random
  from collections import deque, defaultdict
  import statistics

from app.core.config import get_settings
from app.core.redis import get_redis_client
from app.data.processors.data_validator import DataValidator

settings = get_settings()
logger = logging.getLogger(**name**)

class APIStatus(Enum):
“”“API 상태”””
HEALTHY = “healthy”
DEGRADED = “degraded”
FAILED = “failed”
MAINTENANCE = “maintenance”
CIRCUIT_OPEN = “circuit_open”

class RetryStrategy(Enum):
“”“재시도 전략”””
EXPONENTIAL_BACKOFF = “exponential_backoff”
LINEAR_BACKOFF = “linear_backoff”
FIXED_INTERVAL = “fixed_interval”
FIBONACCI_BACKOFF = “fibonacci_backoff”

@dataclass
class APIEndpoint:
“”“API 엔드포인트 정보”””
name: str
url: str
priority: int  # 1=highest, 5=lowest
timeout_seconds: int
max_retries: int
retry_strategy: RetryStrategy
health_check_interval: int
circuit_breaker_threshold: int
is_primary: bool
fallback_endpoints: List[str]

@dataclass
class APIHealthMetrics:
“”“API 건강 지표”””
endpoint_name: str
status: APIStatus
response_time_ms: float
success_rate: float
error_count: int
last_success: Optional[datetime]
last_failure: Optional[datetime]
consecutive_failures: int
circuit_breaker_open: bool
next_retry_time: Optional[datetime]

@dataclass
class FailureEvent:
“”“장애 이벤트”””
endpoint_name: str
error_type: str
error_message: str
timestamp: datetime
response_time_ms: Optional[float]
http_status: Optional[int]
retry_attempt: int
fallback_used: bool

class CircuitBreaker:
“”“Circuit Breaker 패턴 구현”””

```
def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
    self.failure_threshold = failure_threshold
    self.timeout_seconds = timeout_seconds
    self.failure_count = 0
    self.last_failure_time = None
    self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

def call(self, func: Callable) -> Callable:
    """함수 호출 래퍼"""
    async def wrapper(*args, **kwargs):
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise Exception(f"Circuit breaker is OPEN. Next attempt in {self._time_until_reset():.1f}s")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    return wrapper

def _on_success(self):
    """성공 시 처리"""
    self.failure_count = 0
    self.state = 'CLOSED'

def _on_failure(self):
    """실패 시 처리"""
    self.failure_count += 1
    self.last_failure_time = time.time()
    
    if self.failure_count >= self.failure_threshold:
        self.state = 'OPEN'

def _should_attempt_reset(self) -> bool:
    """재시도 가능한지 확인"""
    return (time.time() - self.last_failure_time) >= self.timeout_seconds

def _time_until_reset(self) -> float:
    """재시도까지 남은 시간"""
    if self.last_failure_time:
        elapsed = time.time() - self.last_failure_time
        return max(0, self.timeout_seconds - elapsed)
    return 0
```

class APIFailureHandler:
“”“외부 API 장애 대응 시스템”””

```
def __init__(self):
    self.redis_client = get_redis_client()
    self.validator = DataValidator()
    self.session = None
    
    # API 엔드포인트 설정
    self.api_endpoints = {
        # 한투증권 API
        'kis_primary': APIEndpoint(
            name='kis_primary',
            url='https://openapi.koreainvestment.com',
            priority=1,
            timeout_seconds=10,
            max_retries=3,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            health_check_interval=30,
            circuit_breaker_threshold=5,
            is_primary=True,
            fallback_endpoints=['kis_backup', 'yahoo_finance']
        ),
        'kis_backup': APIEndpoint(
            name='kis_backup',
            url='https://openapi-backup.koreainvestment.com',
            priority=2,
            timeout_seconds=15,
            max_retries=2,
            retry_strategy=RetryStrategy.LINEAR_BACKOFF,
            health_check_interval=60,
            circuit_breaker_threshold=3,
            is_primary=False,
            fallback_endpoints=['yahoo_finance']
        ),
        
        # Yahoo Finance API
        'yahoo_finance': APIEndpoint(
            name='yahoo_finance',
            url='https://query1.finance.yahoo.com',
            priority=3,
            timeout_seconds=20,
            max_retries=5,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            health_check_interval=45,
            circuit_breaker_threshold=10,
            is_primary=False,
            fallback_endpoints=['alpha_vantage']
        ),
        
        # Alpha Vantage (폴백)
        'alpha_vantage': APIEndpoint(
            name='alpha_vantage',
            url='https://www.alphavantage.co',
            priority=4,
            timeout_seconds=30,
            max_retries=3,
            retry_strategy=RetryStrategy.FIBONACCI_BACKOFF,
            health_check_interval=120,
            circuit_breaker_threshold=5,
            is_primary=False,
            fallback_endpoints=['local_cache']
        ),
        
        # FRED API
        'fred_api': APIEndpoint(
            name='fred_api',
            url='https://api.stlouisfed.org',
            priority=1,
            timeout_seconds=15,
            max_retries=4,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            health_check_interval=60,
            circuit_breaker_threshold=5,
            is_primary=True,
            fallback_endpoints=['local_economic_cache']
        ),
        
        # 네이버 금융
        'naver_finance': APIEndpoint(
            name='naver_finance',
            url='https://finance.naver.com',
            priority=1,
            timeout_seconds=10,
            max_retries=3,
            retry_strategy=RetryStrategy.LINEAR_BACKOFF,
            health_check_interval=30,
            circuit_breaker_threshold=8,
            is_primary=True,
            fallback_endpoints=['daum_finance', 'local_news_cache']
        ),
        
        # 로컬 캐시 (최종 폴백)
        'local_cache': APIEndpoint(
            name='local_cache',
            url='local://cache',
            priority=5,
            timeout_seconds=1,
            max_retries=1,
            retry_strategy=RetryStrategy.FIXED_INTERVAL,
            health_check_interval=300,
            circuit_breaker_threshold=1,
            is_primary=False,
            fallback_endpoints=[]
        )
    }
    
    # Circuit Breaker 인스턴스들
    self.circuit_breakers = {
        name: CircuitBreaker(
            failure_threshold=endpoint.circuit_breaker_threshold,
            timeout_seconds=60
        )
        for name, endpoint in self.api_endpoints.items()
    }
    
    # 헬스 체크 메트릭
    self.health_metrics = {}
    self.failure_events = deque(maxlen=1000)  # 최근 1000개 장애 이벤트
    
    # 재시도 설정
    self.retry_configs = {
        RetryStrategy.EXPONENTIAL_BACKOFF: {
            'base_delay': 1.0,
            'max_delay': 60.0,
            'multiplier': 2.0,
            'jitter': True
        },
        RetryStrategy.LINEAR_BACKOFF: {
            'base_delay': 2.0,
            'increment': 1.0,
            'max_delay': 30.0,
            'jitter': False
        },
        RetryStrategy.FIXED_INTERVAL: {
            'delay': 5.0,
            'jitter': False
        },
        RetryStrategy.FIBONACCI_BACKOFF: {
            'base_delay': 1.0,
            'max_delay': 45.0,
            'sequence': [1, 1, 2, 3, 5, 8, 13, 21, 34],
            'jitter': True
        }
    }
    
    # 데이터 소스 우선순위
    self.data_source_priority = {
        'market_data': ['kis_primary', 'kis_backup', 'yahoo_finance', 'local_cache'],
        'economic_data': ['fred_api', 'local_economic_cache'],
        'news_data': ['naver_finance', 'daum_finance', 'local_news_cache'],
        'global_data': ['yahoo_finance', 'alpha_vantage', 'local_cache']
    }

async def __aenter__(self):
    """비동기 컨텍스트 매니저 진입"""
    connector = aiohttp.TCPConnector(
        limit=50,
        limit_per_host=10,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    
    self.session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={'User-Agent': 'MastersEye-API-Monitor/1.0'}
    )
    
    # API 상태 모니터링 시작
    asyncio.create_task(self._start_health_monitoring())
    
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    """비동기 컨텍스트 매니저 종료"""
    if self.session:
        await self.session.close()

async def robust_api_call(
    self, 
    data_type: str,
    api_function: Callable,
    *args,
    **kwargs
) -> Any:
    """강건한 API 호출 (자동 폴백 및 재시도)"""
    try:
        data_sources = self.data_source_priority.get(data_type, ['local_cache'])
        last_exception = None
        
        for source_name in data_sources:
            try:
                endpoint = self.api_endpoints.get(source_name)
                if not endpoint:
                    continue
                
                # Circuit Breaker 확인
                circuit_breaker = self.circuit_breakers[source_name]
                if circuit_breaker.state == 'OPEN':
                    logger.warning(f"Circuit breaker OPEN for {source_name}, skipping")
                    continue
                
                # 건강 상태 확인
                health = await self._get_endpoint_health(source_name)
                if health.status == APIStatus.FAILED:
                    logger.warning(f"Endpoint {source_name} is marked as FAILED, skipping")
                    continue
                
                # API 호출 시도
                logger.info(f"Attempting API call to {source_name}")
                start_time = time.time()
                
                result = await self._execute_with_retry(
                    source_name,
                    api_function,
                    *args,
                    **kwargs
                )
                
                # 성공 시 메트릭 업데이트
                response_time = (time.time() - start_time) * 1000
                await self._record_success(source_name, response_time)
                
                logger.info(f"API call successful: {source_name} ({response_time:.1f}ms)")
                return result
                
            except Exception as e:
                last_exception = e
                response_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
                
                # 실패 기록
                await self._record_failure(source_name, str(e), response_time)
                
                logger.warning(f"API call failed: {source_name} - {str(e)}")
                continue
        
        # 모든 소스 실패시 최종 처리
        if last_exception:
            # 캐시된 데이터 시도
            cached_result = await self._get_cached_fallback_data(data_type)
            if cached_result:
                logger.info(f"Using cached fallback data for {data_type}")
                return cached_result
            
            # 기본값 반환 또는 예외 발생
            default_result = await self._get_default_fallback_data(data_type)
            if default_result:
                logger.warning(f"Using default fallback data for {data_type}")
                return default_result
            
            raise last_exception
        
        raise Exception(f"No available data sources for {data_type}")
        
    except Exception as e:
        logger.error(f"Critical error in robust_api_call: {e}")
        raise

async def _execute_with_retry(
    self,
    endpoint_name: str,
    api_function: Callable,
    *args,
    **kwargs
) -> Any:
    """재시도 로직으로 API 실행"""
    endpoint = self.api_endpoints[endpoint_name]
    circuit_breaker = self.circuit_breakers[endpoint_name]
    
    last_exception = None
    
    for attempt in range(endpoint.max_retries + 1):
        try:
            # Circuit Breaker로 감싸서 실행
            @circuit_breaker.call
            async def protected_call():
                return await api_function(*args, **kwargs)
            
            result = await protected_call()
            return result
            
        except Exception as e:
            last_exception = e
            
            # 장애 이벤트 기록
            failure_event = FailureEvent(
                endpoint_name=endpoint_name,
                error_type=type(e).__name__,
                error_message=str(e),
                timestamp=datetime.now(),
                response_time_ms=None,
                http_status=getattr(e, 'status', None),
                retry_attempt=attempt,
                fallback_used=False
            )
            self.failure_events.append(failure_event)
            
            # 마지막 시도가 아니면 재시도 대기
            if attempt < endpoint.max_retries:
                delay = self._calculate_retry_delay(endpoint.retry_strategy, attempt)
                logger.info(f"Retrying {endpoint_name} in {delay:.1f}s (attempt {attempt + 1}/{endpoint.max_retries})")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All retry attempts failed for {endpoint_name}")
    
    raise last_exception

def _calculate_retry_delay(self, strategy: RetryStrategy, attempt: int) -> float:
    """재시도 지연 시간 계산"""
    config = self.retry_configs[strategy]
    
    if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        base_delay = config['base_delay']
        multiplier = config['multiplier']
        max_delay = config['max_delay']
        
        delay = min(base_delay * (multiplier ** attempt), max_delay)
        
        if config['jitter']:
            delay *= (0.5 + random.random() * 0.5)  # ±50% jitter
            
    elif strategy == RetryStrategy.LINEAR_BACKOFF:
        base_delay = config['base_delay']
        increment = config['increment']
        max_delay = config['max_delay']
        
        delay = min(base_delay + (increment * attempt), max_delay)
        
    elif strategy == RetryStrategy.FIXED_INTERVAL:
        delay = config['delay']
        
        if config.get('jitter', False):
            delay *= (0.8 + random.random() * 0.4)  # ±20% jitter
            
    elif strategy == RetryStrategy.FIBONACCI_BACKOFF:
        sequence = config['sequence']
        base_delay = config['base_delay']
        max_delay = config['max_delay']
        
        if attempt < len(sequence):
            delay = min(base_delay * sequence[attempt], max_delay)
        else:
            delay = max_delay
            
        if config['jitter']:
            delay *= (0.7 + random.random() * 0.6)  # ±30% jitter
    else:
        delay = 5.0  # 기본값
    
    return delay

async def _start_health_monitoring(self):
    """API 헬스 모니터링 시작"""
    try:
        logger.info("Starting API health monitoring")
        
        while True:
            # 모든 엔드포인트 헬스 체크
            health_check_tasks = []
            
            for endpoint_name, endpoint in self.api_endpoints.items():
                if endpoint.url.startswith('local://'):
                    continue  # 로컬 엔드포인트는 스킵
                
                task = asyncio.create_task(
                    self._perform_health_check(endpoint_name)
                )
                health_check_tasks.append(task)
            
            # 병렬 헬스 체크 실행
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
            
            # 헬스 메트릭 업데이트
            await self._update_health_metrics()
            
            # 30초 대기 후 다음 체크
            await asyncio.sleep(30)
            
    except Exception as e:
        logger.error(f"Health monitoring error: {e}")

async def _perform_health_check(self, endpoint_name: str):
    """개별 엔드포인트 헬스 체크"""
    try:
        endpoint = self.api_endpoints[endpoint_name]
        start_time = time.time()
        
        # 간단한 HEAD 요청으로 헬스 체크
        timeout = aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
        
        async with self.session.head(
            endpoint.url,
            timeout=timeout,
            allow_redirects=True
        ) as response:
            response_time = (time.time() - start_time) * 1000
            
            if response.status < 400:
                await self._record_success(endpoint_name, response_time)
                status = APIStatus.HEALTHY
            elif response.status < 500:
                status = APIStatus.DEGRADED
            else:
                await self._record_failure(endpoint_name, f"HTTP {response.status}", response_time)
                status = APIStatus.FAILED
            
            # 헬스 메트릭 업데이트
            if endpoint_name not in self.health_metrics:
                self.health_metrics[endpoint_name] = APIHealthMetrics(
                    endpoint_name=endpoint_name,
                    status=status,
                    response_time_ms=response_time,
                    success_rate=100.0 if status == APIStatus.HEALTHY else 0.0,
                    error_count=0,
                    last_success=datetime.now() if status == APIStatus.HEALTHY else None,
                    last_failure=None,
                    consecutive_failures=0,
                    circuit_breaker_open=False,
                    next_retry_time=None
                )
            else:
                metrics = self.health_metrics[endpoint_name]
                metrics.status = status
                metrics.response_time_ms = response_time
                
                if status == APIStatus.HEALTHY:
                    metrics.last_success = datetime.now()
                    metrics.consecutive_failures = 0
                else:
                    metrics.consecutive_failures += 1
                    metrics.last_failure = datetime.now()
            
    except Exception as e:
        # 헬스 체크 실패
        response_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
        await self._record_failure(endpoint_name, str(e), response_time)
        
        if endpoint_name in self.health_metrics:
            metrics = self.health_metrics[endpoint_name]
            metrics.status = APIStatus.FAILED
            metrics.consecutive_failures += 1
            metrics.last_failure = datetime.now()

async def _record_success(self, endpoint_name: str, response_time_ms: float):
    """성공 기록"""
    try:
        # Redis에 성공 메트릭 저장
        success_key = f"api_success:{endpoint_name}:{datetime.now().strftime('%Y%m%d_%H')}"
        await self.redis_client.incr(success_key)
        await self.redis_client.expire(success_key, 86400)  # 24시간 TTL
        
        # 응답 시간 저장
        response_time_key = f"api_response_time:{endpoint_name}"
        await self.redis_client.lpush(response_time_key, response_time_ms)
        await self.redis_client.ltrim(response_time_key, 0, 99)  # 최근 100개만 보관
        
        # Circuit Breaker 성공 처리
        circuit_breaker = self.circuit_breakers[endpoint_name]
        circuit_breaker._on_success()
        
    except Exception as e:
        logger.error(f"Error recording success for {endpoint_name}: {e}")

async def _record_failure(self, endpoint_name: str, error_message: str, response_time_ms: float):
    """실패 기록"""
    try:
        # Redis에 실패 메트릭 저장
        failure_key = f"api_failure:{endpoint_name}:{datetime.now().strftime('%Y%m%d_%H')}"
        await self.redis_client.incr(failure_key)
        await self.redis_client.expire(failure_key, 86400)  # 24시간 TTL
        
        # 에러 상세 정보 저장
        error_detail = {
            'timestamp': datetime.now().isoformat(),
            'error_message': error_message,
            'response_time_ms': response_time_ms
        }
        
        error_key = f"api_errors:{endpoint_name}"
        await self.redis_client.lpush(error_key, json.dumps(error_detail))
        await self.redis_client.ltrim(error_key, 0, 49)  # 최근 50개만 보관
        
        # Circuit Breaker 실패 처리
        circuit_breaker = self.circuit_breakers[endpoint_name]
        circuit_breaker._on_failure()
        
        # 장애 알림 (심각한 경우)
        if circuit_breaker.state == 'OPEN':
            await self._send_failure_alert(endpoint_name, error_message)
        
    except Exception as e:
        logger.error(f"Error recording failure for {endpoint_name}: {e}")

async def _get_endpoint_health(self, endpoint_name: str) -> APIHealthMetrics:
    """엔드포인트 건강 상태 조회"""
    try:
        if endpoint_name in self.health_metrics:
            return self.health_metrics[endpoint_name]
        
        # 기본 메트릭 반환
        return APIHealthMetrics(
            endpoint_name=endpoint_name,
            status=APIStatus.HEALTHY,
            response_time_ms=0.0,
            success_rate=100.0,
            error_count=0,
            last_success=None,
            last_failure=None,
            consecutive_failures=0,
            circuit_breaker_open=False,
            next_retry_time=None
        )
        
    except Exception as e:
        logger.error(f"Error getting endpoint health for {endpoint_name}: {e}")
        return APIHealthMetrics(
            endpoint_name=endpoint_name,
            status=APIStatus.FAILED,
            response_time_ms=999999.0,
            success_rate=0.0,
            error_count=999,
            last_success=None,
            last_failure=datetime.now(),
            consecutive_failures=999,
            circuit_breaker_open=True,
            next_retry_time=datetime.now() + timedelta(minutes=5)
        )

async def _update_health_metrics(self):
    """헬스 메트릭 종합 업데이트"""
    try:
        for endpoint_name in self.api_endpoints.keys():
            if endpoint_name.startswith('local'):
                continue
            
            # 최근 1시간 성공/실패 통계
            current_hour = datetime.now().strftime('%Y%m%d_%H')
            success_key = f"api_success:{endpoint_name}:{current_hour}"
            failure_key = f"api_failure:{endpoint_name}:{current_hour}"
            
            success_count = int(await self.redis_client.get(success_key) or 0)
            failure_count = int(await self.redis_client.get(failure_key) or 0)
            
            total_requests = success_count + failure_count
            success_rate = (success_count / total_requests * 100) if total_requests > 0 else 100.0
            
            # 평균 응답 시간
            response_time_key = f"api_response_time:{endpoint_name}"
            response_times = await self.redis_client.lrange(response_time_key, 0, -1)
            
            if response_times:
                avg_response_time = statistics.mean([float(rt) for rt in response_times])
            else:
                avg_response_time = 0.0
            
            # 메트릭 업데이트
            if endpoint_name in self.health_metrics:
                metrics = self.health_metrics[endpoint_name]
                metrics.success_rate = success_rate
                metrics.error_count = failure_count
                metrics.response_time_ms = avg_response_time
                metrics.circuit_breaker_open = self.circuit_breakers[endpoint_name].state == 'OPEN'
        
    except Exception as e:
        logger.error(f"Error updating health metrics: {e}")

async def _get_cached_fallback_data(self, data_type: str) -> Optional[Any]:
    """캐시된 폴백 데이터 조회"""
    try:
        cache_key = f"fallback_data:{data_type}"
        cached_data = await self.redis_client.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            
            # 캐시 데이터가 너무 오래되지 않았는지 확인
            cache_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01T00:00:00'))
            if (datetime.now() - cache_time).total_seconds() < 3600:  # 1시간 이내
                logger.info(f"Using cached fallback data for {data_type}")
                return data.get('data')
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting cached fallback data: {e}")
        return None

async def _get_default_fallback_data(self, data_type: str) -> Optional[Any]:
    """기본 폴백 데이터 반환"""
    try:
        # 데이터 타입별 기본값
        default_data = {
            'market_data': {
                'status': 'fallback',
                'message': '시장 데이터 서비스 일시 중단',
                'data': []
            },
            'economic_data': {
                'status': 'fallback',
                'message': '경제 지표 서비스 일시 중단',
                'indicators': {}
            },
            'news_data': {
                'status': 'fallback',
                'message': '뉴스 서비스 일시 중단',
                'articles': []
            },
            'global_data': {
                'status': 'fallback',
                'message': '글로벌 데이터 서비스 일시 중단',
                'data': {}
            }
        }
        
        return default_data.get(data_type)
        
    except Exception as e:
        logger.error(f"Error getting default fallback data: {e}")
        return None

async def _send_failure_alert(self, endpoint
```