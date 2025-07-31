# backend/app/core/logging.py

“””
로깅 설정 시스템

- 구조화된 로깅
- 파일 및 콘솔 출력
- 로그 로테이션
- 성능 로깅
  “””

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path

class JSONFormatter(logging.Formatter):
“”“JSON 형식 로그 포맷터”””

```
def format(self, record: logging.LogRecord) -> str:
    log_entry = {
        'timestamp': datetime.fromtimestamp(record.created).isoformat(),
        'level': record.levelname,
        'logger': record.name,
        'message': record.getMessage(),
        'module': record.module,
        'function': record.funcName,
        'line': record.lineno,
    }
    
    # 예외 정보 추가
    if record.exc_info:
        log_entry['exception'] = self.formatException(record.exc_info)
        
    # 추가 필드 포함
    if hasattr(record, 'extra_fields'):
        log_entry.update(record.extra_fields)
        
    return json.dumps(log_entry, ensure_ascii=False)
```

class ContextualFormatter(logging.Formatter):
“”“컨텍스트 정보가 포함된 포맷터”””

```
def format(self, record: logging.LogRecord) -> str:
    # 기본 포맷
    base_format = super().format(record)
    
    # 컨텍스트 정보 추가
    context_info = []
    
    if hasattr(record, 'ticker'):
        context_info.append(f"ticker={record.ticker}")
    if hasattr(record, 'user_id'):
        context_info.append(f"user_id={record.user_id}")
    if hasattr(record, 'request_id'):
        context_info.append(f"request_id={record.request_id}")
        
    if context_info:
        base_format += f" [{', '.join(context_info)}]"
        
    return base_format
```

def setup_logger(
name: str,
level: int = logging.INFO,
console_output: bool = True,
file_output: bool = True,
json_format: bool = False
) -> logging.Logger:
“”“로거 설정”””

```
logger = logging.getLogger(name)

# 이미 설정된 로거는 재설정하지 않음
if logger.handlers:
    return logger
    
logger.setLevel(level)
logger.propagate = False

# 포맷터 설정
if json_format:
    formatter = JSONFormatter()
else:
    formatter = ContextualFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# 콘솔 핸들러
if console_output:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# 파일 핸들러
if file_output:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 일반 로그 파일 (로테이션)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"{name}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 에러 로그 파일 (ERROR 이상만)
    if level <= logging.ERROR:
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{name}_error.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

return logger
```

def setup_application_logging():
“”“애플리케이션 전체 로깅 설정”””

```
# 로그 레벨 설정
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# JSON 로깅 여부
json_logging = os.getenv("JSON_LOGGING", "false").lower() == "true"

# 주요 로거들 설정
loggers = [
    "masters_eye",           # 메인 애플리케이션
    "kis_api",              # KIS API
    "realtime_collector",   # 실시간 수집기
    "data_processor",       # 데이터 처리
    "portfolio_engine",     # 포트폴리오 엔진
    "masters_algorithm",    # 4대 거장 알고리즘
]

for logger_name in loggers:
    setup_logger(
        logger_name,
        level=log_level,
        json_format=json_logging
    )

# 외부 라이브러리 로깅 레벨 조정
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)
```

class PerformanceLogger:
“”“성능 로깅 클래스”””

```
def __init__(self, logger: logging.Logger):
    self.logger = logger
    
def log_execution_time(self, func_name: str, execution_time: float, **kwargs):
    """실행 시간 로깅"""
    self.logger.info(
        f"함수 실행 시간: {func_name} - {execution_time:.3f}초",
        extra={'extra_fields': {
            'function': func_name,
            'execution_time': execution_time,
            'performance_metric': True,
            **kwargs
        }}
    )
    
def log_api_call(self, endpoint: str, method: str, response_time: float, status_code: int):
    """API 호출 로깅"""
    self.logger.info(
        f"API 호출: {method} {endpoint} - {response_time:.3f}초 (상태: {status_code})",
        extra={'extra_fields': {
            'endpoint': endpoint,
            'method': method,
            'response_time': response_time,
            'status_code': status_code,
            'api_metric': True
        }}
    )
    
def log_data_processing(self, data_type: str, record_count: int, processing_time: float):
    """데이터 처리 로깅"""
    self.logger.info(
        f"데이터 처리: {data_type} - {record_count}건, {processing_time:.3f}초",
        extra={'extra_fields': {
            'data_type': data_type,
            'record_count': record_count,
            'processing_time': processing_time,
            'data_metric': True
        }}
    )
```

class BusinessLogger:
“”“비즈니스 로직 로깅 클래스”””

```
def __init__(self, logger: logging.Logger):
    self.logger = logger
    
def log_portfolio_creation(self, user_id: int, portfolio_id: int, strategy: str, **kwargs):
    """포트폴리오 생성 로깅"""
    self.logger.info(
        f"포트폴리오 생성: 사용자={user_id}, 전략={strategy}",
        extra={'extra_fields': {
            'user_id': user_id,
            'portfolio_id': portfolio_id,
            'strategy': strategy,
            'business_event': 'portfolio_creation',
            **kwargs
        }}
    )
    
def log_trade_execution(self, ticker: str, side: str, quantity: int, price: int, **kwargs):
    """거래 실행 로깅"""
    self.logger.info(
        f"거래 실행: {ticker} {side} {quantity}주 @ {price:,}원",
        extra={'extra_fields': {
            'ticker': ticker,
            'side': side,
            'quantity': quantity,
            'price': price,
            'business_event': 'trade_execution',
            **kwargs
        }}
    )
    
def log_rebalancing(self, portfolio_id: int, changes: Dict[str, Any], **kwargs):
    """리밸런싱 로깅"""
    self.logger.info(
        f"포트폴리오 리밸런싱: {portfolio_id}",
        extra={'extra_fields': {
            'portfolio_id': portfolio_id,
            'changes': changes,
            'business_event': 'rebalancing',
            **kwargs
        }}
    )
```

class SecurityLogger:
“”“보안 관련 로깅 클래스”””

```
def __init__(self, logger: logging.Logger):
    self.logger = logger
    
def log_authentication_attempt(self, user_id: Optional[int], success: bool, ip_address: str):
    """인증 시도 로깅"""
    level = logging.INFO if success else logging.WARNING
    message = f"인증 {'성공' if success else '실패'}: 사용자={user_id}, IP={ip_address}"
    
    self.logger.log(level, message, extra={'extra_fields': {
        'user_id': user_id,
        'success': success,
        'ip_address': ip_address,
        'security_event': 'authentication_attempt'
    }})
    
def log_api_rate_limit(self, key: str, limit: int, current: int):
    """API 속도 제한 로깅"""
    self.logger.warning(
        f"API 속도 제한 초과: {key} - {current}/{limit}",
        extra={'extra_fields': {
            'rate_limit_key': key,
            'limit': limit,
            'current': current,
            'security_event': 'rate_limit_exceeded'
        }}
    )
```

# 전역 로거 인스턴스들

_main_logger: Optional[logging.Logger] = None
_performance_logger: Optional[PerformanceLogger] = None
_business_logger: Optional[BusinessLogger] = None
_security_logger: Optional[SecurityLogger] = None

def get_main_logger() -> logging.Logger:
“”“메인 로거 반환”””
global _main_logger
if _main_logger is None:
_main_logger = setup_logger(“masters_eye”)
return _main_logger

def get_performance_logger() -> PerformanceLogger:
“”“성능 로거 반환”””
global _performance_logger
if _performance_logger is None:
logger = setup_logger(“performance”)
_performance_logger = PerformanceLogger(logger)
return _performance_logger

def get_business_logger() -> BusinessLogger:
“”“비즈니스 로거 반환”””
global _business_logger
if _business_logger is None:
logger = setup_logger(“business”)
_business_logger = BusinessLogger(logger)
return _business_logger

def get_security_logger() -> SecurityLogger:
“”“보안 로거 반환”””
global _security_logger
if _security_logger is None:
logger = setup_logger(“security”)
_security_logger = SecurityLogger(logger)
return _security_logger

# 데코레이터

def log_execution_time(logger: Optional[logging.Logger] = None):
“”“실행 시간 로깅 데코레이터”””
import functools
import time

```
def decorator(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            log = logger or get_main_logger()
            log.info(f"{func.__name__} 실행 완료: {execution_time:.3f}초")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            log = logger or get_main_logger()
            log.error(f"{func.__name__} 실행 실패: {execution_time:.3f}초, 오류: {e}")
            raise
            
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            log = logger or get_main_logger()
            log.info(f"{func.__name__} 실행 완료: {execution_time:.3f}초")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            log = logger or get_main_logger()
            log.error(f"{func.__name__} 실행 실패: {execution_time:.3f}초, 오류: {e}")
            raise
            
    # 비동기 함수인지 확인
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
        
return decorator
```

# 초기화 함수

def initialize_logging():
“”“로깅 시스템 초기화”””
try:
setup_application_logging()
logger = get_main_logger()
logger.info(“로깅 시스템 초기화 완료”)
except Exception as e:
print(f”로깅 초기화 실패: {e}”)
raise