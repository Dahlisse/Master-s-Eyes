# backend/app/api/v1/system.py

“””
시스템 상태 및 관리 REST API 엔드포인트

- 시스템 헬스 체크
- API 상태 모니터링
- 성능 지표 조회
  “””

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import psutil
import sys
import os

from …data.collectors.kis_api import KISApiClient, KISConfig
from …core.config import get_kis_config
from …core.logging import get_main_logger, get_performance_logger
from …core.redis import get_redis_client

logger = get_main_logger()
perf_logger = get_performance_logger()

router = APIRouter(prefix=”/system”, tags=[“system”])

@router.get(”/health”)
async def health_check():
“””
시스템 종합 헬스 체크

```
- API 서버 상태
- 데이터베이스 연결
- Redis 연결  
- KIS API 연결
"""
try:
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # 1. API 서버 기본 상태
    health_status["checks"]["api_server"] = {
        "status": "healthy",
        "uptime": _get_uptime(),
        "python_version": sys.version.split()[0],
        "memory_usage": _get_memory_usage()
    }
    
    # 2. Redis 연결 체크
    try:
        redis_client = await get_redis_client()
        await redis_client.client.ping()
        health_status["checks"]["redis"] = {
            "status": "healthy",
            "connected": True
        }
    except Exception as e:
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # 3. KIS API 연결 체크
    try:
        config = get_kis_config()
        client = KISApiClient(config)
        await client.initialize()
        
        # 간단한 API 호출로 연결 테스트
        await client.get_market_status()
        await client.close()
        
        health_status["checks"]["kis_api"] = {
            "status": "healthy",
            "connected": True,
            "is_mock": config.is_mock
        }
    except Exception as e:
        health_status["checks"]["kis_api"] = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # 4. 실시간 수집기 상태 체크
    try:
        from ...data.collectors.realtime_collector import get_realtime_collector
        collector = await get_realtime_collector()
        collector_stats = collector.get_stats()
        
        health_status["checks"]["realtime_collector"] = {
            "status": "healthy" if collector_stats["is_running"] else "stopped",
            "is_running": collector_stats["is_running"],
            "active_connections": collector_stats["active_connections"],
            "subscribed_tickers": collector_stats["subscribed_tickers"]
        }
    except Exception as e:
        health_status["checks"]["realtime_collector"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return health_status
    
except Exception as e:
    logger.error(f"헬스 체크 실패: {e}")
    return {
        "status": "unhealthy",
        "error": str(e),
        "timestamp": datetime.now().isoformat()
    }
```

@router.get(”/status”)
async def get_system_status():
“””
시스템 상태 상세 조회

```
- 서버 리소스 사용량
- 메모리 사용량
- CPU 사용률
- 디스크 사용량
"""
try:
    # CPU 사용률
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 메모리 사용량
    memory = psutil.virtual_memory()
    
    # 디스크 사용량
    disk = psutil.disk_usage('/')
    
    # 네트워크 통계
    network = psutil.net_io_counters()
    
    # 프로세스 정보
    process = psutil.Process()
    process_memory = process.memory_info()
    
    system_status = {
        "server": {
            "hostname": os.uname().nodename,
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "uptime": _get_uptime()
        },
        "resources": {
            "cpu": {
                "usage_percent": cpu_percent,
                "core_count": psutil.cpu_count(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "usage_percent": round((disk.used / disk.total) * 100, 2)
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        },
        "process": {
            "pid": process.pid,
            "memory_mb": round(process_memory.rss / (1024**2), 2),
            "memory_percent": round(process.memory_percent(), 2),
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "open_files": len(process.open_files())
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return system_status
    
except Exception as e:
    logger.error(f"시스템 상태 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"시스템 상태 조회 실패: {str(e)}")
```

@router.get(”/performance”)
async def get_performance_metrics():
“””
성능 지표 조회

```
- API 응답 시간
- 데이터 처리 성능
- 에러율
"""
try:
    # 기본 성능 지표 (실제로는 메트릭 수집 시스템에서 가져와야 함)
    performance_metrics = {
        "api_metrics": {
            "average_response_time_ms": 150,  # 실제 메트릭으로 교체 필요
            "requests_per_minute": 120,
            "error_rate_percent": 0.5,
            "active_connections": 25
        },
        "data_processing": {
            "realtime_messages_per_second": 450,
            "batch_processing_time_ms": 2500,
            "cache_hit_rate_percent": 85.5,
            "queue_size": 45
        },
        "memory_performance": {
            "heap_usage_mb": round(psutil.Process().memory_info().rss / (1024**2), 2),
            "gc_collections": "N/A",  # Python GC 정보
            "memory_leaks_detected": False
        },
        "database_performance": {
            "connection_pool_usage": 15,
            "query_average_time_ms": 25,
            "slow_queries_count": 2
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return performance_metrics
    
except Exception as e:
    logger.error(f"성능 지표 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"성능 지표 조회 실패: {str(e)}")
```

@router.get(”/logs”)
async def get_recent_logs(
level: str = Query(“INFO”, description=“로그 레벨 (DEBUG, INFO, WARNING, ERROR)”),
count: int = Query(100, ge=1, le=1000, description=“조회할 로그 수”),
module: Optional[str] = Query(None, description=“모듈 필터”)
):
“””
최근 로그 조회

```
- **level**: 로그 레벨 필터
- **count**: 조회할 로그 수
- **module**: 특정 모듈 로그만 조회
"""
try:
    # 실제로는 로그 파일에서 읽어오거나 로그 수집 시스템에서 가져와야 함
    # 여기서는 예시 데이터 반환
    
    sample_logs = [
        {
            "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
            "level": "INFO" if i % 3 == 0 else ("WARNING" if i % 5 == 0 else "ERROR"),
            "module": f"module_{i % 5}",
            "message": f"Sample log message {i}",
            "extra": {"request_id": f"req_{i}"}
        }
        for i in range(count)
    ]
    
    # 레벨 필터 적용
    if level != "ALL":
        level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        min_priority = level_priority.get(level.upper(), 1)
        sample_logs = [
            log for log in sample_logs 
            if level_priority.get(log["level"], 1) >= min_priority
        ]
    
    # 모듈 필터 적용
    if module:
        sample_logs = [log for log in sample_logs if module in log["module"]]
    
    return {
        "logs": sample_logs[:count],
        "total_count": len(sample_logs),
        "filters": {
            "level": level,
            "module": module,
            "count": count
        },
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"로그 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"로그 조회 실패: {str(e)}")
```

@router.get(”/config”)
async def get_system_config():
“””
시스템 설정 정보 조회 (민감 정보 제외)

```
- 환경 변수 (마스킹)
- 설정 값
- 버전 정보
"""
try:
    # 환경 변수 (민감 정보 마스킹)
    env_vars = {}
    for key, value in os.environ.items():
        if any(sensitive in key.upper() for sensitive in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']):
            env_vars[key] = "*" * 8
        else:
            env_vars[key] = value
    
    config_info = {
        "application": {
            "name": "Master's Eye API",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true"
        },
        "api_settings": {
            "kis_api_mock": os.getenv("KIS_IS_MOCK", "true"),
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            "log_level": os.getenv("LOG_LEVEL", "INFO")
        },
        "realtime_settings": {
            "max_connections": os.getenv("REALTIME_MAX_CONNECTIONS", "5"),
            "batch_size": os.getenv("REALTIME_BATCH_SIZE", "100"),
            "flush_interval": os.getenv("REALTIME_FLUSH_INTERVAL", "5")
        },
        "environment_variables": env_vars,
        "timestamp": datetime.now().isoformat()
    }
    
    return config_info
    
except Exception as e:
    logger.error(f"설정 정보 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"설정 정보 조회 실패: {str(e)}")
```

@router.post(”/maintenance”)
async def maintenance_mode(
enable: bool = Query(…, description=“유지보수 모드 활성화 여부”),
message: str = Query(“시스템 유지보수 중입니다”, description=“유지보수 메시지”)
):
“””
유지보수 모드 설정

```
- **enable**: 유지보수 모드 활성화/비활성화
- **message**: 사용자에게 표시할 메시지
"""
try:
    # 실제로는 Redis나 설정 파일에 저장
    redis_client = await get_redis_client()
    
    if enable:
        maintenance_data = {
            "enabled": True,
            "message": message,
            "started_at": datetime.now().isoformat()
        }
        await redis_client.set("maintenance_mode", str(maintenance_data), expire=3600)
        logger.warning(f"유지보수 모드 활성화: {message}")
    else:
        await redis_client.delete("maintenance_mode")
        logger.info("유지보수 모드 비활성화")
    
    return {
        "maintenance_mode": enable,
        "message": message if enable else "유지보수 모드가 비활성화되었습니다",
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"유지보수 모드 설정 실패: {e}")
    raise HTTPException(status_code=500, detail=f"유지보수 모드 설정 실패: {str(e)}")
```

@router.get(”/version”)
async def get_version_info():
“””
버전 정보 조회

```
- 애플리케이션 버전
- 의존성 버전
- 빌드 정보
"""
try:
    import pkg_resources
    
    # 주요 의존성 버전 조회
    dependencies = {}
    major_packages = ['fastapi', 'aiohttp', 'websockets', 'sqlalchemy', 'redis', 'pandas']
    
    for package in major_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            dependencies[package] = version
        except pkg_resources.DistributionNotFound:
            dependencies[package] = "Not installed"
    
    version_info = {
        "application": {
            "name": "Master's Eye API",
            "version": "1.0.0",
            "build_date": "2024-01-01",  # 실제 빌드 시 설정
            "git_commit": "abc1234",     # 실제 Git 커밋 해시
            "environment": os.getenv("ENVIRONMENT", "development")
        },
        "runtime": {
            "python_version": sys.version,
            "platform": sys.platform,
            "architecture": os.uname().machine if hasattr(os, 'uname') else "unknown"
        },
        "dependencies": dependencies,
        "timestamp": datetime.now().isoformat()
    }
    
    return version_info
    
except Exception as e:
    logger.error(f"버전 정보 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"버전 정보 조회 실패: {str(e)}")
```

# 유틸리티 함수들

def _get_uptime() -> str:
“”“서버 업타임 반환”””
try:
uptime_seconds = time.time() - psutil.boot_time()
uptime_timedelta = timedelta(seconds=uptime_seconds)

```
    days = uptime_timedelta.days
    hours, remainder = divmod(uptime_timedelta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    return f"{days}일 {hours}시간 {minutes}분"
except:
    return "Unknown"
```

def _get_memory_usage() -> Dict[str, float]:
“”“메모리 사용량 반환”””
try:
process = psutil.Process()
memory_info = process.memory_info()

```
    return {
        "rss_mb": round(memory_info.rss / (1024**2), 2),
        "vms_mb": round(memory_info.vms / (1024**2), 2),
        "percent": round(process.memory_percent(), 2)
    }
except:
    return {"rss_mb": 0, "vms_mb": 0, "percent": 0}
```

import time