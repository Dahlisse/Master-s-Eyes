# backend/app/api/v1/scheduler.py

“””
데이터 수집 스케줄러 관리 API

- 스케줄러 시작/정지
- 작업 관리 (추가/제거/실행)
- 작업 상태 모니터링
  “””

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime

from …scheduler.data_scheduler import get_data_scheduler, start_data_scheduling, stop_data_scheduling, ScheduleJob, ScheduleType
from …core.logging import get_main_logger

logger = get_main_logger()

router = APIRouter(prefix=”/scheduler”, tags=[“scheduler”])

@router.post(”/start”)
async def start_scheduler():
“”“데이터 수집 스케줄러 시작”””
try:
await start_data_scheduling()
return {
“status”: “started”,
“message”: “데이터 수집 스케줄러가 시작되었습니다”,
“timestamp”: datetime.now().isoformat()
}
except Exception as e:
logger.error(f”스케줄러 시작 실패: {e}”)
raise HTTPException(status_code=500, detail=f”스케줄러 시작 실패: {str(e)}”)

@router.post(”/stop”)
async def stop_scheduler():
“”“데이터 수집 스케줄러 정지”””
try:
await stop_data_scheduling()
return {
“status”: “stopped”,
“message”: “데이터 수집 스케줄러가 정지되었습니다”,
“timestamp”: datetime.now().isoformat()
}
except Exception as e:
logger.error(f”스케줄러 정지 실패: {e}”)
raise HTTPException(status_code=500, detail=f”스케줄러 정지 실패: {str(e)}”)

@router.get(”/status”)
async def get_scheduler_status():
“”“스케줄러 상태 조회”””
try:
scheduler = await get_data_scheduler()
status = scheduler.get_job_status()
return {
“scheduler_status”: status,
“timestamp”: datetime.now().isoformat()
}
except Exception as e:
logger.error(f”스케줄러 상태 조회 실패: {e}”)
raise HTTPException(status_code=500, detail=f”상태 조회 실패: {str(e)}”)

@router.post(”/jobs/{job_id}/execute”)
async def execute_job_now(job_id: str):
“”“작업 즉시 실행”””
try:
scheduler = await get_data_scheduler()
success = await scheduler.execute_job_now(job_id)

```
    if success:
        return {
            "status": "executed",
            "job_id": job_id,
            "message": "작업이 즉시 실행되었습니다",
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {job_id}")
        
except HTTPException:
    raise
except Exception as e:
    logger.error(f"작업 즉시 실행 실패 ({job_id}): {e}")
    raise HTTPException(status_code=500, detail=f"작업 실행 실패: {str(e)}")
```

@router.post(”/jobs/{job_id}/pause”)
async def pause_job(job_id: str):
“”“작업 일시 정지”””
try:
scheduler = await get_data_scheduler()
success = await scheduler.pause_job(job_id)

```
    if success:
        return {
            "status": "paused",
            "job_id": job_id,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {job_id}")
        
except HTTPException:
    raise
except Exception as e:
    logger.error(f"작업 일시 정지 실패 ({job_id}): {e}")
    raise HTTPException(status_code=500, detail=f"작업 일시 정지 실패: {str(e)}")
```

@router.post(”/jobs/{job_id}/resume”)
async def resume_job(job_id: str):
“”“작업 재개”””
try:
scheduler = await get_data_scheduler()
success = await scheduler.resume_job(job_id)

```
    if success:
        return {
            "status": "resumed",
            "job_id": job_id,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {job_id}")
        
except HTTPException:
    raise
except Exception as e:
    logger.error(f"작업 재개 실패 ({job_id}): {e}")
    raise HTTPException(status_code=500, detail=f"작업 재개 실패: {str(e)}")
```

@router.get(”/jobs”)
async def list_jobs():
“”“등록된 모든 작업 목록 조회”””
try:
scheduler = await get_data_scheduler()
status = scheduler.get_job_status()

```
    return {
        "jobs": status["jobs"],
        "summary": {
            "total_jobs": status["total_jobs"],
            "active_jobs": status["active_jobs"],
            "is_running": status["is_running"]
        },
        "timestamp": datetime.now().isoformat()
    }
except Exception as e:
    logger.error(f"작업 목록 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"작업 목록 조회 실패: {str(e)}")
```

@router.get(”/jobs/{job_id}”)
async def get_job_detail(job_id: str):
“”“특정 작업 상세 정보 조회”””
try:
scheduler = await get_data_scheduler()
status = scheduler.get_job_status()

```
    if job_id not in status["jobs"]:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {job_id}")
    
    return {
        "job": status["jobs"][job_id],
        "timestamp": datetime.now().isoformat()
    }
except HTTPException:
    raise
except Exception as e:
    logger.error(f"작업 상세 조회 실패 ({job_id}): {e}")
    raise HTTPException(status_code=500, detail=f"작업 상세 조회 실패: {str(e)}")
```

# backend/app/main.py (완전 통합 버전)

“””
FastAPI 메인 애플리케이션 - 완전 통합 버전

- 모든 API 엔드포인트 통합
- 실시간 데이터 수집
- 스케줄러 통합
- 시스템 모니터링
  “””

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

from .api.v1.router import api_router
from .api.v1.scheduler import router as scheduler_router
from .core.logging import initialize_logging, get_main_logger
from .core.redis import close_redis_client
from .scheduler.data_scheduler import start_data_scheduling, stop_data_scheduling

# 로깅 초기화

initialize_logging()
logger = get_main_logger()

# 애플리케이션 생명주기 관리

@asynccontextmanager
async def lifespan(app: FastAPI):
“”“애플리케이션 생명주기 관리”””
# 시작 시 실행
logger.info(“🚀 Master’s Eye API 서버 시작”)

```
try:
    # 데이터 수집 스케줄러 시작
    await start_data_scheduling()
    logger.info("📅 데이터 수집 스케줄러 시작 완료")
    
    # 실시간 데이터 수집 시작 (선택적)
    # from .data.collectors.realtime_collector import start_realtime_collection
    # await start_realtime_collection()
    # logger.info("📊 실시간 데이터 수집 시작 완료")
    
except Exception as e:
    logger.error(f"시작 시 오류: {e}")

yield  # 애플리케이션 실행

# 종료 시 실행
logger.info("🛑 Master's Eye API 서버 종료 중...")

try:
    # 스케줄러 정지
    await stop_data_scheduling()
    logger.info("📅 데이터 수집 스케줄러 정지 완료")
    
    # 실시간 수집 정지
    # from .data.collectors.realtime_collector import stop_realtime_collection
    # await stop_realtime_collection()
    # logger.info("📊 실시간 데이터 수집 정지 완료")
    
    # Redis 연결 종료
    await close_redis_client()
    logger.info("💾 Redis 연결 종료 완료")
    
except Exception as e:
    logger.error(f"종료 시 오류: {e}")

logger.info("✅ Master's Eye API 서버 종료 완료")
```

# FastAPI 앱 생성

app = FastAPI(
title=“Master’s Eye API”,
description=”””
🎯 **4대 거장 융합 주식 포트폴리오 시스템**

```
워렌 버핏, 레이 달리오, 리처드 파인만, 짐 사이먼스의 투자 철학을 융합한 
지능형 주식 포트폴리오 관리 시스템입니다.

## 🚀 주요 기능

- **실시간 주식 데이터**: WebSocket 기반 실시간 주가/호가 정보
- **4대 거장 알고리즘**: 각 거장의 투자 철학을 구현한 분석 엔진
- **자동 데이터 수집**: 스케줄러 기반 정기적 데이터 수집
- **포트폴리오 최적화**: AI 기반 포트폴리오 구성 및 리밸런싱
- **백테스팅**: 과거 데이터 기반 전략 검증

## 📊 API 카테고리

- `/stocks`: 주식 정보 조회 (현재가, 호가창, 차트, 재무정보)
- `/market`: 시장 정보 (상태, 지수, 일정, 거래시간)
- `/trading`: 거래 분석 (매매동향, 체결강도)
- `/analysis`: 기술적 분석 (지표, 변동성)
- `/realtime`: 실시간 데이터 (구독, WebSocket)
- `/scheduler`: 데이터 수집 스케줄러 관리
- `/system`: 시스템 상태 및 모니터링
""",
version="1.0.0",
docs_url="/docs",
redoc_url="/redoc",
lifespan=lifespan
```

)

# CORS 설정

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],  # 개발 환경용, 프로덕션에서는 제한 필요
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

# 요청 로깅 미들웨어

@app.middleware(“http”)
async def log_requests(request: Request, call_next):
start_time = time.time()

```
# 요청 로깅 (DEBUG 레벨로)
logger.debug(f"🌐 {request.method} {request.url}")

try:
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # 응답 로깅
    status_emoji = "✅" if response.status_code < 400 else "❌"
    logger.info(
        f"{status_emoji} {request.method} {request.url} - "
        f"{response.status_code} ({process_time:.3f}s)"
    )
    
    # 응답 헤더에 처리 시간 추가
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-API-Version"] = "1.0.0"
    
    return response
    
except Exception as e:
    process_time = time.time() - start_time
    logger.error(
        f"💥 {request.method} {request.url} - "
        f"ERROR: {str(e)} ({process_time:.3f}s)"
    )
    raise
```

# 전역 예외 처리

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
logger.error(f”🚨 전역 예외: {request.method} {request.url} - {str(exc)}”)

```
return JSONResponse(
    status_code=500,
    content={
        "error": "내부 서버 오류",
        "message": "요청 처리 중 오류가 발생했습니다",
        "path": str(request.url),
        "method": request.method,
        "timestamp": datetime.now().isoformat()
    }
)
```

# API 라우터 등록

app.include_router(api_router)                    # /api/v1/*
app.include_router(scheduler_router, prefix=”/api/v1”)  # /api/v1/scheduler

# 루트 엔드포인트

@app.get(”/”)
async def root():
“”“🏠 루트 엔드포인트”””
return {
“🎯 service”: “Master’s Eye - 4대 거장 융합 주식 포트폴리오 시스템”,
“🚀 version”: “1.0.0”,
“📊 status”: “running”,
“⏰ timestamp”: datetime.now().isoformat(),
“📚 api_docs”: “/docs”,
“🔄 redoc”: “/redoc”,
“🎪 api_base”: “/api/v1”,
“🏛️ masters”: [“워렌 버핏”, “레이 달리오”, “리처드 파인만”, “짐 사이먼스”],
“🔗 endpoints”: {
“stocks”: “/api/v1/stocks - 주식 정보”,
“market”: “/api/v1/market - 시장 정보”,
“trading”: “/api/v1/trading - 거래 분석”,
“analysis”: “/api/v1/analysis - 기술적 분석”,
“realtime”: “/api/v1/realtime - 실시간 데이터”,
“scheduler”: “/api/v1/scheduler - 스케줄러 관리”,
“system”: “/api/v1/system - 시스템 상태”
}
}

# 헬스 체크 (간단 버전)

@app.get(”/health”)
async def simple_health_check():
“”“💊 간단한 헬스 체크”””
return {
“status”: “healthy”,
“service”: “Master’s Eye API”,
“version”: “1.0.0”,
“timestamp”: datetime.now().isoformat(),
“uptime”: “계산 필요”,  # 실제로는 업타임 계산
“components”: {
“api”: “✅ healthy”,
“database”: “🔍 check /api/v1/system/health”,
“redis”: “🔍 check /api/v1/system/health”,
“scheduler”: “🔍 check /api/v1/scheduler/status”
}
}

# 개발용 디버그 엔드포인트

@app.get(”/debug/info”)
async def debug_info():
“”“🐛 디버그 정보 (개발용)”””
import sys
import os

```
return {
    "python_version": sys.version,
    "platform": sys.platform,
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
    "current_time": datetime.now().isoformat(),
    "api_endpoints_count": len([route for route in app.routes]),
    "middleware_count": len(app.middleware_stack),
}
```

if **name** == “**main**”:
import uvicorn

```
# 개발 서버 실행
uvicorn.run(
    "app.main:app",
    host="0.0.0.0",
    port=8000,
    reload=True,
    log_level="info",
    access_log=True
)
```