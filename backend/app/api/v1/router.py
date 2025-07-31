# backend/app/api/v1/router.py

“””
API v1 메인 라우터

- 모든 엔드포인트 통합
- 라우터 등록 및 관리
  “””

from fastapi import APIRouter
from .stocks import router as stocks_router
from .market import router as market_router
from .trading import router as trading_router
from .analysis import analysis_router
from .realtime import router as realtime_router  
from .system import router as system_router

# API v1 메인 라우터

api_router = APIRouter(prefix=”/api/v1”)

# 각 모듈 라우터 등록

api_router.include_router(stocks_router)      # /api/v1/stocks
api_router.include_router(market_router)      # /api/v1/market  
api_router.include_router(trading_router)     # /api/v1/trading
api_router.include_router(analysis_router)    # /api/v1/analysis
api_router.include_router(realtime_router)    # /api/v1/realtime
api_router.include_router(system_router)      # /api/v1/system

# 루트 엔드포인트

@api_router.get(”/”)
async def api_root():
“”“API v1 루트”””
return {
“message”: “Master’s Eye API v1”,
“version”: “1.0.0”,
“endpoints”: {
“stocks”: “/api/v1/stocks - 주식 정보 조회”,
“market”: “/api/v1/market - 시장 정보 조회”,
“trading”: “/api/v1/trading - 거래 및 매매동향”,
“analysis”: “/api/v1/analysis - 기술적 분석”,
“realtime”: “/api/v1/realtime - 실시간 데이터”,
“system”: “/api/v1/system - 시스템 상태”
},
“docs”: “/docs”,
“redoc”: “/redoc”
}

# backend/app/api/**init**.py

“””
API 패키지 초기화
“””

from .v1.router import api_router

**all** = [“api_router”]

# backend/app/main.py (수정)

“””
FastAPI 메인 애플리케이션
“””

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime

from .api import api_router
from .core.logging import initialize_logging, get_main_logger
from .core.redis import close_redis_client

# 로깅 초기화

initialize_logging()
logger = get_main_logger()

# FastAPI 앱 생성

app = FastAPI(
title=“Master’s Eye API”,
description=“4대 거장 융합 주식 포트폴리오 시스템”,
version=“1.0.0”,
docs_url=”/docs”,
redoc_url=”/redoc”
)

# CORS 설정

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],  # 개발 환경, 프로덕션에서는 제한 필요
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

# 요청 로깅 미들웨어

@app.middleware(“http”)
async def log_requests(request: Request, call_next):
start_time = time.time()

```
# 요청 로깅
logger.info(f"요청 시작: {request.method} {request.url}")

try:
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # 응답 로깅
    logger.info(
        f"요청 완료: {request.method} {request.url} - "
        f"상태: {response.status_code}, 처리시간: {process_time:.3f}초"
    )
    
    # 응답 헤더에 처리 시간 추가
    response.headers["X-Process-Time"] = str(process_time)
    
    return response
    
except Exception as e:
    process_time = time.time() - start_time
    logger.error(
        f"요청 오류: {request.method} {request.url} - "
        f"오류: {str(e)}, 처리시간: {process_time:.3f}초"
    )
    raise
```

# 전역 예외 처리

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
logger.error(f”전역 예외 발생: {request.method} {request.url} - {str(exc)}”)

```
return JSONResponse(
    status_code=500,
    content={
        "error": "내부 서버 오류",
        "message": "요청 처리 중 오류가 발생했습니다",
        "timestamp": datetime.now().isoformat()
    }
)
```

# API 라우터 등록

app.include_router(api_router)

# 루트 엔드포인트

@app.get(”/”)
async def root():
“”“루트 엔드포인트”””
return {
“message”: “Master’s Eye - 4대 거장 융합 주식 포트폴리오 시스템”,
“version”: “1.0.0”,
“status”: “running”,
“timestamp”: datetime.now().isoformat(),
“api_docs”: “/docs”,
“api_base”: “/api/v1”
}

# 헬스 체크 (간단 버전)

@app.get(”/health”)
async def simple_health_check():
“”“간단한 헬스 체크”””
return {
“status”: “healthy”,
“timestamp”: datetime.now().isoformat()
}

# 애플리케이션 시작 이벤트

@app.on_event(“startup”)
async def startup_event():
“”“애플리케이션 시작 시 실행”””
logger.info(“Master’s Eye API 서버 시작”)
logger.info(“모든 엔드포인트가 준비되었습니다”)

# 애플리케이션 종료 이벤트

@app.on_event(“shutdown”)
async def shutdown_event():
“”“애플리케이션 종료 시 실행”””
logger.info(“Master’s Eye API 서버 종료 중…”)

```
# Redis 연결 종료
await close_redis_client()

logger.info("Master's Eye API 서버 종료 완료")
```

if **name** == “**main**”:
import uvicorn

```
uvicorn.run(
    "app.main:app",
    host="0.0.0.0",
    port=8000,
    reload=True,
    log_level="info"
)
```