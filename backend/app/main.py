“””
Master’s Eye - FastAPI Main Application
4대 거장 융합 주식 AI 포트폴리오 앱
“””

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
from loguru import logger

from app.config import settings
from app.core.database import init_db
from app.core.redis import init_redis
from app.core.logging import setup_logging
from app.api.v1.router import api_router
from app.api.websocket import WebSocketManager
from app.core.exceptions import (
APIException,
api_exception_handler,
validation_exception_handler,
http_exception_handler
)
from app.tasks.data_collection import start_data_collection

# WebSocket Manager 인스턴스

websocket_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
“”“애플리케이션 시작/종료 시 실행되는 라이프사이클 관리”””
# Startup
logger.info(”=== Master’s Eye 시작 ===”)

```
# 데이터베이스 초기화
await init_db()
logger.info("✅ 데이터베이스 연결 완료")

# Redis 초기화
await init_redis()
logger.info("✅ Redis 연결 완료")

# 백그라운드 데이터 수집 시작
if settings.ENVIRONMENT != "test":
    asyncio.create_task(start_data_collection())
    logger.info("✅ 백그라운드 데이터 수집 시작")

logger.info("🚀 Master's Eye 시스템 준비 완료!")

yield

# Shutdown
logger.info("=== Master's Eye 종료 ===")
await websocket_manager.disconnect_all()
logger.info("✅ 모든 WebSocket 연결 종료")
```

# FastAPI 애플리케이션 생성

app = FastAPI(
title=“Master’s Eye API”,
description=“4대 거장 융합 주식 AI 포트폴리오 시스템”,
version=“1.0.0”,
docs_url=”/docs” if settings.DEBUG else None,
redoc_url=”/redoc” if settings.DEBUG else None,
lifespan=lifespan
)

# 로깅 시스템 초기화

setup_logging()

# 미들웨어 설정

app.add_middleware(
CORSMiddleware,
allow_origins=settings.ALLOWED_ORIGINS,
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 예외 핸들러 등록

app.add_exception_handler(APIException, api_exception_handler)
app.add_exception_handler(422, validation_exception_handler)
app.add_exception_handler(Exception, http_exception_handler)

# API 라우터 등록

app.include_router(api_router, prefix=”/api/v1”)

# 정적 파일 서빙 (프론트엔드)

if settings.SERVE_STATIC:
app.mount(”/static”, StaticFiles(directory=“static”), name=“static”)

@app.get(”/”, response_class=HTMLResponse)
async def root():
“”“루트 엔드포인트 - 간단한 상태 페이지”””
return “””
<html>
<head>
<title>Master’s Eye API</title>
<style>
body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
.logo { font-size: 48px; color: #2E8B57; margin-bottom: 20px; }
.subtitle { font-size: 18px; color: #666; margin-bottom: 30px; }
.status { font-size: 24px; color: #28a745; }
.features { text-align: left; max-width: 600px; margin: 0 auto; }
.feature { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }
</style>
</head>
<body>
<div class="logo">📊 Master’s Eye</div>
<div class="subtitle">4대 거장 융합 주식 AI 포트폴리오 시스템</div>
<div class="status">🚀 시스템 정상 운영 중</div>

```
        <div class="features">
            <div class="feature">💡 워렌 버핏 - 가치 투자 철학</div>
            <div class="feature">🌊 레이 달리오 - 거시경제 & All Weather</div>
            <div class="feature">🔬 리처드 파인만 - 과학적 사고 & 불확실성</div>
            <div class="feature">📐 짐 사이먼스 - 퀀트 & 패턴 인식</div>
        </div>
        
        <div style="margin-top: 40px;">
            <a href="/docs" style="margin: 10px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">API 문서</a>
            <a href="/health" style="margin: 10px; padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px;">헬스 체크</a>
        </div>
    </body>
</html>
"""
```

@app.get(”/health”)
async def health_check():
“”“헬스 체크 엔드포인트”””
from app.core.database import get_db_session
from app.core.redis import get_redis

```
try:
    # 데이터베이스 연결 확인
    async with get_db_session() as db:
        await db.execute("SELECT 1")
    
    # Redis 연결 확인  
    redis = await get_redis()
    await redis.ping()
    
    return {
        "status": "healthy",
        "timestamp": settings.get_current_time(),
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "services": {
            "database": "connected",
            "redis": "connected",
            "websocket": f"{len(websocket_manager.active_connections)} connections"
        }
    }
except Exception as e:
    logger.error(f"Health check failed: {e}")
    return {
        "status": "unhealthy",
        "timestamp": settings.get_current_time(),
        "error": str(e)
    }
```

@app.websocket(”/ws”)
async def websocket_endpoint(websocket: WebSocket):
“”“메인 WebSocket 엔드포인트”””
await websocket_manager.connect(websocket)

```
try:
    while True:
        # 클라이언트로부터 메시지 수신
        data = await websocket.receive_json()
        
        # 메시지 타입에 따른 처리
        message_type = data.get("type")
        
        if message_type == "subscribe_market":
            # 실시간 시장 데이터 구독
            symbols = data.get("symbols", [])
            await websocket_manager.subscribe_market_data(websocket, symbols)
            
        elif message_type == "subscribe_portfolio":
            # 포트폴리오 업데이트 구독
            portfolio_id = data.get("portfolio_id")
            await websocket_manager.subscribe_portfolio(websocket, portfolio_id)
            
        elif message_type == "unsubscribe":
            # 구독 해제
            await websocket_manager.unsubscribe(websocket, data.get("channel"))
            
        elif message_type == "ping":
            # 연결 유지 핑
            await websocket.send_json({"type": "pong", "timestamp": settings.get_current_time()})
            
        else:
            # 알 수 없는 메시지 타입
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            })
            
except WebSocketDisconnect:
    await websocket_manager.disconnect(websocket)
except Exception as e:
    logger.error(f"WebSocket error: {e}")
    await websocket_manager.disconnect(websocket)
```

@app.websocket(”/ws/chat”)
async def ai_chat_websocket(websocket: WebSocket):
“”“AI 채팅 전용 WebSocket 엔드포인트”””
await websocket_manager.connect(websocket, connection_type=“chat”)

```
try:
    while True:
        data = await websocket.receive_json()
        
        # AI 채팅 처리
        from app.services.ai_chat import process_chat_message
        
        user_message = data.get("message", "")
        user_id = data.get("user_id")
        portfolio_id = data.get("portfolio_id")
        
        # AI 응답 생성 (스트리밍)
        async for response_chunk in process_chat_message(
            message=user_message,
            user_id=user_id,
            portfolio_id=portfolio_id
        ):
            await websocket.send_json({
                "type": "chat_response",
                "chunk": response_chunk,
                "timestamp": settings.get_current_time()
            })
        
        # 응답 완료 신호
        await websocket.send_json({
            "type": "chat_complete",
            "timestamp": settings.get_current_time()
        })
        
except WebSocketDisconnect:
    await websocket_manager.disconnect(websocket)
except Exception as e:
    logger.error(f"AI Chat WebSocket error: {e}")
    await websocket_manager.disconnect(websocket)
```

# 개발 서버 실행

if **name** == “**main**”:
uvicorn.run(
“app.main:app”,
host=“0.0.0.0”,
port=8000,
reload=settings.DEBUG,
log_level=“info”,
access_log=True
)