“””
AI 대화 API 엔드포인트 - Master’s Eye
파일 위치: backend/app/api/v1/ai.py

자연어 포트폴리오 조정 및 AI 대화 관련 API
“””

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import asyncio
import json
import logging
from datetime import datetime

from …core.database import get_db
from …core.security import get_current_user
from …services.ai_chat import ai_assistant, PortfolioAdjustment
from …utils.explainable_ai import explanation_service, ExplanationLevel
from …utils.prompts import PromptValidator, UserLevel, PromptContext, MarketCondition
from …models.user import User
from …models.portfolio import Portfolio
from …schemas.response import StandardResponse

logger = logging.getLogger(**name**)

router = APIRouter(prefix=”/ai”, tags=[“AI Assistant”])

# Request/Response 모델들

class ChatMessageRequest(BaseModel):
“”“채팅 메시지 요청”””
message: str = Field(…, min_length=1, max_length=1000, description=“사용자 메시지”)
portfolio_id: Optional[int] = Field(None, description=“포트폴리오 ID”)
context: Optional[Dict[str, Any]] = Field(default_factory=dict, description=“추가 컨텍스트”)
explanation_level: Optional[str] = Field(“detailed”, description=“설명 수준 (simple/detailed/technical)”)

class ChatMessageResponse(BaseModel):
“”“채팅 메시지 응답”””
message: str = Field(…, description=“AI 응답 메시지”)
adjustments: List[Dict[str, Any]] = Field(default_factory=list, description=“포트폴리오 조정 사항”)
context: Dict[str, Any] = Field(default_factory=dict, description=“응답 컨텍스트”)
metadata: Dict[str, Any] = Field(default_factory=dict, description=“메타데이터”)
explanation_id: Optional[str] = Field(None, description=“설명 ID”)

class ApplyAdjustmentsRequest(BaseModel):
“”“조정 적용 요청”””
portfolio_id: int = Field(…, description=“포트폴리오 ID”)
adjustments: List[Dict[str, Any]] = Field(…, description=“적용할 조정 사항”)
confirm: bool = Field(False, description=“최종 확인”)

class ChatHistoryResponse(BaseModel):
“”“채팅 히스토리 응답”””
messages: List[Dict[str, Any]] = Field(…, description=“메시지 목록”)
total_count: int = Field(…, description=“전체 메시지 수”)

class ExplanationRequest(BaseModel):
“”“설명 요청”””
decision_data: Dict[str, Any] = Field(…, description=“의사결정 데이터”)
explanation_level: str = Field(“detailed”, description=“설명 수준”)

# API 엔드포인트들

@router.post(”/chat”, response_model=StandardResponse[ChatMessageResponse])
async def chat_with_ai(
request: ChatMessageRequest,
current_user: User = Depends(get_current_user),
db: Session = Depends(get_db)
):
“””
AI와 대화하여 포트폴리오 조정

```
**사용 예시:**
- "삼성전자를 10% 비중으로 추가해줘"
- "현재 포트폴리오 성과는 어때?"
- "더 공격적인 전략으로 변경해줘"
"""
try:
    # 입력 검증
    validation_result = PromptValidator.validate_user_input(request.message)
    if not validation_result["is_valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"입력 검증 실패: {', '.join(validation_result['issues'])}"
        )
    
    # 사용자 레벨 결정
    user_level = _determine_user_level(current_user)
    explanation_level = ExplanationLevel(request.explanation_level.upper())
    
    # AI 처리
    response = await ai_assistant.process_chat_message(
        user_id=current_user.id,
        message=request.message,
        portfolio_id=request.portfolio_id
    )
    
    # 응답 구성
    chat_response = ChatMessageResponse(
        message=response["message"],
        adjustments=response.get("adjustments", []),
        context=response.get("context", {}),
        metadata=response.get("metadata", {}),
        explanation_id=response.get("explanation_id")
    )
    
    return StandardResponse(
        success=True,
        data=chat_response,
        message="AI 응답이 생성되었습니다."
    )
    
except Exception as e:
    logger.error(f"AI chat error for user {current_user.id}: {e}")
    
    # 사용자 친화적 에러 메시지
    if "timeout" in str(e).lower():
        error_msg = "AI 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
    elif "model" in str(e).lower():
        error_msg = "AI 모델 로딩 중입니다. 잠시 후 다시 시도해주세요."
    else:
        error_msg = "일시적인 오류가 발생했습니다. 다시 시도해주세요."
    
    return StandardResponse(
        success=False,
        data=ChatMessageResponse(
            message=error_msg,
            adjustments=[],
            context={},
            metadata={"error": str(e)}
        ),
        message="AI 응답 생성 실패"
    )
```

@router.post(”/chat/stream”)
async def chat_with_ai_stream(
request: ChatMessageRequest,
current_user: User = Depends(get_current_user)
):
“””
스트리밍 방식 AI 대화

```
실시간으로 AI 응답을 스트리밍하여 더 자연스러운 대화 경험 제공
"""
async def generate_stream():
    try:
        # 스트리밍을 위한 청크 단위 응답 생성
        response = await ai_assistant.process_chat_message(
            user_id=current_user.id,
            message=request.message,
            portfolio_id=request.portfolio_id
        )
        
        # 응답을 청크로 나누어 전송
        message = response["message"]
        words = message.split()
        
        for i in range(0, len(words), 3):  # 3단어씩 청크
            chunk = " ".join(words[i:i+3])
            
            stream_data = {
                "type": "message_chunk",
                "content": chunk,
                "is_complete": i + 3 >= len(words)
            }
            
            yield f"data: {json.dumps(stream_data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.1)  # 자연스러운 속도
        
        # 최종 메타데이터 전송
        final_data = {
            "type": "complete",
            "adjustments": response.get("adjustments", []),
            "context": response.get("context", {}),
            "metadata": response.get("metadata", {})
        }
        
        yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        error_data = {
            "type": "error",
            "message": "스트리밍 중 오류가 발생했습니다.",
            "error": str(e)
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

return StreamingResponse(
    generate_stream(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*"
    }
)
```

@router.post(”/adjustments/apply”, response_model=StandardResponse[Dict[str, Any]])
async def apply_portfolio_adjustments(
request: ApplyAdjustmentsRequest,
background_tasks: BackgroundTasks,
current_user: User = Depends(get_current_user),
db: Session = Depends(get_db)
):
“””
포트폴리오 조정사항 적용

```
AI가 제안한 조정사항을 실제 포트폴리오에 적용
"""
try:
    # 포트폴리오 소유권 확인
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == request.portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=404,
            detail="포트폴리오를 찾을 수 없습니다."
        )
    
    # 조정사항을 PortfolioAdjustment 객체로 변환
    adjustments = []
    for adj_data in request.adjustments:
        adjustment = PortfolioAdjustment(
            action=adj_data.get("action"),
            ticker=adj_data.get("ticker"),
            weight=adj_data.get("weight"),
            reason=adj_data.get("reason"),
            parameters=adj_data.get("parameters")
        )
        adjustments.append(adjustment)
    
    # 조정사항 적용
    result = await ai_assistant.apply_portfolio_adjustments(
        portfolio_id=request.portfolio_id,
        adjustments=adjustments
    )
    
    if result["success"]:
        # 백그라운드에서 성과 분석 실행
        background_tasks.add_task(
            _analyze_adjustment_impact,
            request.portfolio_id,
            adjustments
        )
        
        return StandardResponse(
            success=True,
            data=result,
            message="포트폴리오 조정이 성공적으로 적용되었습니다."
        )
    else:
        return StandardResponse(
            success=False,
            data=result,
            message=result.get("message", "조정 적용 중 오류가 발생했습니다.")
        )
        
except Exception as e:
    logger.error(f"Portfolio adjustment error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

@router.get(”/chat/history”, response_model=StandardResponse[ChatHistoryResponse])
async def get_chat_history(
limit: int = 50,
offset: int = 0,
current_user: User = Depends(get_current_user)
):
“””
채팅 히스토리 조회
“””
try:
# AI 어시스턴트에서 히스토리 조회
history = await ai_assistant.get_chat_history(
user_id=current_user.id,
limit=limit + offset
)

```
    # 페이지네이션 적용
    paginated_history = history[offset:offset + limit]
    
    response = ChatHistoryResponse(
        messages=paginated_history,
        total_count=len(history)
    )
    
    return StandardResponse(
        success=True,
        data=response,
        message="채팅 히스토리를 조회했습니다."
    )
    
except Exception as e:
    logger.error(f"Chat history error: {e}")
    raise HTTPException(status_code=500, detail="히스토리 조회 실패")
```

@router.delete(”/chat/history”)
async def clear_chat_history(
current_user: User = Depends(get_current_user)
):
“””
채팅 히스토리 삭제
“””
try:
success = await ai_assistant.clear_chat_history(current_user.id)

```
    if success:
        return StandardResponse(
            success=True,
            data={"cleared": True},
            message="채팅 히스토리가 삭제되었습니다."
        )
    else:
        return StandardResponse(
            success=False,
            data={"cleared": False},
            message="히스토리 삭제에 실패했습니다."
        )
        
except Exception as e:
    logger.error(f"Clear history error: {e}")
    raise HTTPException(status_code=500, detail="히스토리 삭제 실패")
```

@router.post(”/explain”, response_model=StandardResponse[Dict[str, Any]])
async def explain_decision(
request: ExplanationRequest,
current_user: User = Depends(get_current_user)
):
“””
투자 의사결정 설명 생성

```
AI의 투자 결정에 대한 상세한 설명과 근거 제공
"""
try:
    # 설명 수준 검증
    try:
        explanation_level = ExplanationLevel(request.explanation_level.upper())
    except ValueError:
        explanation_level = ExplanationLevel.DETAILED
    
    # 설명 생성
    result = await explanation_service.explain_realtime_decision(
        decision_data=request.decision_data,
        user_level=explanation_level
    )
    
    return StandardResponse(
        success=result["success"],
        data=result,
        message="의사결정 설명이 생성되었습니다." if result["success"] else "설명 생성 실패"
    )
    
except Exception as e:
    logger.error(f"Decision explanation error: {e}")
    raise HTTPException(status_code=500, detail="설명 생성 실패")
```

@router.get(”/explain/history”)
async def get_explanation_history(
limit: int = 10,
decision_ids: Optional[str] = None,
current_user: User = Depends(get_current_user)
):
“””
의사결정 설명 히스토리 조회
“””
try:
decision_id_list = None
if decision_ids:
decision_id_list = decision_ids.split(”,”)

```
    history = await explanation_service.get_explanation_history(
        decision_ids=decision_id_list,
        limit=limit
    )
    
    return StandardResponse(
        success=True,
        data={"explanations": history},
        message="설명 히스토리를 조회했습니다."
    )
    
except Exception as e:
    logger.error(f"Explanation history error: {e}")
    raise HTTPException(status_code=500, detail="히스토리 조회 실패")
```

@router.get(”/transparency/report/{portfolio_id}”)
async def get_transparency_report(
portfolio_id: int,
period_days: int = 30,
current_user: User = Depends(get_current_user),
db: Session = Depends(get_db)
):
“””
포트폴리오 투명성 리포트

```
최근 의사결정들에 대한 투명성 분석 리포트 제공
"""
try:
    # 포트폴리오 소유권 확인
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="포트폴리오를 찾을 수 없습니다.")
    
    # 투명성 리포트 생성
    summary = await explanation_service.generate_explanation_summary(portfolio_id)
    
    return StandardResponse(
        success=True,
        data={"summary": summary},
        message="투명성 리포트가 생성되었습니다."
    )
    
except Exception as e:
    logger.error(f"Transparency report error: {e}")
    raise HTTPException(status_code=500, detail="리포트 생성 실패")
```

@router.get(”/commands”)
async def get_available_commands():
“””
사용 가능한 AI 명령어 목록

```
사용자가 AI에게 요청할 수 있는 명령어들과 예시 제공
"""
try:
    commands = ai_assistant.get_available_commands()
    
    return StandardResponse(
        success=True,
        data={"commands": commands},
        message="사용 가능한 명령어 목록입니다."
    )
    
except Exception as e:
    logger.error(f"Commands error: {e}")
    raise HTTPException(status_code=500, detail="명령어 조회 실패")
```

@router.post(”/feedback”)
async def submit_ai_feedback(
message_id: str,
rating: int = Field(…, ge=1, le=5, description=“평점 (1-5)”),
feedback: Optional[str] = Field(None, max_length=500, description=“피드백 내용”),
current_user: User = Depends(get_current_user)
):
“””
AI 응답에 대한 피드백 제출

```
사용자의 피드백을 통해 AI 성능 개선에 활용
"""
try:
    # 피드백 저장 (실제 구현에서는 데이터베이스에 저장)
    feedback_data = {
        "message_id": message_id,
        "user_id": current_user.id,
        "rating": rating,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat()
    }
    
    # TODO: 데이터베이스에 피드백 저장
    logger.info(f"AI feedback received: {feedback_data}")
    
    return StandardResponse(
        success=True,
        data={"submitted": True},
        message="피드백이 성공적으로 제출되었습니다."
    )
    
except Exception as e:
    logger.error(f"Feedback submission error: {e}")
    raise HTTPException(status_code=500, detail="피드백 제출 실패")
```

@router.get(”/status”)
async def get_ai_status():
“””
AI 시스템 상태 확인

```
AI 모델 로딩 상태, 응답 시간 등 확인
"""
try:
    # AI 서비스 상태 확인
    start_time = datetime.now()
    
    # 간단한 테스트 메시지로 응답 시간 측정
    test_response = await ai_assistant.ollama.generate_response(
        prompt="안녕하세요",
        temperature=0.1
    )
    
    response_time = (datetime.now() - start_time).total_seconds()
    
    status = {
        "ai_service": "online",
        "model": test_response.get("model", "unknown"),
        "response_time_seconds": response_time,
        "status": "healthy" if response_time < 10 else "slow",
        "last_check": datetime.now().isoformat()
    }
    
    return StandardResponse(
        success=True,
        data=status,
        message="AI 시스템 상태가 확인되었습니다."
    )
    
except Exception as e:
    logger.error(f"AI status check error: {e}")
    
    status = {
        "ai_service": "offline",
        "model": "unknown",
        "response_time_seconds": None,
        "status": "error",
        "error": str(e),
        "last_check": datetime.now().isoformat()
    }
    
    return StandardResponse(
        success=False,
        data=status,
        message="AI 시스템 상태 확인 실패"
    )
```

@router.post(”/portfolio/suggest”)
async def suggest_portfolio_improvements(
portfolio_id: int,
focus_area: Optional[str] = Field(None, description=“개선 집중 영역 (risk/return/diversification)”),
current_user: User = Depends(get_current_user),
db: Session = Depends(get_db)
):
“””
포트폴리오 개선 제안

```
현재 포트폴리오를 분석하여 구체적인 개선 방안 제안
"""
try:
    # 포트폴리오 소유권 확인
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="포트폴리오를 찾을 수 없습니다.")
    
    # 개선 제안 생성을 위한 프롬프트 구성
    improvement_prompt = f"""
```

현재 포트폴리오를 분석하여 구체적인 개선 방안을 제안해주세요.

집중 영역: {focus_area or ‘전반적 개선’}

4대 거장의 관점에서 다음 사항들을 고려해주세요:

1. 리스크 대비 수익률 개선 방안
1. 분산투자 효과 극대화
1. 시장 사이클에 따른 대응 전략
1. 장기 성장 가능성

구체적이고 실행 가능한 조치들을 제안해주세요.
“””

```
    # AI 응답 생성
    response = await ai_assistant.process_chat_message(
        user_id=current_user.id,
        message=improvement_prompt,
        portfolio_id=portfolio_id
    )
    
    return StandardResponse(
        success=True,
        data={
            "suggestions": response["message"],
            "adjustments": response.get("adjustments", []),
            "focus_area": focus_area
        },
        message="포트폴리오 개선 제안이 생성되었습니다."
    )
    
except Exception as e:
    logger.error(f"Portfolio suggestion error: {e}")
    raise HTTPException(status_code=500, detail="개선 제안 생성 실패")
```

# 헬퍼 함수들

def _determine_user_level(user: User) -> UserLevel:
“”“사용자 투자 수준 결정”””
# 사용자 이름 기반으로 레벨 결정 (실제로는 더 정교한 로직 필요)
if user.name == “엄마”:
return UserLevel.BEGINNER
elif user.name == “나”:
return UserLevel.INTERMEDIATE
else:
return UserLevel.INTERMEDIATE

async def _analyze_adjustment_impact(portfolio_id: int, adjustments: List[PortfolioAdjustment]):
“”“조정사항 영향 분석 (백그라운드 작업)”””
try:
# 조정 전후 성과 비교 분석
# 실제 구현에서는 더 정교한 분석 수행
logger.info(f”Analyzing adjustment impact for portfolio {portfolio_id}”)

```
    # 분석 결과를 데이터베이스에 저장하거나 알림 발송
    # TODO: 실제 분석 로직 구현
    
except Exception as e:
    logger.error(f"Background analysis error: {e}")
```

# 웹소켓 엔드포인트 (실시간 대화)

@router.websocket(”/chat/ws”)
async def websocket_chat(websocket, current_user: User = Depends(get_current_user)):
“””
웹소켓 기반 실시간 AI 대화

```
더 자연스러운 실시간 대화 경험 제공
"""
await websocket.accept()

try:
    while True:
        # 클라이언트로부터 메시지 수신
        data = await websocket.receive_text()
        message_data = json.loads(data)
        
        # 메시지 처리
        response = await ai_assistant.process_chat_message(
            user_id=current_user.id,
            message=message_data["message"],
            portfolio_id=message_data.get("portfolio_id")
        )
        
        # 응답 전송
        await websocket.send_text(json.dumps({
            "type": "ai_response",
            "message": response["message"],
            "adjustments": response.get("adjustments", []),
            "timestamp": datetime.now().isoformat()
        }, ensure_ascii=False))
        
except Exception as e:
    logger.error(f"WebSocket chat error: {e}")
    await websocket.send_text(json.dumps({
        "type": "error",
        "message": "연결 오류가 발생했습니다.",
        "error": str(e)
    }, ensure_ascii=False))
finally:
    await websocket.close()
```

# 추가적인 유틸리티 엔드포인트들

@router.get(”/models/info”)
async def get_model_info():
“”“AI 모델 정보 조회”””
try:
model_info = {
“primary_model”: “llama3.1:8b”,
“model_type”: “Local LLM (Ollama)”,
“capabilities”: [
“자연어 이해”,
“포트폴리오 분석”,
“투자 조언”,
“리스크 평가”,
“시장 분석”
],
“supported_languages”: [“한국어”, “English”],
“max_context_length”: 2048,
“response_time_target”: “< 5초”
}

```
    return StandardResponse(
        success=True,
        data=model_info,
        message="AI 모델 정보입니다."
    )
    
except Exception as e:
    logger.error(f"Model info error: {e}")
    raise HTTPException(status_code=500, detail="모델 정보 조회 실패")
```

@router.post(”/prompt/validate”)
async def validate_prompt(
prompt: str = Field(…, description=“검증할 프롬프트”),
current_user: User = Depends(get_current_user)
):
“””
프롬프트 검증

```
사용자 입력의 안전성과 유효성을 사전 검증
"""
try:
    validation_result = PromptValidator.validate_user_input(prompt)
    
    return StandardResponse(
        success=validation_result["is_valid"],
        data=validation_result,
        message="프롬프트 검증이 완료되었습니다."
    )
    
except Exception as e:
    logger.error(f"Prompt validation error: {e}")
    raise HTTPException(status_code=500, detail="프롬프트 검증 실패")
```

@router.get(”/analytics/usage”)
async def get_ai_usage_analytics(
period_days: int = 30,
current_user: User = Depends(get_current_user)
):
“””
AI 사용 분석

```
사용자의 AI 사용 패턴과 통계 제공
"""
try:
    # 실제 구현에서는 데이터베이스에서 사용 통계 조회
    analytics = {
        "period_days": period_days,
        "total_messages": 42,  # 임시 값
        "avg_messages_per_day": 1.4,
        "most_used_commands": [
            {"command": "종목 추천", "count": 15},
            {"command": "포트폴리오 분석", "count": 12},
            {"command": "리스크 평가", "count": 8}
        ],
        "satisfaction_rating": 4.2,
        "response_time_avg": 3.2
    }
    
    return StandardResponse(
        success=True,
        data=analytics,
        message="AI 사용 분석 결과입니다."
    )
    
except Exception as e:
    logger.error(f"Usage analytics error: {e}")
    raise HTTPException(status_code=500, detail="사용 분석 실패")
```

# 에러 핸들러들

@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
“”“HTTP 예외 처리”””
return StandardResponse(
success=False,
data=None,
message=exc.detail,
error_code=exc.status_code
)

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
“”“일반 예외 처리”””
logger.error(f”Unhandled exception in AI API: {exc}”)
return StandardResponse(
success=False,
data=None,
message=“내부 서버 오류가 발생했습니다.”,
error_code=500
)