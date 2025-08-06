“””
Ollama LLM 연동 서비스 - Master’s Eye
파일 위치: backend/app/services/ai_chat.py

로컬 LLM(Llama 3.1 8B)을 활용한 대화형 포트폴리오 조정 시스템
“””

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import httpx
from sqlalchemy.orm import Session

from ..core.database import get_db_session
from ..models.user import User
from ..models.portfolio import Portfolio
from ..models.news import ChatLog
from ..masters.fusion import MastersFusionEngine, MasterWeight
from ..utils.formatters import format_currency, format_percentage
from ..utils.calculations import calculate_portfolio_risk

logger = logging.getLogger(**name**)

@dataclass
class ChatMessage:
“”“채팅 메시지 구조”””
role: str  # “user” or “assistant”
content: str
timestamp: datetime
metadata: Optional[Dict] = None

@dataclass
class PortfolioAdjustment:
“”“포트폴리오 조정 명령”””
action: str  # “add”, “remove”, “modify”, “rebalance”
ticker: Optional[str] = None
weight: Optional[float] = None
reason: Optional[str] = None
parameters: Optional[Dict] = None

class OllamaClient:
“”“Ollama API 클라이언트”””

```
def __init__(self, base_url: str = "http://localhost:11434"):
    self.base_url = base_url
    self.model = "llama3.1:8b"  # Llama 3.1 8B 모델
    self.timeout = 60.0

async def generate_response(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None,
                          temperature: float = 0.3) -> Dict[str, Any]:
    """LLM 응답 생성"""
    try:
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                "content": result["message"]["content"],
                "model": result["model"],
                "created_at": result["created_at"],
                "done": result["done"]
            }
            
    except httpx.TimeoutException:
        logger.error("Ollama request timeout")
        raise Exception("AI 응답 시간 초과")
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama HTTP error: {e}")
        raise Exception("AI 서비스 오류")
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        raise Exception("AI 응답 생성 실패")
```

class FinancialPromptBuilder:
“”“금융 특화 프롬프트 빌더”””

```
def __init__(self):
    self.base_system_prompt = """
```

당신은 워렌 버핏, 레이 달리오, 리처드 파인만, 짐 사이먼스의 투자 철학을 융합한
세계 최고 수준의 AI 포트폴리오 매니저입니다.

핵심 역할:

1. 사용자의 자연어 요청을 포트폴리오 조정 명령으로 변환
1. 투자 결정에 대한 명확하고 근거 있는 설명 제공
1. 리스크와 불확실성에 대한 정직한 평가
1. 4대 거장의 관점을 종합한 균형잡힌 조언

응답 원칙:

- 명확하고 실행 가능한 조언
- 리스크 경고 포함
- 투자 근거 상세 설명
- 불확실성 솔직히 인정
- 감정적 판단보다 데이터 기반 분석

응답 형식:

1. 요약 (한 줄)
1. 구체적 조정안
1. 투자 근거
   4