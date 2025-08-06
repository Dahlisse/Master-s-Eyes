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
1. 리스크 평가
1. 실행 시점 제안

한국어로 친근하게 대답하되, 전문성을 잃지 마세요.
“””

```
def build_portfolio_context(self, 
                          portfolio: Dict[str, Any], 
                          market_data: Optional[Dict] = None) -> str:
    """포트폴리오 컨텍스트 생성"""
    context = f"""
```

현재 포트폴리오 정보:

전략: {portfolio.get(‘strategy’, ‘N/A’)}
총 투자금액: {format_currency(portfolio.get(‘total_amount’, 0))}

4대 거장 가중치:

- 워렌 버핏: {portfolio.get(‘master_weights’, {}).get(‘buffett’, 0):.1%}
- 레이 달리오: {portfolio.get(‘master_weights’, {}).get(‘dalio’, 0):.1%}
- 리처드 파인만: {portfolio.get(‘master_weights’, {}).get(‘feynman’, 0):.1%}
- 짐 사이먼스: {portfolio.get(‘master_weights’, {}).get(‘simons’, 0):.1%}

현재 보유 종목:
“””

```
    positions = portfolio.get('positions', {})
    for ticker, position in positions.items():
        context += f"- {ticker}: {position.get('weight', 0):.1%} ({format_currency(position.get('actual_amount', 0))})\n"
    
    # 성과 정보 추가
    backtest = portfolio.get('backtest_results', {})
    if backtest:
        context += f"""
```

백테스팅 결과:

- 연환산 수익률: {format_percentage(backtest.get(‘annual_return’, 0))}
- 변동성: {format_percentage(backtest.get(‘volatility’, 0))}
- 샤프 비율: {backtest.get(‘sharpe_ratio’, 0):.2f}
- 최대 낙폭: {format_percentage(backtest.get(‘max_drawdown’, 0))}
  “””
  
  ```
    # 시장 상황 추가
    if market_data:
        context += f"""
  ```

현재 시장 상황:

- KOSPI: {market_data.get(‘kospi’, ‘N/A’)}
- 달러/원: {market_data.get(‘usdkrw’, ‘N/A’)}
- 미국 10년 금리: {market_data.get(‘us_10y’, ‘N/A’)}
  “””
  
  ```
    return context
  ```
  
  def build_user_context(self, user_id: int, user_name: str) -> str:
  “”“사용자 컨텍스트 생성”””
  # 사용자별 맞춤 설명 수준 설정
  if user_name == “엄마”:
  return “””
  사용자: 엄마 (투자 초보자)
- 쉽고 이해하기 편한 설명 필요
- 전문 용어 최소화
- 구체적이고 실용적인 조언 선호
- 리스크에 대한 충분한 설명 필요
  “””
  else:
  return “””
  사용자: 나 (중급 투자자)
- 어느 정도 투자 지식 보유
- 분석적이고 상세한 설명 선호
- 데이터와 근거 중시
- 다양한 관점의 분석 원함
  “””

class AIPortfolioAssistant:
“”“AI 포트폴리오 어시스턴트”””

```
def __init__(self):
    self.ollama = OllamaClient()
    self.prompt_builder = FinancialPromptBuilder()
    self.fusion_engine = MastersFusionEngine()
    self.conversation_history: Dict[int, List[ChatMessage]] = {}

async def process_chat_message(self, 
                             user_id: int,
                             message: str,
                             portfolio_id: Optional[int] = None) -> Dict[str, Any]:
    """채팅 메시지 처리"""
    try:
        # 사용자 정보 조회
        user = await self._get_user_info(user_id)
        
        # 포트폴리오 정보 조회
        portfolio = None
        if portfolio_id:
            portfolio = await self._get_portfolio_info(portfolio_id)
        
        # 시장 데이터 조회
        market_data = await self._get_current_market_data()
        
        # 대화 컨텍스트 구성
        context = self._build_conversation_context(
            user, portfolio, market_data, message
        )
        
        # AI 응답 생성
        ai_response = await self._generate_ai_response(context)
        
        # 포트폴리오 조정 명령 파싱
        adjustments = self._parse_portfolio_adjustments(ai_response["content"])
        
        # 대화 기록 저장
        await self._save_chat_log(user_id, message, ai_response["content"], adjustments)
        
        # 응답 구성
        response = {
            "message": ai_response["content"],
            "adjustments": adjustments,
            "context": {
                "portfolio_id": portfolio_id,
                "market_summary": self._summarize_market_data(market_data)
            },
            "metadata": {
                "model": ai_response.get("model"),
                "timestamp": datetime.now(),
                "processing_time": ai_response.get("processing_time")
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return {
            "message": "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다. 다시 시도해 주세요.",
            "adjustments": [],
            "error": str(e)
        }

def _build_conversation_context(self, 
                              user: Dict,
                              portfolio: Optional[Dict],
                              market_data: Dict,
                              user_message: str) -> Dict[str, str]:
    """대화 컨텍스트 구성"""
    
    # 시스템 프롬프트
    system_prompt = self.prompt_builder.base_system_prompt
    
    # 사용자 컨텍스트
    user_context = self.prompt_builder.build_user_context(
        user["id"], user["name"]
    )
    
    # 포트폴리오 컨텍스트  
    portfolio_context = ""
    if portfolio:
        portfolio_context = self.prompt_builder.build_portfolio_context(
            portfolio, market_data
        )
    
    # 대화 히스토리
    history_context = self._build_history_context(user["id"])
    
    # 최종 프롬프트 구성
    prompt = f"""
```

{user_context}

{portfolio_context}

{history_context}

사용자 요청: {user_message}

위 정보를 바탕으로 포트폴리오 조정안을 제시해 주세요.
“””

```
    return {
        "system_prompt": system_prompt,
        "user_prompt": prompt
    }

def _build_history_context(self, user_id: int, max_messages: int = 5) -> str:
    """대화 히스토리 컨텍스트"""
    history = self.conversation_history.get(user_id, [])
    if not history:
        return ""
    
    context = "최근 대화 내용:\n"
    for msg in history[-max_messages:]:
        context += f"- {msg.role}: {msg.content[:100]}...\n"
    
    return context

async def _generate_ai_response(self, context: Dict[str, str]) -> Dict[str, Any]:
    """AI 응답 생성"""
    start_time = datetime.now()
    
    response = await self.ollama.generate_response(
        prompt=context["user_prompt"],
        system_prompt=context["system_prompt"],
        temperature=0.3
    )
    
    end_time = datetime.now()
    response["processing_time"] = (end_time - start_time).total_seconds()
    
    return response

def _parse_portfolio_adjustments(self, ai_response: str) -> List[PortfolioAdjustment]:
    """AI 응답에서 포트폴리오 조정 명령 파싱"""
    adjustments = []
    
    # 정규식 패턴들
    patterns = {
        "add_stock": r"추가[:\s]+([A-Z0-9]{6})\s*([0-9\.]+%?)",
        "remove_stock": r"제거[:\s]+([A-Z0-9]{6})",
        "modify_weight": r"([A-Z0-9]{6})\s*비중[:\s]+([0-9\.]+%)",
        "rebalance": r"리밸런싱|재조정"
    }
    
    # 종목 추가 파싱
    for match in re.finditer(patterns["add_stock"], ai_response):
        ticker = match.group(1)
        weight_str = match.group(2)
        weight = self._parse_percentage(weight_str)
        
        adjustments.append(PortfolioAdjustment(
            action="add",
            ticker=ticker,
            weight=weight,
            reason="AI 추천"
        ))
    
    # 종목 제거 파싱
    for match in re.finditer(patterns["remove_stock"], ai_response):
        ticker = match.group(1)
        
        adjustments.append(PortfolioAdjustment(
            action="remove",
            ticker=ticker,
            reason="AI 추천"
        ))
    
    # 비중 조정 파싱
    for match in re.finditer(patterns["modify_weight"], ai_response):
        ticker = match.group(1)
        weight_str = match.group(2)
        weight = self._parse_percentage(weight_str)
        
        adjustments.append(PortfolioAdjustment(
            action="modify",
            ticker=ticker,
            weight=weight,
            reason="비중 조정"
        ))
    
    # 리밸런싱 파싱
    if re.search(patterns["rebalance"], ai_response):
        adjustments.append(PortfolioAdjustment(
            action="rebalance",
            reason="전체 리밸런싱"
        ))
    
    return adjustments

def _parse_percentage(self, percentage_str: str) -> float:
    """퍼센트 문자열을 소수로 변환"""
    # "15%" -> 0.15, "0.15" -> 0.15
    cleaned = percentage_str.replace("%", "").strip()
    value = float(cleaned)
    
    # 1보다 크면 퍼센트로 간주
    if value > 1:
        return value / 100
    return value

async def apply_portfolio_adjustments(self, 
                                    portfolio_id: int,
                                    adjustments: List[PortfolioAdjustment]) -> Dict[str, Any]:
    """포트폴리오 조정 적용"""
    try:
        # 현재 포트폴리오 로드
        portfolio = await self._get_portfolio_info(portfolio_id)
        if not portfolio:
            raise ValueError("Portfolio not found")
        
        # 조정사항 적용
        updated_portfolio = await self._apply_adjustments(portfolio, adjustments)
        
        # 새 포트폴리오 성과 시뮬레이션
        simulation_results = await self._simulate_adjusted_portfolio(updated_portfolio)
        
        # 결과 저장
        saved_portfolio = await self._save_adjusted_portfolio(
            portfolio_id, updated_portfolio, adjustments
        )
        
        return {
            "success": True,
            "portfolio": saved_portfolio,
            "simulation": simulation_results,
            "applied_adjustments": len(adjustments),
            "message": "포트폴리오가 성공적으로 조정되었습니다."
        }
        
    except Exception as e:
        logger.error(f"Error applying adjustments: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "포트폴리오 조정 중 오류가 발생했습니다."
        }

async def _apply_adjustments(self, 
                           portfolio: Dict,
                           adjustments: List[PortfolioAdjustment]) -> Dict:
    """조정사항을 포트폴리오에 적용"""
    updated_positions = portfolio["positions"].copy()
    
    for adjustment in adjustments:
        if adjustment.action == "add" and adjustment.ticker:
            # 종목 추가
            updated_positions[adjustment.ticker] = {
                "weight": adjustment.weight,
                "ticker": adjustment.ticker,
                "reason": adjustment.reason
            }
            
        elif adjustment.action == "remove" and adjustment.ticker:
            # 종목 제거
            if adjustment.ticker in updated_positions:
                del updated_positions[adjustment.ticker]
                
        elif adjustment.action == "modify" and adjustment.ticker:
            # 비중 수정
            if adjustment.ticker in updated_positions:
                updated_positions[adjustment.ticker]["weight"] = adjustment.weight
                
        elif adjustment.action == "rebalance":
            # 전체 리밸런싱 - 동일 가중치로 재조정
            if updated_positions:
                equal_weight = 1.0 / len(updated_positions)
                for ticker in updated_positions:
                    updated_positions[ticker]["weight"] = equal_weight
    
    # 가중치 정규화
    total_weight = sum(pos.get("weight", 0) for pos in updated_positions.values())
    if total_weight > 0:
        for ticker in updated_positions:
            updated_positions[ticker]["weight"] /= total_weight
    
    # 업데이트된 포트폴리오 구성
    updated_portfolio = portfolio.copy()
    updated_portfolio["positions"] = updated_positions
    
    return updated_portfolio

async def _simulate_adjusted_portfolio(self, portfolio: Dict) -> Dict[str, Any]:
    """조정된 포트폴리오 시뮬레이션"""
    try:
        # 간단한 리스크 평가
        positions = portfolio.get("positions", {})
        weights = {ticker: pos["weight"] for ticker, pos in positions.items()}
        
        # 예상 수익률과 리스크 계산
        expected_return = sum(weights.values()) * 0.10  # 임시 값
        expected_risk = calculate_portfolio_risk(weights)
        
        return {
            "expected_annual_return": expected_return,
            "expected_volatility": expected_risk,
            "sharpe_ratio": expected_return / expected_risk if expected_risk > 0 else 0,
            "diversification_score": len(positions) / 20,  # 20개 기준
            "risk_level": "낮음" if expected_risk < 0.15 else "보통" if expected_risk < 0.25 else "높음"
        }
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return {"error": "시뮬레이션 실패"}

async def _get_user_info(self, user_id: int) -> Dict:
    """사용자 정보 조회"""
    # 실제 구현에서는 데이터베이스 조회
    return {
        "id": user_id,
        "name": "엄마" if user_id == 2 else "나"
    }

async def _get_portfolio_info(self, portfolio_id: int) -> Optional[Dict]:
    """포트폴리오 정보 조회"""
    # 실제 구현에서는 데이터베이스 조회
    # 임시 데이터 반환
    return {
        "id": portfolio_id,
        "strategy": "균형형",
        "total_amount": 10000000,
        "positions": {
            "005930": {"weight": 0.2, "ticker": "005930", "actual_amount": 2000000},
            "000660": {"weight": 0.15, "ticker": "000660", "actual_amount": 1500000},
        },
        "master_weights": {
            "buffett": 0.3,
            "dalio": 0.3,
            "feynman": 0.2,
            "simons": 0.2
        },
        "backtest_results": {
            "annual_return": 0.11,
            "volatility": 0.18,
            "sharpe_ratio": 0.61,
            "max_drawdown": -0.12
        }
    }

async def _get_current_market_data(self) -> Dict:
    """현재 시장 데이터 조회"""
    # 실제 구현에서는 실시간 데이터 조회
    return {
        "kospi": "2,650.15 (+1.2%)",
        "usdkrw": "1,320.50 (-0.3%)",
        "us_10y": "4.25% (+0.05%p)",
        "vix": "18.5 (-2.1%)"
    }

def _summarize_market_data(self, market_data: Dict) -> str:
    """시장 데이터 요약"""
    return f"KOSPI {market_data.get('kospi', 'N/A')}, 달러원 {market_data.get('usdkrw', 'N/A')}"

async def _save_chat_log(self, 
                       user_id: int,
                       user_message: str, 
                       ai_response: str,
                       adjustments: List[PortfolioAdjustment]):
    """채팅 로그 저장"""
    try:
        # 데이터베이스에 저장
        # 실제 구현 필요
        
        # 메모리에 대화 히스토리 저장
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append(
            ChatMessage("user", user_message, datetime.now())
        )
        self.conversation_history[user_id].append(
            ChatMessage("assistant", ai_response, datetime.now())
        )
        
        # 최대 20개 메시지만 유지
        if len(self.conversation_history[user_id]) > 20:
            self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
            
    except Exception as e:
        logger.error(f"Error saving chat log: {e}")

async def _save_adjusted_portfolio(self, 
                                 portfolio_id: int,
                                 updated_portfolio: Dict,
                                 adjustments: List[PortfolioAdjustment]) -> Dict:
    """조정된 포트폴리오 저장"""
    # 실제 구현에서는 데이터베이스에 저장
    return updated_portfolio

async def get_chat_history(self, user_id: int, limit: int = 50) -> List[Dict]:
    """채팅 히스토리 조회"""
    history = self.conversation_history.get(user_id, [])
    
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "metadata": msg.metadata
        }
        for msg in history[-limit:]
    ]

async def clear_chat_history(self, user_id: int) -> bool:
    """채팅 히스토리 삭제"""
    try:
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
        return True
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return False

def get_available_commands(self) -> List[Dict[str, str]]:
    """사용 가능한 명령어 목록"""
    return [
        {
            "command": "종목 추가",
            "example": "삼성전자를 10% 비중으로 추가해줘",
            "description": "새로운 종목을 포트폴리오에 추가"
        },
        {
            "command": "종목 제거", 
            "example": "현대차를 포트폴리오에서 제거해줘",
            "description": "기존 종목을 포트폴리오에서 제거"
        },
        {
            "command": "비중 조정",
            "example": "LG에너지솔루션 비중을 15%로 조정해줘", 
            "description": "기존 종목의 투자 비중 변경"
        },
        {
            "command": "리밸런싱",
            "example": "포트폴리오를 리밸런싱해줘",
            "description": "전체 포트폴리오 균형 재조정"
        },
        {
            "command": "전략 변경",
            "example": "더 공격적인 전략으로 변경해줘",
            "description": "투자 전략 스타일 변경"
        },
        {
            "command": "성과 분석",
            "example": "현재 포트폴리오 성과는 어때?",
            "description": "포트폴리오 성과 및 분석 결과 확인"
        }
    ]
```

# 전역 인스턴스

ai_assistant = AIPortfolioAssistant()