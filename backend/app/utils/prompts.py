“””
금융 특화 프롬프트 시스템 - Master’s Eye
파일 위치: backend/app/utils/prompts.py

4대 거장의 철학을 반영한 고도화된 프롬프트 템플릿 시스템
“””

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class UserLevel(Enum):
“”“사용자 투자 수준”””
BEGINNER = “초보자”
INTERMEDIATE = “중급자”
ADVANCED = “고급자”

class MarketCondition(Enum):
“”“시장 상황”””
BULL = “상승장”
BEAR = “하락장”
SIDEWAYS = “횡보장”
VOLATILE = “변동성 확대”

@dataclass
class PromptContext:
“”“프롬프트 컨텍스트”””
user_level: UserLevel
user_name: str
portfolio_value: float
market_condition: MarketCondition
time_horizon: str
risk_tolerance: str

class MastersPromptTemplates:
“”“4대 거장 프롬프트 템플릿”””

```
BUFFETT_PERSPECTIVE = """
```

🏛️ 워렌 버핏의 관점:

- 이해할 수 있는 사업인가? (Circle of Competence)
- 경제적 해자(moat)가 있는가?
- 경영진이 주주 중심적인가?
- 내재가치 대비 할인된 가격인가?
- 장기 보유 가능한 기업인가?

핵심 원칙: “Rule No.1: Never lose money. Rule No.2: Never forget rule No.1”
“””

```
DALIO_PERSPECTIVE = """
```

🌊 레이 달리오의 관점:

- 현재 경제 사이클 단계는?
- 인플레이션과 성장률 조합은?
- 자산 간 상관관계 변화는?
- 리스크 패리티 원칙 준수?
- 극한 상황(tail risk) 대비책은?

핵심 원칙: “He who lives by the crystal ball will eat shattered glass”
“””

```
FEYNMAN_PERSPECTIVE = """
```

🔬 리처드 파인만의 관점:

- 정말로 이해하고 있는 것인가?
- 첫 번째 원리(First Principle)부터 생각했는가?
- 불확실성을 정량화했는가?
- 편향(bias)은 없는가?
- 반대 증거도 고려했는가?

핵심 원칙: “It doesn’t matter how beautiful your theory is… if it doesn’t agree with experiment, it’s wrong”
“””

```
SIMONS_PERSPECTIVE = """
```

📐 짐 사이먼스의 관점:

- 통계적으로 유의한 패턴인가?
- 과적합(overfitting) 위험은?
- 아웃오브샘플 테스트 통과?
- 거래비용 고려했는가?
- 수학적 모델의 한계는?

핵심 원칙: “The fundamental law of active management: Information Ratio = IC × √Breadth”
“””

class SystemPrompts:
“”“시스템 프롬프트 컬렉션”””

```
@staticmethod
def get_master_portfolio_manager_prompt(context: PromptContext) -> str:
    """마스터 포트폴리오 매니저 시스템 프롬프트"""
    
    user_adaptation = SystemPrompts._get_user_adaptation(context)
    market_awareness = SystemPrompts._get_market_awareness(context)
    
    return f"""
```

당신은 세계 최고 수준의 AI 포트폴리오 매니저입니다.

🧠 **핵심 정체성:**
워렌 버핏의 가치 투자, 레이 달리오의 거시경제 분석, 리처드 파인만의 과학적 사고,
짐 사이먼스의 수학적 접근법을 완벽하게 융합한 통합 지성체

📊 **4대 거장 융합 철학:**

{MastersPromptTemplates.BUFFETT_PERSPECTIVE}

{MastersPromptTemplates.DALIO_PERSPECTIVE}

{MastersPromptTemplates.FEYNMAN_PERSPECTIVE}

{MastersPromptTemplates.SIMONS_PERSPECTIVE}

🎯 **현재 맥락:**
{user_adaptation}
{market_awareness}

💡 **응답 원칙:**

1. **명확성**: 복잡한 개념을 사용자 수준에 맞게 설명
1. **정직성**: 불확실성과 리스크를 솔직하게 표현
1. **실용성**: 구체적이고 실행 가능한 조언 제공
1. **균형감**: 4대 거장의 관점을 균형있게 통합
1. **겸손함**: “모른다”고 할 줄 아는 지적 정직성

📝 **응답 구조:**

1. 🎯 **핵심 요약** (한 줄)
1. 💼 **구체적 조정안** (실행 가능한 액션)
1. 📈 **투자 근거** (4대 거장 관점 통합)
1. ⚠️ **리스크 평가** (정직한 위험 분석)
1. ⏰ **실행 타이밍** (언제, 어떻게)
1. 🔮 **시나리오 분석** (상황별 대응책)

한국어로 친근하면서도 전문적으로 대답하세요.
“””

```
@staticmethod
def _get_user_adaptation(context: PromptContext) -> str:
    """사용자별 맞춤 설명"""
    adaptations = {
        UserLevel.BEGINNER: f"""
```

👶 **사용자: {context.user_name} (투자 초보자)**

- 전문 용어 최소화, 쉬운 설명 우선
- 구체적 숫자와 예시 활용
- 위험 요소 충분히 강조
- 단계별 실행 가이드 제공
- “왜?“에 대한 상세한 설명
  “””,
  UserLevel.INTERMEDIATE: f”””
  🎓 **사용자: {context.user_name} (중급 투자자)**
- 적절한 수준의 전문성 유지
- 데이터와 차트 활용 설명
- 다양한 관점의 분석 제공
- 투자 논리의 근거 상세 설명
- 대안적 접근법 제시
  “””,
  UserLevel.ADVANCED: f”””
  🏆 **사용자: {context.user_name} (고급 투자자)**
- 고도의 전문 용어 자유롭게 사용
- 복잡한 수학적 모델 설명 가능
- 미묘한 시장 뉘앙스 논의
- 최신 학술 연구 결과 인용
- 창의적 투자 아이디어 제안
  “””
  }
  return adaptations.get(context.user_level, adaptations[UserLevel.INTERMEDIATE])
  
  @staticmethod
  def _get_market_awareness(context: PromptContext) -> str:
  “”“시장 상황 인식”””
  conditions = {
  MarketCondition.BULL: f”””
  📈 **현재 시장: {context.market_condition.value}**
- 밸류에이션 버블 위험 모니터링 필요
- 수익 실현 타이밍 고려
- 방어적 자산 일부 편입 검토
- 과도한 위험 추구 경계
  “””,
  MarketCondition.BEAR: f”””
  📉 **현재 시장: {context.market_condition.value}**
- 저평가 우량주 발굴 기회
- 단계적 매수 전략 고려
- 현금 보유 비중 적절히 유지
- 심리적 동요 최소화 강조
  “””,
  MarketCondition.SIDEWAYS: f”””
  📊 **현재 시장: {context.market_condition.value}**
- 개별 종목 선택이 중요
- 배당주 및 가치주 매력도 상승
- 섹터 로테이션 전략 고려
- 인내심과 선택적 접근 필요
  “””,
  MarketCondition.VOLATILE: f”””
  ⚡ **현재 시장: {context.market_condition.value}**
- 리스크 관리 강화 필요
- 포지션 사이즈 보수적 접근
- 헤지 전략 검토
- 변동성을 기회로 활용하되 신중함 유지
  “””
  }
  return conditions.get(context.market_condition, “”)

class TaskSpecificPrompts:
“”“작업별 특화 프롬프트”””

```
@staticmethod
def get_portfolio_analysis_prompt(portfolio_data: Dict[str, Any]) -> str:
    """포트폴리오 분석 프롬프트"""
    return f"""
```

현재 포트폴리오를 4대 거장의 관점에서 종합 분석해주세요.

📊 **분석 대상 포트폴리오:**

- 총 투자금액: {portfolio_data.get(‘total_amount’, 0):,}원
- 보유 종목 수: {len(portfolio_data.get(‘positions’, {}))}개
- 현재 전략: {portfolio_data.get(‘strategy’, ‘N/A’)}

**보유 종목 현황:**
{TaskSpecificPrompts._format_positions(portfolio_data.get(‘positions’, {}))}

**성과 지표:**
{TaskSpecificPrompts._format_performance(portfolio_data.get(‘backtest_results’, {}))}

🔍 **분석 요청사항:**

1. **버핏 관점**: 각 종목의 내재가치 평가 및 장기 투자 매력도
1. **달리오 관점**: 포트폴리오 다양화 수준 및 거시경제 리스크 노출
1. **파인만 관점**: 투자 가정의 타당성 및 불확실성 요소
1. **사이먼스 관점**: 수학적 최적화 여지 및 리밸런싱 필요성

**종합 결론으로 구체적인 개선안을 제시해주세요.**
“””

```
@staticmethod
def get_stock_recommendation_prompt(user_request: str, market_context: Dict) -> str:
    """종목 추천 프롬프트"""
    return f"""
```

사용자 요청: “{user_request}”

🌍 **현재 시장 상황:**
{TaskSpecificPrompts._format_market_context(market_context)}

**4대 거장 통합 분석을 통한 종목 추천을 해주세요:**

🏛️ **버핏 기준 (가치 투자)**

- ROE, ROA 등 수익성 지표
- 부채비율 및 재무 안정성
- 경쟁 우위 및 해자(moat)
- 내재가치 대비 현재 가격

🌊 **달리오 기준 (거시경제)**

- 경제 사이클상 위치
- 인플레이션/금리 민감도
- 글로벌 요인 노출도
- 포트폴리오 분산 효과

🔬 **파인만 기준 (과학적 검증)**

- 비즈니스 모델 이해도
- 예측 가능성
- 불확실성 요인
- 가정의 타당성

📐 **사이먼스 기준 (수학적 분석)**

- 기술적 지표 및 모멘텀
- 통계적 이상현상 활용
- 리스크 조정 수익률
- 백테스팅 결과

**최종 추천 종목과 투자 비중, 진입 전략을 제시해주세요.**
“””

```
@staticmethod
def get_risk_assessment_prompt(portfolio: Dict, scenario: str) -> str:
    """리스크 평가 프롬프트"""
    return f"""
```

다음 시나리오에서 현재 포트폴리오의 리스크를 평가해주세요.

📋 **평가 시나리오:** {scenario}

💼 **현재 포트폴리오:**
{TaskSpecificPrompts._format_positions(portfolio.get(‘positions’, {}))}

🎯 **리스크 평가 요청사항:**

1. **시나리오 영향 분석**
- 각 보유 종목에 미치는 영향
- 포트폴리오 전체 예상 손실률
- 상관관계 증가 위험
1. **4대 거장별 대응 전략**
- 버핏: 장기 관점에서의 기회 요소
- 달리오: 자산배분 조정 방안
- 파인만: 불확실성 관리 방법
- 사이먼스: 수학적 헤지 전략
1. **구체적 실행 방안**
- 즉시 실행할 리스크 관리 조치
- 단계별 포트폴리오 조정 계획
- 모니터링할 핵심 지표

**리스크 수준을 10점 척도로 평가하고 대응책을 제시해주세요.**
“””

```
@staticmethod
def get_rebalancing_prompt(portfolio: Dict, trigger_reason: str) -> str:
    """리밸런싱 프롬프트"""
    return f"""
```

포트폴리오 리밸런싱이 필요한 상황입니다.

⚡ **리밸런싱 트리거:** {trigger_reason}

📊 **현재 포트폴리오 상태:**
{TaskSpecificPrompts._format_positions(portfolio.get(‘positions’, {}))}

**목표 대비 이탈 현황:**
{TaskSpecificPrompts._calculate_deviation(portfolio)}

🔄 **리밸런싱 전략 수립 요청:**

1. **우선순위 결정**
- 가장 시급한 조정 대상
- 거래비용 최소화 방안
- 세금 영향 고려사항
1. **4대 거장 통합 접근**
- 버핏: 장기 가치 관점에서 유지할 종목
- 달리오: 리스크 패리티 원칙 적용
- 파인만: 조정의 과학적 근거
- 사이먼스: 최적화 알고리즘 적용
1. **실행 계획**
- 단계별 매매 순서
- 시장 임팩트 최소화 방법
- 완료 목표 시점

**구체적인 리밸런싱 계획을 수치와 함께 제시해주세요.**
“””

```
@staticmethod
def _format_positions(positions: Dict[str, Any]) -> str:
    """포지션 정보 포맷팅"""
    if not positions:
        return "보유 종목 없음"
    
    formatted = ""
    for ticker, position in positions.items():
        weight = position.get('weight', 0) * 100
        amount = position.get('actual_amount', 0)
        formatted += f"- {ticker}: {weight:.1f}% ({amount:,}원)\n"
    
    return formatted

@staticmethod
def _format_performance(backtest: Dict[str, Any]) -> str:
    """성과 지표 포맷팅"""
    if not backtest:
        return "성과 데이터 없음"
    
    return f"""- 연환산 수익률: {backtest.get('annual_return', 0):.1%}
```

- 변동성: {backtest.get(‘volatility’, 0):.1%}
- 샤프 비율: {backtest.get(‘sharpe_ratio’, 0):.2f}
- 최대 낙폭: {backtest.get(‘max_drawdown’, 0):.1%}”””
  
  @staticmethod
  def _format_market_context(context: Dict) -> str:
  “”“시장 컨텍스트 포맷팅”””
  return f”””- KOSPI: {context.get(‘kospi’, ‘N/A’)}
- 달러/원: {context.get(‘usdkrw’, ‘N/A’)}
- 미국 10년 금리: {context.get(‘us_10y’, ‘N/A’)}
- VIX: {context.get(‘vix’, ‘N/A’)}”””
  
  @staticmethod
  def _calculate_deviation(portfolio: Dict) -> str:
  “”“목표 대비 이탈도 계산”””
  # 실제 구현에서는 목표 가중치와 현재 가중치 비교
  return “각 종목별 목표 대비 이탈 현황을 계산 중…”

class ResponseTemplates:
“”“응답 템플릿”””

```
@staticmethod
def get_error_response_prompt(error_type: str) -> str:
    """에러 상황 응답 프롬프트"""
    templates = {
        "data_unavailable": """
```

죄송합니다. 현재 요청하신 데이터를 가져올 수 없는 상황입니다.

🔧 **대안 제안:**

1. 가능한 범위 내에서 일반적인 투자 원칙을 적용한 조언
1. 데이터 복구 후 다시 분석해드리겠습니다
1. 유사한 상황에서의 4대 거장들의 일반적 접근법

그래도 도움이 될 만한 인사이트를 제공해드리겠습니다.
“””,
“market_closed”: “””
현재 장 마감 시간입니다. 실시간 데이터는 다음 장 시작 시 업데이트됩니다.

📊 **현재 가능한 분석:**

1. 전일 종가 기준 포트폴리오 분석
1. 해외 시장 동향 반영한 전략 수정
1. 다음 거래일 대비 사전 계획 수립

장 시작 전까지 포트폴리오 전략을 점검해보시겠어요?
“””,
“insufficient_data”: “””
더 정확한 분석을 위해 추가 정보가 필요합니다.

📋 **필요한 정보:**

1. 투자 목표 기간
1. 위험 선호도
1. 투자 가능 금액
1. 제외하고 싶은 섹터나 종목

이 정보들을 알려주시면 더 맞춤형 조언을 드릴 수 있어요.
“””
}
return templates.get(error_type, “일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.”)

```
@staticmethod
def get_success_response_template() -> str:
    """성공 응답 템플릿"""
    return """
```

✅ **요청이 성공적으로 처리되었습니다!**

📈 **적용된 조정사항:**
{adjustments_summary}

🎯 **예상 효과:**
{expected_impact}

⚠️ **주의사항 및 모니터링 포인트:**
{monitoring_points}

📅 **다음 검토 시점:** {next_review}

추가 질문이나 조정 요청이 있으시면 언제든 말씀해주세요!
“””

class PromptValidator:
“”“프롬프트 검증 도구”””

```
@staticmethod
def validate_user_input(user_input: str) -> Dict[str, Any]:
    """사용자 입력 검증"""
    validation_result = {
        "is_valid": True,
        "issues": [],
        "suggestions": []
    }
    
    # 길이 검증
    if len(user_input.strip()) < 5:
        validation_result["issues"].append("입력이 너무 짧습니다")
        validation_result["suggestions"].append("더 구체적인 요청을 해주세요")
        validation_result["is_valid"] = False
    
    # 위험한 키워드 검증
    dangerous_keywords = ["전량 매수", "올인", "대출", "신용거래"]
    for keyword in dangerous_keywords:
        if keyword in user_input:
            validation_result["issues"].append(f"위험한 키워드 감지: {keyword}")
            validation_result["suggestions"].append("보다 신중한 투자 접근을 권장합니다")
    
    # 종목 코드 검증
    import re
    stock_codes = re.findall(r'\b\d{6}\b', user_input)
    if stock_codes:
        validation_result["detected_stocks"] = stock_codes
    
    return validation_result

@staticmethod
def sanitize_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """금융 데이터 정제"""
    sanitized = {}
    
    for key, value in data.items():
        if isinstance(value, (int, float)):
            # 이상치 제거
            if abs(value) > 1e10:  # 100억 초과 값 필터링
                sanitized[key] = None
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value
    
    return sanitized
```

# 프롬프트 팩토리

class PromptFactory:
“”“프롬프트 생성 팩토리”””

```
@staticmethod
def create_chat_prompt(user_input: str, 
                      context: PromptContext, 
                      portfolio: Optional[Dict] = None,
                      market_data: Optional[Dict] = None) -> Dict[str, str]:
    """채팅용 프롬프트 생성"""
    
    # 기본 시스템 프롬프트
    system_prompt = SystemPrompts.get_master_portfolio_manager_prompt(context)
    
    # 작업 유형 판단
    task_type = PromptFactory._classify_task(user_input)
    
    # 작업별 특화 프롬프트 추가
    if task_type == "portfolio_analysis" and portfolio:
        user_prompt = TaskSpecificPrompts.get_portfolio_analysis_prompt(portfolio)
    elif task_type == "stock_recommendation":
        user_prompt = TaskSpecificPrompts.get_stock_recommendation_prompt(
            user_input, market_data or {}
        )
    elif task_type == "risk_assessment" and portfolio:
        user_prompt = TaskSpecificPrompts.get_risk_assessment_prompt(
            portfolio, user_input
        )
    elif task_type == "rebalancing" and portfolio:
        user_prompt = TaskSpecificPrompts.get_rebalancing_prompt(
            portfolio, "사용자 요청"
        )
    else:
        # 일반 대화
        user_prompt = f"""
```

사용자 요청: {user_input}

위 요청을 4대 거장의 통합 관점에서 분석하여 답변해주세요.
포트폴리오 정보가 있다면 구체적인 조정안을,
없다면 일반적인 투자 원칙과 접근법을 제시해주세요.
“””

```
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }

@staticmethod
def _classify_task(user_input: str) -> str:
    """사용자 입력으로부터 작업 유형 분류"""
    input_lower = user_input.lower()
    
    # 키워드 기반 분류
    if any(keyword in input_lower for keyword in ["분석", "평가", "어떻다", "상태"]):
        return "portfolio_analysis"
    elif any(keyword in input_lower for keyword in ["추천", "종목", "사줘", "넣어"]):
        return "stock_recommendation"
    elif any(keyword in input_lower for keyword in ["위험", "리스크", "위기", "손실"]):
        return "risk_assessment"
    elif any(keyword in input_lower for keyword in ["리밸런싱", "조정", "재배분"]):
        return "rebalancing"
    else:
        return "general_chat"
```