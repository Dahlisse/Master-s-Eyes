# backend/app/api/v1/market.py

“””
시장 정보 REST API 엔드포인트

- 시장 상태, 지수 정보
- 거래량 상위 종목
- 시장 개장 시간 정보
  “””

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio

from …data.collectors.kis_api import KISApiClient, KISConfig, MarketType
from …core.config import get_kis_config
from …core.logging import get_main_logger, log_execution_time
from …core.redis import get_redis_client

logger = get_main_logger()

router = APIRouter(prefix=”/market”, tags=[“market”])

async def get_kis_client() -> KISApiClient:
“”“KIS API 클라이언트 의존성”””
config = get_kis_config()
client = KISApiClient(config)
await client.initialize()
return client

@router.get(”/status”)
@log_execution_time()
async def get_market_status(
client: KISApiClient = Depends(get_kis_client)
):
“””
시장 상태 조회

```
- 장 운영 시간 및 상태
- 코스피, 코스닥 지수
- 현재 시간 정보
"""
try:
    async with client:
        market_status = await client.get_market_status()
        
    # 추가 시장 정보 계산
    current_time = datetime.now()
    market_schedule = _get_market_schedule(current_time)
    
    response_data = {
        "market_open": market_status["market_open"],
        "market_status": market_status["market_status"],
        "current_time": market_status["current_time"],
        "kospi": {
            "index": market_status["kospi_index"],
            "change": market_status["kospi_change"],
            "change_rate": market_status.get("kospi_change_rate", 0.0)
        },
        "schedule": market_schedule,
        "server_time": current_time.isoformat(),
        "timezone": "Asia/Seoul"
    }
    
    return {
        "data": response_data,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"시장 상태 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"시장 상태 조회 실패: {str(e)}")
```

@router.get(”/indices”)
@log_execution_time()
async def get_market_indices(
client: KISApiClient = Depends(get_kis_client)
):
“””
주요 지수 정보 조회

```
- 코스피, 코스닥 지수
- 섹터별 지수 (선택적)
"""
try:
    # 주요 지수 종목 코드 (지수는 특별한 코드 사용)
    major_indices = {
        "KOSPI": "001",
        "KOSDAQ": "101", 
        "KRX100": "003"
    }
    
    indices_data = {}
    
    async with client:
        # 기본 시장 상태에서 코스피 정보 획득
        market_status = await client.get_market_status()
        
        indices_data["KOSPI"] = {
            "name": "코스피",
            "value": market_status["kospi_index"],
            "change": market_status["kospi_change"],
            "change_rate": market_status.get("kospi_change_rate", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        # 추가 지수 정보는 실제 API 엔드포인트가 있을 때 구현
        # 현재는 기본 정보만 제공
        
    return {
        "indices": indices_data,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"지수 정보 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"지수 정보 조회 실패: {str(e)}")
```

@router.get(”/schedule”)
@log_execution_time()
async def get_market_schedule(
date: Optional[str] = Query(None, description=“조회 날짜 (YYYY-MM-DD)”)
):
“””
시장 개장 일정 조회

```
- **date**: 조회할 날짜 (기본: 오늘)
"""
try:
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="날짜 형식이 올바르지 않습니다 (YYYY-MM-DD)")
    else:
        target_date = datetime.now()
        
    schedule = _get_market_schedule(target_date)
    
    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "schedule": schedule,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"시장 일정 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"시장 일정 조회 실패: {str(e)}")
```

@router.get(”/trading-hours”)
@log_execution_time()
async def get_trading_hours():
“””
거래 시간 정보 조회
“””
try:
trading_hours = {
“regular_session”: {
“start”: “09:00”,
“end”: “15:30”,
“description”: “정규장”
},
“pre_market”: {
“start”: “08:30”,
“end”: “09:00”,
“description”: “동시호가 (장 시작)”
},
“after_market”: {
“start”: “15:30”,
“end”: “16:00”,
“description”: “동시호가 (장 마감)”
},
“extended_hours”: {
“start”: “16:00”,
“end”: “18:00”,
“description”: “시간외 거래”
},
“timezone”: “Asia/Seoul”,
“market_days”: [“Monday”, “Tuesday”, “Wednesday”, “Thursday”, “Friday”]
}

```
    # 현재 세션 판단
    current_time = datetime.now()
    current_session = _get_current_session(current_time)
    
    return {
        "trading_hours": trading_hours,
        "current_session": current_session,
        "current_time": current_time.strftime("%H:%M"),
        "timestamp": current_time.isoformat()
    }
    
except Exception as e:
    logger.error(f"거래 시간 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"거래 시간 조회 실패: {str(e)}")
```

@router.get(”/holidays”)
@log_execution_time()
async def get_market_holidays(
year: int = Query(None, description=“조회할 연도”)
):
“””
시장 휴장일 조회

```
- **year**: 조회할 연도 (기본: 현재 연도)
"""
try:
    if year is None:
        year = datetime.now().year
        
    if year < 2020 or year > 2030:
        raise HTTPException(status_code=400, detail="지원하지 않는 연도입니다 (2020-2030)")
        
    holidays = _get_market_holidays(year)
    
    return {
        "year": year,
        "holidays": holidays,
        "total_count": len(holidays),
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"휴장일 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"휴장일 조회 실패: {str(e)}")
```

@router.get(”/summary”)
@log_execution_time()
async def get_market_summary(
client: KISApiClient = Depends(get_kis_client)
):
“””
시장 종합 정보 조회

```
- 시장 상태, 주요 지수
- 거래 시간 정보
- 오늘 일정
"""
try:
    async with client:
        market_status = await client.get_market_status()
        
    current_time = datetime.now()
    
    summary_data = {
        "market_status": {
            "is_open": market_status["market_open"],
            "status_text": market_status["market_status"],
            "current_session": _get_current_session(current_time)
        },
        "indices": {
            "kospi": {
                "value": market_status["kospi_index"],
                "change": market_status["kospi_change"],
                "change_rate": market_status.get("kospi_change_rate", 0.0)
            }
        },
        "schedule": _get_market_schedule(current_time),
        "current_time": current_time.strftime("%H:%M"),
        "server_time": current_time.isoformat()
    }
    
    return {
        "summary": summary_data,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"시장 종합 정보 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"시장 종합 정보 조회 실패: {str(e)}")
```

# 유틸리티 함수들

def _get_market_schedule(date: datetime) -> Dict[str, Any]:
“”“특정 날짜의 시장 일정 반환”””
weekday = date.weekday()  # 0=월요일, 6=일요일

```
if weekday >= 5:  # 토요일(5), 일요일(6)
    return {
        "is_trading_day": False,
        "reason": "주말",
        "next_trading_day": _get_next_trading_day(date).strftime("%Y-%m-%d")
    }

# 공휴일 체크
holidays = _get_market_holidays(date.year)
date_str = date.strftime("%Y-%m-%d")

for holiday in holidays:
    if holiday["date"] == date_str:
        return {
            "is_trading_day": False,
            "reason": f"공휴일 ({holiday['name']})",
            "next_trading_day": _get_next_trading_day(date).strftime("%Y-%m-%d")
        }

return {
    "is_trading_day": True,
    "regular_hours": "09:00 ~ 15:30",
    "pre_market": "08:30 ~ 09:00",
    "after_market": "15:30 ~ 16:00",
    "extended_hours": "16:00 ~ 18:00"
}
```

def _get_current_session(current_time: datetime) -> str:
“”“현재 거래 세션 반환”””
hour = current_time.hour
minute = current_time.minute
current_minutes = hour * 60 + minute

```
# 분 단위로 변환된 시간들
pre_market_start = 8 * 60 + 30    # 08:30
regular_start = 9 * 60            # 09:00
regular_end = 15 * 60 + 30        # 15:30
after_market_end = 16 * 60        # 16:00
extended_end = 18 * 60            # 18:00

weekday = current_time.weekday()
if weekday >= 5:  # 주말
    return "closed"

if current_minutes < pre_market_start:
    return "closed"
elif current_minutes < regular_start:
    return "pre_market"
elif current_minutes < regular_end:
    return "regular"
elif current_minutes < after_market_end:
    return "after_market"
elif current_minutes < extended_end:
    return "extended"
else:
    return "closed"
```

def _get_next_trading_day(date: datetime) -> datetime:
“”“다음 거래일 반환”””
next_day = date + timedelta(days=1)

```
# 최대 10일까지 확인
for _ in range(10):
    weekday = next_day.weekday()
    
    # 주말이 아니고 공휴일이 아닌 경우
    if weekday < 5:  # 평일
```