# backend/app/api/v1/trading.py

“””
거래 및 분석 REST API 엔드포인트

- 투자자별 매매동향
- 체결강도 분석
- 거래량 분석
  “””

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio

from …data.collectors.kis_api import KISApiClient, KISConfig, MarketType
from …core.config import get_kis_config
from …core.logging import get_main_logger, log_execution_time

logger = get_main_logger()

router = APIRouter(prefix=”/trading”, tags=[“trading”])

async def get_kis_client() -> KISApiClient:
“”“KIS API 클라이언트 의존성”””
config = get_kis_config()
client = KISApiClient(config)
await client.initialize()
return client

@router.get(”/{ticker}/investor”)
@log_execution_time()
async def get_investor_trading(
ticker: str,
market: str = Query(“KOSPI”, description=“시장 구분”),
client: KISApiClient = Depends(get_kis_client)
):
“””
투자자별 매매동향 조회

```
- **ticker**: 6자리 종목 코드
- **market**: 시장 구분 (KOSPI, KOSDAQ, KONEX)
"""
try:
    if not _validate_ticker(ticker):
        raise HTTPException(status_code=400, detail="유효하지 않은 종목 코드입니다")
    
    market_type = _get_market_type(market)
    
    async with client:
        trading_data = await client.get_trading_by_investor(ticker, market_type)
        
    # 데이터 구조화
    response_data = {
        "ticker": trading_data.ticker,
        "individual": {
            "buy_volume": trading_data.individual["buy"],
            "sell_volume": trading_data.individual["sell"],
            "net_volume": trading_data.individual["net"],
            "net_amount": trading_data.individual["net"]  # 실제로는 금액 계산 필요
        },
        "foreign": {
            "buy_volume": trading_data.foreign["buy"],
            "sell_volume": trading_data.foreign["sell"],
            "net_volume": trading_data.foreign["net"],
            "net_amount": trading_data.foreign["net"]
        },
        "institutional": {
            "buy_volume": trading_data.institutional["buy"],
            "sell_volume": trading_data.institutional["sell"],
            "net_volume": trading_data.institutional["net"],
            "net_amount": trading_data.institutional["net"]
        },
        "total_volume": (trading_data.individual["buy"] + trading_data.individual["sell"] +
                       trading_data.foreign["buy"] + trading_data.foreign["sell"] +
                       trading_data.institutional["buy"] + trading_data.institutional["sell"]),
        "timestamp": trading_data.timestamp.isoformat()
    }
    
    # 투자자별 비중 계산
    total_buy = (trading_data.individual["buy"] + trading_data.foreign["buy"] + 
                trading_data.institutional["buy"])
    
    if total_buy > 0:
        response_data["buy_ratio"] = {
            "individual": round(trading_data.individual["buy"] / total_buy * 100, 2),
            "foreign": round(trading_data.foreign["buy"] / total_buy * 100, 2),
            "institutional": round(trading_data.institutional["buy"] / total_buy * 100, 2)
        }
    
    return {
        "ticker": ticker,
        "data": response_data,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"투자자별 매매동향 조회 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"매매동향 조회 실패: {str(e)}")
```

@router.get(”/{ticker}/strength”)
@log_execution_time()
async def get_execution_strength(
ticker: str,
market: str = Query(“KOSPI”, description=“시장 구분”),
client: KISApiClient = Depends(get_kis_client)
):
“””
체결강도 조회

```
- **ticker**: 6자리 종목 코드
- **market**: 시장 구분
"""
try:
    if not _validate_ticker(ticker):
        raise HTTPException(status_code=400, detail="유효하지 않은 종목 코드입니다")
    
    market_type = _get_market_type(market)
    
    async with client:
        strength_data = await client.get_execution_strength(ticker, market_type)
        
    # 체결강도 레벨 계산
    strength_level = _calculate_strength_level(strength_data.strength)
    
    response_data = {
        "ticker": strength_data.ticker,
        "strength": round(strength_data.strength, 2),
        "strength_level": strength_level,
        "buy_volume": strength_data.buy_volume,
        "sell_volume": strength_data.sell_volume,
        "net_volume": strength_data.net_volume,
        "buy_ratio": round(strength_data.buy_volume / (strength_data.buy_volume + strength_data.sell_volume) * 100, 2) if (strength_data.buy_volume + strength_data.sell_volume) > 0 else 0,
        "interpretation": _interpret_strength(strength_data.strength),
        "timestamp": strength_data.timestamp.isoformat()
    }
    
    return {
        "ticker": ticker,
        "data": response_data,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"체결강도 조회 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"체결강도 조회 실패: {str(e)}")
```

@router.get(”/batch/investor”)
@log_execution_time()
async def get_batch_investor_trading(
tickers: List[str] = Query(…, description=“종목 코드 목록”),
market: str = Query(“KOSPI”, description=“시장 구분”),
client: KISApiClient = Depends(get_kis_client)
):
“””
다수 종목 투자자별 매매동향 일괄 조회

```
- **tickers**: 종목 코드 목록 (최대 20개)
- **market**: 시장 구분
"""
try:
    if len(tickers) > 20:
        raise HTTPException(status_code=400, detail="한 번에 최대 20개 종목까지 조회 가능합니다")
    
    # 티커 유효성 검증
    invalid_tickers = [t for t in tickers if not _validate_ticker(t)]
    if invalid_tickers:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 종목 코드: {invalid_tickers}")
    
    market_type = _get_market_type(market)
    
    results = {}
    
    async with client:
        for ticker in tickers:
            try:
                trading_data = await client.get_trading_by_investor(ticker, market_type)
                
                results[ticker] = {
                    "individual_net": trading_data.individual["net"],
                    "foreign_net": trading_data.foreign["net"],
                    "institutional_net": trading_data.institutional["net"],
                    "timestamp": trading_data.timestamp.isoformat()
                }
                
                # API 호출 간격
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"종목 {ticker} 매매동향 조회 실패: {e}")
                results[ticker] = {
                    "error": str(e)
                }
    
    return {
        "requested_tickers": tickers,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"일괄 매매동향 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=f"일괄 조회 실패: {str(e)}")
```

# backend/app/api/v1/analysis.py

“””
분석 REST API 엔드포인트

- 기술적 분석
- 가격 분석
- 시장 분석
  “””

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import math

from …data.collectors.kis_api import KISApiClient, KISConfig, MarketType
from …core.config import get_kis_config
from …core.logging import get_main_logger, log_execution_time

logger = get_main_logger()

analysis_router = APIRouter(prefix=”/analysis”, tags=[“analysis”])

async def get_kis_client() -> KISApiClient:
“”“KIS API 클라이언트 의존성”””
config = get_kis_config()
client = KISApiClient(config)
await client.initialize()
return client

@analysis_router.get(”/{ticker}/technical”)
@log_execution_time()
async def get_technical_analysis(
ticker: str,
period: str = Query(“D”, description=“기간 (D:일봉, W:주봉, M:월봉)”),
count: int = Query(50, ge=20, le=200, description=“분석 기간”),
client: KISApiClient = Depends(get_kis_client)
):
“””
기술적 분석

```
- **ticker**: 6자리 종목 코드
- **period**: 분석 기간
- **count**: 분석할 봉 개수
"""
try:
    if not _validate_ticker(ticker):
        raise HTTPException(status_code=400, detail="유효하지 않은 종목 코드입니다")
    
    async with client:
        ohlcv_data = await client.get_ohlcv_data(ticker, period, count)
        
    if len(ohlcv_data) < 20:
        raise HTTPException(status_code=400, detail="분석에 필요한 최소 데이터가 부족합니다")
    
    # 기술적 지표 계산
    technical_indicators = _calculate_technical_indicators(ohlcv_data)
    
    # 추세 분석
    trend_analysis = _analyze_trend(ohlcv_data)
    
    # 지지/저항선 분석
    support_resistance = _calculate_support_resistance(ohlcv_data)
    
    response_data = {
        "ticker": ticker,
        "period": period,
        "data_count": len(ohlcv_data),
        "technical_indicators": technical_indicators,
        "trend_analysis": trend_analysis,
        "support_resistance": support_resistance,
        "last_update": ohlcv_data[0]["date"] if ohlcv_data else None,
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "ticker": ticker,
        "analysis": response_data,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"기술적 분석 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"기술적 분석 실패: {str(e)}")
```

@analysis_router.get(”/{ticker}/volatility”)
@log_execution_time()
async def get_volatility_analysis(
ticker: str,
period: str = Query(“D”, description=“기간”),
count: int = Query(30, ge=10, le=100, description=“분석 기간”),
client: KISApiClient = Depends(get_kis_client)
):
“””
변동성 분석

```
- **ticker**: 6자리 종목 코드
- **period**: 분석 기간
- **count**: 분석할 봉 개수
"""
try:
    if not _validate_ticker(ticker):
        raise HTTPException(status_code=400, detail="유효하지 않은 종목 코드입니다")
    
    async with client:
        ohlcv_data = await client.get_ohlcv_data(ticker, period, count)
        
    if len(ohlcv_data) < 10:
        raise HTTPException(status_code=400, detail="분석에 필요한 최소 데이터가 부족합니다")
    
    # 변동성 계산
    volatility_data = _calculate_volatility(ohlcv_data)
    
    response_data = {
        "ticker": ticker,
        "period": period,
        "analysis_period": f"{count}일",
        "historical_volatility": volatility_data["historical_volatility"],
        "price_volatility": volatility_data["price_volatility"],
        "volume_volatility": volatility_data["volume_volatility"],
        "volatility_trend": volatility_data["trend"],
        "risk_level": volatility_data["risk_level"],
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "ticker": ticker,
        "volatility_analysis": response_data,
        "timestamp": datetime.now().isoformat()
    }
    
except Exception as e:
    logger.error(f"변동성 분석 실패 ({ticker}): {e}")
    raise HTTPException(status_code=500, detail=f"변동성 분석 실패: {str(e)}")
```

# 유틸리티 함수들

def _validate_ticker(ticker: str) -> bool:
“”“티커 유효성 검증”””
import re
if not isinstance(ticker, str):
return False
pattern = r’^\d{6}$’
return bool(re.match(pattern, ticker))

def _get_market_type(market: str) -> MarketType:
“”“시장 구분 문자열을 MarketType으로 변환”””
market_map = {
“KOSPI”: MarketType.KOSPI,
“KOSDAQ”: MarketType.KOSDAQ,
“KONEX”: MarketType.KONEX
}

```
if market.upper() not in market_map:
    raise HTTPException(status_code=400, detail=f"유효하지 않은 시장 구분: {market}")

return market_map[market.upper()]
```

def _calculate_strength_level(strength: float) -> str:
“”“체결강도 레벨 계산”””
if strength >= 2.0:
return “매우 강함”
elif strength >= 1.5:
return “강함”
elif strength >= 1.2:
return “보통”
elif strength >= 0.8:
return “약함”
else:
return “매우 약함”

def _interpret_strength(strength: float) -> str:
“”“체결강도 해석”””
if strength >= 2.0:
return “매수 우세가 매우 강합니다”
elif strength >= 1.5:
return “매수 우세입니다”
elif strength >= 1.2:
return “매수가 다소 우세합니다”
elif strength >= 0.8:
return “매도가 다소 우세합니다”
elif strength >= 0.5:
return “매도 우세입니다”
else:
return “매도 우세가 매우 강합니다”

def _calculate_technical_indicators(ohlcv_data: List[Dict]) -> Dict[str, Any]:
“”“기술적 지표 계산”””
if len(ohlcv_data) < 20:
return {}

```
# 종가 데이터 추출
closes = [item["close"] for item in ohlcv_data[::-1]]  # 최신 -> 과거 순으로 정렬

# 이동평균선
ma5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else None
ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
ma60 = sum(closes[-60:]) / 60 if len(closes) >= 60 else None

# RSI 계산
rsi = _calculate_rsi(closes, 14)

# 볼린저 밴드
bollinger = _calculate_bollinger_bands(closes, 20)

return {
    "moving_averages": {
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60
    },
    "rsi": rsi,
    "bollinger_bands": bollinger,
    "current_price": closes[-1] if closes else None
}
```

def _calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
“”“RSI 계산”””
if len(prices) < period + 1:
return None

```
gains = []
losses = []

for i in range(1, len(prices)):
    change = prices[i] - prices[i-1]
    if change > 0:
        gains.append(change)
        losses.append(0)
    else:
        gains.append(0)
        losses.append(abs(change))

if len(gains) < period:
    return None

avg_gain = sum(gains[-period:]) / period
avg_loss = sum(losses[-period:]) / period

if avg_loss == 0:
    return 100

rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

return round(rsi, 2)
```

def _calculate_bollinger_bands(prices: List[float], period: int = 20) -> Dict[str, float]:
“”“볼린저 밴드 계산”””
if len(prices) < period:
return {}

```
recent_prices = prices[-period:]
mean_price = sum(recent_prices) / len(recent_prices)

# 표준편차 계산
variance = sum((p - mean_price) ** 2 for p in recent_prices) / len(recent_prices)
std_dev = math.sqrt(variance)

return {
    "middle": round(mean_price, 0),
    "upper": round(mean_price + (2 * std_dev), 0),
    "lower": round(mean_price - (2 * std_dev), 0),
    "bandwidth": round((4 * std_dev) / mean_price * 100, 2)
}
```

def _analyze_trend(ohlcv_data: List[Dict]) -> Dict[str, Any]:
“”“추세 분석”””
if len(ohlcv_data) < 10:
return {}

```
# 최근 10일 종가
recent_closes = [item["close"] for item in ohlcv_data[:10]]

# 단순 추세 계산
first_price = recent_closes[-1]  # 10일 전
last_price = recent_closes[0]    # 현재

change_rate = (last_price - first_price) / first_price * 100

if change_rate > 5:
    trend = "상승"
elif change_rate < -5:
    trend = "하락"
else:
    trend = "횡보"

return {
    "short_term_trend": trend,
    "change_rate_10d": round(change_rate, 2),
    "trend_strength": "강함" if abs(change_rate) > 10 else "보통" if abs(change_rate) > 5 else "약함"
}
```

def _calculate_support_resistance(ohlcv_data: List[Dict]) -> Dict[str, Any]:
“”“지지/저항선 계산”””
if len(ohlcv_data) < 20:
return {}

```
# 최근 20일 데이터에서 고가/저가 추출
highs = [item["high"] for item in ohlcv_data[:20]]
lows = [item["low"] for item in ohlcv_data[:20]]

# 간단한 지지/저항선 계산
resistance = max(highs)
support = min(lows)
current_price = ohlcv_data[0]["close"]

return {
    "resistance": resistance,
    "support": support,
    "current_price": current_price,
    "distance_to_resistance": round((resistance - current_price) / current_price * 100, 2),
    "distance_to_support": round((current_price - support) / current_price * 100, 2)
}
```

def _calculate_volatility(ohlcv_data: List[Dict]) -> Dict[str, Any]:
“”“변동성 계산”””
if len(ohlcv_data) < 10:
return {}

```
closes = [item["close"] for item in ohlcv_data]
volumes = [item["volume"] for item in ohlcv_data]

# 가격 변동성 (일간 수익률의 표준편차)
returns = []
for i in range(1, len(closes)):
    daily_return = (closes[i-1] - closes[i]) / closes[i] * 100
    returns.append(daily_return)

if returns:
    price_volatility = math.sqrt(sum(r**2 for r in returns) / len(returns))
else:
    price_volatility = 0

# 거래량 변동성
if len(volumes) > 1:
    avg_volume = sum(volumes) / len(volumes)
    volume_variance = sum((v - avg_volume)**2 for v in volumes) / len(volumes)
    volume_volatility = math.sqrt(volume_variance) / avg_volume * 100
else:
    volume_volatility = 0

# 위험 레벨 판단
if price_volatility > 5:
    risk_level = "높음"
elif price_volatility > 2:
    risk_level = "보통"
else:
    risk_level = "낮음"

return {
    "historical_volatility": round(price_volatility * math.sqrt(252), 2),  # 연환산
    "price_volatility": round(price_volatility, 2),
    "volume_volatility": round(volume_volatility, 2),
    "trend": "증가" if len(returns) > 5 and returns[0] > returns[-1] else "감소",
    "risk_level": risk_level
}
```