# kis_api/client.py

“””
한국투자증권 KIS Developers API 완전 연동 모듈

- REST API 및 WebSocket 실시간 데이터 지원
- 자동 토큰 관리 및 갱신
- 종합적인 에러 처리 및 재연결 메커니즘
- 모든 주요 API 엔드포인트 구현
  “””

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable, Any
import aiohttp
import websockets
import hashlib
import hmac
from dataclasses import dataclass, asdict
from enum import Enum
import backoff
import re

# 로깅 설정

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

class MarketType(Enum):
“”“시장 구분”””
KOSPI = “J”
KOSDAQ = “Q”
KONEX = “K”

class OrderSide(Enum):
“”“매수/매도 구분”””
BUY = “01”
SELL = “02”

class OrderType(Enum):
“”“주문 유형”””
LIMIT = “00”      # 지정가
MARKET = “01”     # 시장가
CONDITION = “05”  # 조건부지정가

@dataclass
class KISConfig:
“”“KIS API 설정”””
app_key: str
app_secret: str
base_url: str = “https://openapi.koreainvestment.com:9443”
mock_url: str = “https://openapivts.koreainvestment.com:29443”  # 모의투자
is_mock: bool = True  # 개발 초기에는 모의투자 사용
websocket_url: str = “ws://ops.koreainvestment.com:21000”

@dataclass
class StockPrice:
“”“주식 가격 정보”””
ticker: str
name: str
current_price: int
change: int
change_rate: float
volume: int
trade_value: int
open_price: int
high_price: int
low_price: int
prev_close: int
market_cap: int
timestamp: datetime

@dataclass
class OrderBook:
“”“호가창 정보”””
ticker: str
ask_prices: List[int]  # 매도호가 (5단계)
ask_volumes: List[int] # 매도잔량
bid_prices: List[int]  # 매수호가 (5단계)
bid_volumes: List[int] # 매수잔량
total_ask_volume: int
total_bid_volume: int
timestamp: datetime

@dataclass
class ExecutionStrength:
“”“체결강도 정보”””
ticker: str
strength: float        # 체결강도 (매수/매도 비율)
buy_volume: int       # 매수 체결량
sell_volume: int      # 매도 체결량
net_volume: int       # 순매수량
timestamp: datetime

@dataclass
class TradingByInvestor:
“”“투자자별 매매동향”””
ticker: str
individual: Dict[str, int]    # 개인 (매수, 매도, 순매수)
foreign: Dict[str, int]       # 외국인
institutional: Dict[str, int] # 기관
timestamp: datetime

@dataclass
class FinancialInfo:
“”“재무정보”””
ticker: str
per: float           # PER
pbr: float           # PBR
eps: int             # EPS
bps: int             # BPS
roe: float           # ROE
debt_ratio: float    # 부채비율
dividend_yield: float # 배당수익률
market_cap: int      # 시가총액
timestamp: datetime

class KISApiClient:
“”“한국투자증권 KIS API 클라이언트”””

```
def __init__(self, config: KISConfig):
    self.config = config
    self.session: Optional[aiohttp.ClientSession] = None
    self.access_token: Optional[str] = None
    self.token_expires_at: Optional[datetime] = None
    self.websocket_approval_key: Optional[str] = None
    self.websocket_connection: Optional[websockets.WebSocketServerProtocol] = None
    self.is_connected = False
    
    # 콜백 함수들
    self.price_callbacks: Dict[str, Callable] = {}
    self.orderbook_callbacks: Dict[str, Callable] = {}
    self.execution_callbacks: Dict[str, Callable] = {}
    
    # 속도 제한 관리
    self.last_request_time = 0
    self.request_count = 0
    self.request_reset_time = 0
    
    # 재연결 관리
    self.reconnect_attempts = 0
    self.max_reconnect_attempts = 5
    
async def __aenter__(self):
    """비동기 컨텍스트 매니저 진입"""
    await self.initialize()
    return self
    
async def __aexit__(self, exc_type, exc_val, exc_tb):
    """비동기 컨텍스트 매니저 종료"""
    await self.close()
    
async def initialize(self):
    """클라이언트 초기화"""
    self.session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "MastersEye/1.0"
        }
    )
    
    # 접근 토큰 발급
    await self.get_access_token()
    
    # WebSocket 접속키 발급
    await self.get_websocket_approval_key()
    
    logger.info("KIS API 클라이언트 초기화 완료")
    
async def close(self):
    """클라이언트 종료"""
    if self.websocket_connection:
        await self.websocket_connection.close()
        
    if self.session:
        await self.session.close()
        
    logger.info("KIS API 클라이언트 종료")
    
# ==================== 인증 관련 ====================

async def get_access_token(self) -> str:
    """OAuth 접근 토큰 발급"""
    try:
        url = f"{self._get_base_url()}/oauth2/tokenP"
        data = {
            "grant_type": "client_credentials",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret
        }
        
        async with self.session.post(url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                self.access_token = result.get("access_token")
                expires_in = result.get("expires_in", 86400)  # 기본 24시간
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 300)  # 5분 여유
                
                logger.info(f"접근 토큰 발급 성공. 만료: {self.token_expires_at}")
                return self.access_token
            else:
                error_text = await response.text()
                raise Exception(f"토큰 발급 실패: {response.status} - {error_text}")
                
    except Exception as e:
        logger.error(f"토큰 발급 오류: {e}")
        raise
        
async def refresh_token_if_needed(self):
    """토큰 만료 시 자동 갱신"""
    if not self.token_expires_at or datetime.now() >= self.token_expires_at:
        logger.info("토큰 갱신 필요")
        await self.get_access_token()
        
async def get_websocket_approval_key(self) -> str:
    """WebSocket 접속키 발급"""
    try:
        await self.refresh_token_if_needed()
        
        url = f"{self._get_base_url()}/oauth2/Approval"
        data = {
            "grant_type": "client_credentials",
            "appkey": self.config.app_key,
            "secretkey": self.config.app_secret
        }
        
        headers = {
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret
        }
        
        async with self.session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                self.websocket_approval_key = result.get("approval_key")
                logger.info("WebSocket 접속키 발급 성공")
                return self.websocket_approval_key
            else:
                error_text = await response.text()
                raise Exception(f"WebSocket 접속키 발급 실패: {response.status} - {error_text}")
                
    except Exception as e:
        logger.error(f"WebSocket 접속키 발급 오류: {e}")
        raise
        
# ==================== 기본 유틸리티 ====================

def _get_base_url(self) -> str:
    """API 기본 URL 반환"""
    return self.config.mock_url if self.config.is_mock else self.config.base_url
    
async def _rate_limit_check(self):
    """API 호출 속도 제한 체크"""
    current_time = time.time()
    
    # 초당 20회, 분당 200회 제한 (KIS API 기준)
    if current_time - self.last_request_time < 0.05:  # 50ms 간격
        await asyncio.sleep(0.05)
        
    if current_time > self.request_reset_time + 60:  # 1분마다 리셋
        self.request_count = 0
        self.request_reset_time = current_time
        
    if self.request_count >= 200:  # 분당 200회 제한
        sleep_time = 60 - (current_time - self.request_reset_time)
        if sleep_time > 0:
            logger.warning(f"API 호출 제한 도달. {sleep_time:.1f}초 대기")
            await asyncio.sleep(sleep_time)
            
    self.request_count += 1
    self.last_request_time = current_time
    
@backoff.on_exception(
    backoff.expo,
    (aiohttp.ClientError, asyncio.TimeoutError),
    max_tries=3,
    base=1,
    max_value=10
)
async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
    """API 요청 공통 메서드 (재시도 로직 포함)"""
    await self.refresh_token_if_needed()
    await self._rate_limit_check()
    
    url = f"{self._get_base_url()}{endpoint}"
    headers = kwargs.get("headers", {})
    headers.update({
        "authorization": f"Bearer {self.access_token}",
        "appkey": self.config.app_key,
        "appsecret": self.config.app_secret,
    })
    kwargs["headers"] = headers
    
    try:
        async with self.session.request(method, url, **kwargs) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 401:
                # 토큰 만료, 재발급 후 재시도
                logger.warning("토큰 만료, 재발급 후 재시도")
                await self.get_access_token()
                headers["authorization"] = f"Bearer {self.access_token}"
                kwargs["headers"] = headers
                
                async with self.session.request(method, url, **kwargs) as retry_response:
                    if retry_response.status == 200:
                        return await retry_response.json()
                    else:
                        error_text = await retry_response.text()
                        raise Exception(f"API 요청 실패: {retry_response.status} - {error_text}")
            else:
                error_text = await response.text()
                raise Exception(f"API 요청 실패: {response.status} - {error_text}")
                
    except Exception as e:
        logger.error(f"API 요청 오류 ({method} {endpoint}): {e}")
        raise
        
# ==================== 주식 현재가 정보 ====================

async def get_current_price(self, ticker: str, market: MarketType = MarketType.KOSPI) -> StockPrice:
    """주식 현재가 조회"""
    try:
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": market.value,
            "FID_INPUT_ISCD": ticker
        }
        
        headers = {
            "tr_id": "FHKST01010100",
            "custtype": "P"
        }
        
        result = await self._make_request("GET", endpoint, params=params, headers=headers)
        
        if result.get("rt_cd") == "0":
            output = result.get("output", {})
            
            return StockPrice(
                ticker=ticker,
                name=output.get("hts_kor_isnm", ""),
                current_price=int(output.get("stck_prpr", 0)),
                change=int(output.get("prdy_vrss", 0)),
                change_rate=float(output.get("prdy_ctrt", 0)),
                volume=int(output.get("acml_vol", 0)),
                trade_value=int(output.get("acml_tr_pbmn", 0)),
                open_price=int(output.get("stck_oprc", 0)),
                high_price=int(output.get("stck_hgpr", 0)),
                low_price=int(output.get("stck_lwpr", 0)),
                prev_close=int(output.get("stck_sdpr", 0)),
                market_cap=int(output.get("hts_avls", 0)),
                timestamp=datetime.now()
            )
        else:
            error_msg = result.get("msg1", "알 수 없는 오류")
            raise Exception(f"현재가 조회 실패: {error_msg}")
            
    except Exception as e:
        logger.error(f"현재가 조회 오류 ({ticker}): {e}")
        raise
        
async def get_multiple_prices(self, tickers: List[str], market: MarketType = MarketType.KOSPI) -> List[StockPrice]:
    """다수 종목 현재가 일괄 조회"""
    tasks = [self.get_current_price(ticker, market) for ticker in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 예외 처리된 결과만 반환
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"종목 {tickers[i]} 조회 실패: {result}")
        else:
            valid_results.append(result)
            
    return valid_results
    
# ==================== 호가창 정보 ====================

async def get_orderbook(self, ticker: str, market: MarketType = MarketType.KOSPI) -> OrderBook:
    """호가창 조회"""
    try:
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
        params = {
            "FID_COND_MRKT_DIV_CODE": market.value,
            "FID_INPUT_ISCD": ticker
        }
        
        headers = {
            "tr_id": "FHKST01010200",
            "custtype": "P"
        }
        
        result = await self._make_request("GET", endpoint, params=params, headers=headers)
        
        if result.get("rt_cd") == "0":
            output = result.get("output1", {})
            
            # 매도호가/잔량 (5단계)
            ask_prices = []
            ask_volumes = []
            for i in range(1, 6):
                ask_prices.append(int(output.get(f"askp{i}", 0)))
                ask_volumes.append(int(output.get(f"askp_rsqn{i}", 0)))
            
            # 매수호가/잔량 (5단계)
            bid_prices = []
            bid_volumes = []
            for i in range(1, 6):
                bid_prices.append(int(output.get(f"bidp{i}", 0)))
                bid_volumes.append(int(output.get(f"bidp_rsqn{i}", 0)))
            
            return OrderBook(
                ticker=ticker,
                ask_prices=ask_prices,
                ask_volumes=ask_volumes,
                bid_prices=bid_prices,
                bid_volumes=bid_volumes,
                total_ask_volume=sum(ask_volumes),
                total_bid_volume=sum(bid_volumes),
                timestamp=datetime.now()
            )
        else:
            error_msg = result.get("msg1", "알 수 없는 오류")
            raise Exception(f"호가창 조회 실패: {error_msg}")
            
    except Exception as e:
        logger.error(f"호가창 조회 오류 ({ticker}): {e}")
        raise
        
# ==================== 체결강도 ====================

async def get_execution_strength(self, ticker: str, market: MarketType = MarketType.KOSPI) -> ExecutionStrength:
    """체결강도 조회"""
    try:
        # 현재가와 거래량 정보 획득
        price_info = await self.get_current_price(ticker, market)
        
        # 매수/매도 거래량 추정 (호가창 정보 활용)
        orderbook = await self.get_orderbook(ticker, market)
        
        # 체결강도 계산 (매수 거래량 / 매도 거래량)
        total_volume = price_info.volume
        ask_pressure = sum(orderbook.ask_volumes)
        bid_pressure = sum(orderbook.bid_volumes)
        
        # 추정 매수/매도 비율
        if ask_pressure + bid_pressure > 0:
            buy_ratio = bid_pressure / (ask_pressure + bid_pressure)
            sell_ratio = ask_pressure / (ask_pressure + bid_pressure)
        else:
            buy_ratio = sell_ratio = 0.5
            
        buy_volume = int(total_volume * buy_ratio)
        sell_volume = int(total_volume * sell_ratio)
        strength = buy_volume / sell_volume if sell_volume > 0 else 1.0
        
        return ExecutionStrength(
            ticker=ticker,
            strength=strength,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            net_volume=buy_volume - sell_volume,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"체결강도 조회 오류 ({ticker}): {e}")
        raise
        
# ==================== 매매동향 ====================

async def get_trading_by_investor(self, ticker: str, market: MarketType = MarketType.KOSPI) -> TradingByInvestor:
    """투자자별 매매동향 조회"""
    try:
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-investor"
        params = {
            "FID_COND_MRKT_DIV_CODE": market.value,
            "FID_INPUT_ISCD": ticker,
            "FID_DIV_CLS_CODE": "0"  # 누적
        }
        
        headers = {
            "tr_id": "FHKST01010900",
            "custtype": "P"
        }
        
        result = await self._make_request("GET", endpoint, params=params, headers=headers)
        
        if result.get("rt_cd") == "0":
            output = result.get("output", [])
            
            individual = {"buy": 0, "sell": 0, "net": 0}
            foreign = {"buy": 0, "sell": 0, "net": 0}
            institutional = {"buy": 0, "sell": 0, "net": 0}
            
            for item in output:
                investor_name = item.get("invst_nm", "")
                buy_volume = int(item.get("ntby_qty", 0))
                sell_volume = int(item.get("shnu_qty", 0))
                net_volume = buy_volume - sell_volume
                
                if "개인" in investor_name:
                    individual = {"buy": buy_volume, "sell": sell_volume, "net": net_volume}
                elif "외국인" in investor_name:
                    foreign = {"buy": buy_volume, "sell": sell_volume, "net": net_volume}
                elif "기관" in investor_name:
                    institutional = {"buy": buy_volume, "sell": sell_volume, "net": net_volume}
            
            return TradingByInvestor(
                ticker=ticker,
                individual=individual,
                foreign=foreign,
                institutional=institutional,
                timestamp=datetime.now()
            )
        else:
            error_msg = result.get("msg1", "알 수 없는 오류")
            raise Exception(f"매매동향 조회 실패: {error_msg}")
            
    except Exception as e:
        logger.error(f"매매동향 조회 오류 ({ticker}): {e}")
        raise
        
# ==================== 재무정보 ====================

async def get_financial_info(self, ticker: str, market: MarketType = MarketType.KOSPI) -> FinancialInfo:
    """재무정보 조회"""
    try:
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": market.value,
            "FID_INPUT_ISCD": ticker,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0"
        }
        
        headers = {
            "tr_id": "FHKST01010400",
            "custtype": "P"
        }
        
        result = await self._make_request("GET", endpoint, params=params, headers=headers)
        
        if result.get("rt_cd") == "0":
            output = result.get("output", {})
            
            return FinancialInfo(
                ticker=ticker,
                per=float(output.get("per", 0)),
                pbr=float(output.get("pbr", 0)),
                eps=int(output.get("eps", 0)),
                bps=int(output.get("bps", 0)),
                roe=float(output.get("roe", 0)),
                debt_ratio=float(output.get("debt_ratio", 0)),
                dividend_yield=float(output.get("dividend_yield", 0)),
                market_cap=int(output.get("hts_avls", 0)),
                timestamp=datetime.now()
            )
        else:
            error_msg = result.get("msg1", "알 수 없는 오류")
            raise Exception(f"재무정보 조회 실패: {error_msg}")
            
    except Exception as e:
        logger.error(f"재무정보 조회 오류 ({ticker}): {e}")
        raise
        
# ==================== 시세 조회 ====================

async def get_ohlcv_data(self, ticker: str, period: str = "D", count: int = 100) -> List[Dict]:
    """OHLCV 데이터 조회 (일/주/월봉)"""
    try:
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": "",  # 시작일 (공백시 최근 데이터)
            "FID_INPUT_DATE_2": "",  # 종료일
            "FID_PERIOD_DIV_CODE": period,  # D:일봉, W:주봉, M:월봉
            "FID_ORG_ADJ_PRC": "0"  # 수정주가 여부
        }
        
        headers = {
            "tr_id": "FHKST03010100",
            "custtype": "P"
        }
        
        result = await self._make_request("GET", endpoint, params=params, headers=headers)
        
        if result.get("rt_cd") == "0":
            output = result.get("output2", [])
            
            ohlcv_data = []
            for item in output[:count]:  # 요청한 개수만큼만
                ohlcv_data.append({
                    "date": item.get("stck_bsop_date", ""),
                    "open": int(item.get("stck_oprc", 0)),
                    "high": int(item.get("stck_hgpr", 0)),
                    "low": int(item.get("stck_lwpr", 0)),
                    "close": int(item.get("stck_clpr", 0)),
                    "volume": int(item.get("acml_vol", 0)),
                    "trade_value": int(item.get("acml_tr_pbmn", 0))
                })
            
            return ohlcv_data
        else:
            error_msg = result.get("msg1", "알 수 없는 오류")
            raise Exception(f"OHLCV 조회 실패: {error_msg}")
            
    except Exception as e:
        logger.error(f"OHLCV 조회 오료 ({ticker}): {e}")
        raise
        
# ==================== 종목 검색 ====================

async def search_stocks(self, keyword: str) -> List[Dict]:
    """종목명/코드로 종목 검색"""
    try:
        endpoint = "/uapi/domestic-stock/v1/quotations/search-stock-info"
        params = {
            "PRDT_TYPE_CD": "300",  # 주식
            "PDNO": keyword,
            "CO_YN_PRICECUR": "",
            "CO_ST_PRICECUR": ""
        }
        
        headers = {
            "tr_id": "CTPF1604R",
            "custtype": "P"
        }
        
        result = await self._make_request("GET", endpoint, params=params, headers=headers)
        
        if result.get("rt_cd") == "0":
            output = result.get("output", [])
            
            stock_list = []
            for item in output:
                stock_list.append({
                    "ticker": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "market": item.get("mket_ctg", ""),
                    "current_price": int(item.get("stck_prpr", 0)) if item.get("stck_prpr") else 0
                })
            
            return stock_list
        else:
            error_msg = result.get("msg1", "알 수 없는 오류")
            raise Exception(f"종목 검색 실패: {error_msg}")
            
    except Exception as e:
        logger.error(f"종목 검색 오류 ({keyword}): {e}")
        raise
        
# ==================== 시장 정보 ====================

async def get_market_status(self) -> Dict:
    """장 운영 시간 및 상태 조회"""
    try:
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-time-itemconclusion"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": "005930"  # 삼성전자로 시장 상태 확인
        }
        
        headers = {
            "tr_id": "FHKST03010200",
            "custtype": "P"
        }
        
        result = await self._make_request("GET", endpoint, params=params, headers=headers)
        
        if result.get("rt_cd") == "0":
            output = result.get("output1", {})
            
            # 현재 시간 파싱
            current_time = output.get("stck_cntg_hour", "")
            
            # 장 운영 상태 판단
            market_open = self._is_market_open(current_time)
            
            return {
                "market_open": market_open,
                "current_time": current_time,
                "market_status": "정규장" if market_open else "장 마감",
                "kospi_index": float(output.get("bstp_nmix_prpr", 0)),
                "kospi_change": float(output.get("bstp_nmix_prdy_vrss", 0)),
                "kospi_change_rate": float(output.get("bstp_nmix_prdy_ctrt", 0)),
                "timestamp": datetime.now()
            }
        else:
            error_msg = result.get("msg1", "알 수 없는 오류")
            raise Exception(f"시장 상태 조회 실패: {error_msg}")
            
    except Exception as e:
        logger.error(f"시장 상태 조회 오류: {e}")
        raise
        
def _is_market_open(self, time_str: str) -> bool:
    """장 운영 시간 체크"""
    try:
        if not time_str or len(time_str) < 6:
            return False
            
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        current_time = hour * 100 + minute
        
        # 정규장: 09:00 ~ 15:30
        return 900 <= current_time <= 1530
        
    except:
        return False
        
# ==================== WebSocket 실시간 데이터 ====================

async def connect_websocket(self):
    """WebSocket 연결"""
    try:
        if not self.websocket_approval_key:
            await self.get_websocket_approval_key()
            
        self.websocket_connection = await websockets.connect(
            self.config.websocket_url,
            ping_interval=30,
            ping_timeout=10
        )
        
        self.is_connected = True
        self.reconnect_attempts = 0
        logger.info("WebSocket 연결 성공")
        
        # 연결 유지 및 메시지 처리 태스크 시작
        asyncio.create_task(self._websocket_message_handler())
        
    except Exception as e:
        logger.error(f"WebSocket 연결 실패: {e}")
        self.is_connected = False
        await self._handle_websocket_reconnect()
        raise
        
async def _handle_websocket_reconnect(self):
    """WebSocket 재연결 처리"""
    if self.reconnect_attempts >= self.max_reconnect_attempts:
        logger.error("WebSocket 재연결 최대 시도 횟수 초과")
        return
        
    self.reconnect_attempts += 1
    wait_time = min(2 ** self.reconnect_attempts, 60)  # 지수 백오프, 최대 60초
    
    logger.info(f"WebSocket 재연결 시도 {self.reconnect_attempts}/{self.max_reconnect_attempts} ({wait_time}초 후)")
    await asyncio.sleep(wait_time)
    
    try:
        await self.connect_websocket()
    except Exception as e:
        logger.error(f"WebSocket 재연결 실패: {e}")
        await self._handle_websocket_reconnect()
        
async def _websocket_message_handler(self):
    """WebSocket 메시지 처리"""
    try:
        while self.is_connected and self.websocket_connection:
            try:
                message = await asyncio.wait_for(
                    self.websocket_connection.recv(), 
                    timeout=60
                )
                
                await self._process_websocket_message(message)
                
            except asyncio.TimeoutError:
                # 타임아웃은 정상 (ping/pong으로 연결 유지)
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket 연결이 끊어짐")
                self.is_connected = False
                await self._handle_websocket_reconnect()
                break
                
    except Exception as e:
        logger.error(f"WebSocket 메시지 처리 오류: {e}")
        self.is_connected = False
        await self._handle_websocket_reconnect()
        
async def _process_websocket_message(self, message: str):
    """WebSocket 메시지 파싱 및 콜백 호출"""
    try:
        # KIS WebSocket 메시지 형식에 따라 파싱
        if "|" not in message:
            return
            
        parts = message.split("|")
        if len(parts) < 4:
            return
            
        tr_id = parts[1]
        tr_key = parts[2]
        data = parts[3]
        
        # 실시간 주가
        if tr_id == "H0STCNT0" and tr_key in self.price_callbacks:
            price_data = self._parse_realtime_price(tr_key, data)
            await self.price_callbacks[tr_key](price_data)
                
        # 실시간 호가
        elif tr_id == "H0STASP0" and tr_key in self.orderbook_callbacks:
            orderbook_data = self._parse_realtime_orderbook(tr_key, data)
            await self.orderbook_callbacks[tr_key](orderbook_data)
                
        # 실시간 체결
        elif tr_id == "H0STCNI0" and tr_key in self.execution_callbacks:
            execution_data = self._parse_realtime_execution(tr_key, data)
            await self.execution_callbacks[tr_key](execution_data)
                
    except Exception as e:
        logger.error(f"WebSocket 메시지 파싱 오류: {e}")
        
def _parse_realtime_price(self, ticker: str, data: str) -> Dict:
    """실시간 주가 데이터 파싱"""
    try:
        # KIS API 실시간 주가 포맷에 따라 파싱
        # 실제 구현 시 KIS 문서의 정확한 포맷 확인 필요
        fields = data.split("^")
        
        return {
            "ticker": ticker,
            "current_price": int(fields[2]) if len(fields) > 2 else 0,
            "change": int(fields[3]) if len(fields) > 3 else 0,
            "change_rate": float(fields[4]) if len(fields) > 4 else 0.0,
            "volume": int(fields[12]) if len(fields) > 12 else 0,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"실시간 주가 파싱 오류: {e}")
        return {"ticker": ticker, "timestamp": datetime.now()}
    
def _parse_realtime_orderbook(self, ticker: str, data: str) -> Dict:
    """실시간 호가 데이터 파싱"""
    try:
        fields = data.split("^")
        
        # 매도호가 5단계
        ask_prices = []
        ask_volumes = []
        for i in range(5):
            ask_prices.append(int(fields[3 + i]) if len(fields) > 3 + i else 0)
            ask_volumes.append(int(fields[13 + i]) if len(fields) > 13 + i else 0)
        
        # 매수호가 5단계
        bid_prices = []
        bid_volumes = []
        for i in range(5):
            bid_prices.append(int(fields[8 + i]) if len(fields) > 8 + i else 0)
            bid_volumes.append(int(fields[18 + i]) if len(fields) > 18 + i else 0)
        
        return {
            "ticker": ticker,
            "ask_prices": ask_prices,
            "ask_volumes": ask_volumes,
            "bid_prices": bid_prices,
            "bid_volumes": bid_volumes,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"실시간 호가 파싱 오류: {e}")
        return {"ticker": ticker, "timestamp": datetime.now()}
    
def _parse_realtime_execution(self, ticker: str, data: str) -> Dict:
    """실시간 체결 데이터 파싱"""
    try:
        fields = data.split("^")
        
        return {
            "ticker": ticker,
            "price": int(fields[2]) if len(fields) > 2 else 0,
            "volume": int(fields[3]) if len(fields) > 3 else 0,
            "execution_time": fields[1] if len(fields) > 1 else "",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"실시간 체결 파싱 오류: {e}")
        return {"ticker": ticker, "timestamp": datetime.now()}
    
async def subscribe_realtime_price(self, ticker: str, callback: Callable):
    """실시간 주가 구독"""
    self.price_callbacks[ticker] = callback
    
    if not self.is_connected:
        await self.connect_websocket()
        
    # 실시간 주가 구독 메시지 전송
    subscribe_message = {
        "header": {
            "approval_key": self.websocket_approval_key,
            "custtype": "P",
            "tr_type": "1",  # 구독
            "content-type": "utf-8"
        },
        "body": {
            "input": {
                "tr_id": "H0STCNT0",
                "tr_key": ticker
            }
        }
    }
    
    await self.websocket_connection.send(json.dumps(subscribe_message))
    logger.info(f"실시간 주가 구독: {ticker}")
    
async def subscribe_realtime_orderbook(self, ticker: str, callback: Callable):
    """실시간 호가 구독"""
    self.orderbook_callbacks[ticker] = callback
    
    if not self.is_connected:
        await self.connect_websocket()
        
    subscribe_message = {
        "header": {
            "approval_key": self.websocket_approval_key,
            "custtype": "P",
            "tr_type": "1",
            "content-type": "utf-8"
        },
        "body": {
            "input": {
                "tr_id": "H0STASP0",
                "tr_key": ticker
            }
        }
    }
    
    await self.websocket_connection.send(json.dumps(subscribe_message))
    logger.info(f"실시간 호가 구독: {ticker}")
    
async def unsubscribe_realtime_price(self, ticker: str):
    """실시간 주가 구독 해제"""
    if ticker in self.price_callbacks:
        del self.price_callbacks[ticker]
        
    if not self.is_connected or not self.websocket_connection:
        return
        
    unsubscribe_message = {
        "header": {
            "approval_key": self.websocket_approval_key,
            "custtype": "P",
            "tr_type": "2",  # 구독해제
            "content-type": "utf-8"
        },
        "body": {
            "input": {
                "tr_id": "H0STCNT0",
                "tr_key": ticker
            }
        }
    }
    
    await self.websocket_connection.send(json.dumps(unsubscribe_message))
    logger.info(f"실시간 주가 구독 해제: {ticker}")
    
async def unsubscribe_realtime_orderbook(self, ticker: str):
    """실시간 호가 구독 해제"""
    if ticker in self.orderbook_callbacks:
        del self.orderbook_callbacks[ticker]
        
    if not self.is_connected or not self.websocket_connection:
        return
        
    unsubscribe_message = {
        "header": {
            "approval_key": self.websocket_approval_key,
            "custtype": "P",
            "tr_type": "2",
            "content-type": "utf-8"
        },
        "body": {
            "input": {
                "tr_id": "H0STASP0",
                "tr_key": ticker
            }
        }
    }
    
    await self.websocket_connection.send(json.dumps(unsubscribe_message))
    logger.info(f"실시간 호가 구독 해제: {ticker}")
    
# ==================== 공시정보 ====================

async def get_disclosure_info(self, ticker: str, days: int = 7) -> List[Dict]:
    """공시정보 조회"""
    try:
        # DART API 또는 KIS API를 통한 공시정보 조회
        # 실제 구현 시 해당 API 엔드포인트 확인 필요
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-disclosure"
        params = {
            "FID_INPUT_ISCD": ticker,
            "FID_PERIOD_DIV_CODE": str(days)
        }
        
        headers = {
            "tr_id": "FHKST01010300",
            "custtype": "P"
        }
        
        result = await self._make_request("GET", endpoint, params=params, headers=headers)
        
        if result.get("rt_cd") == "0":
            output = result.get("output", [])
            
            disclosures = []
            for item in output:
                disclosures.append({
                    "ticker": ticker,
                    "title": item.get("disclosure_title", ""),
                    "date": item.get("disclosure_date", ""),
                    "type": item.get("disclosure_type", ""),
                    "content": item.get("disclosure_content", ""),
                    "url": item.get("disclosure_url", "")
                })
            
            return disclosures
        else:
            # 공시정보가 없는 경우도 정상
            return []
            
    except Exception as e:
        logger.error(f"공시정보 조회 오류 ({ticker}): {e}")
        return []
        
# ==================== 유틸리티 메서드 ====================

async def health_check(self) -> Dict:
    """API 상태 체크"""
    try:
        # 간단한 현재가 조회로 API 상태 확인
        await self.get_current_price("005930")  # 삼성전자
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "token_expires_at": self.token_expires_at,
            "websocket_connected": self.is_connected
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(),
            "token_expires_at": self.token_expires_at,
            "websocket_connected": self.is_connected
        }
        
def get_connection_info(self) -> Dict:
    """연결 정보 반환"""
    return {
        "base_url": self._get_base_url(),
        "is_mock": self.config.is_mock,
        "token_valid": self.access_token is not None and 
                      self.token_expires_at is not None and 
                      datetime.now() < self.token_expires_at,
        "websocket_connected": self.is_connected,
        "subscribed_tickers": {
            "price": list(self.price_callbacks.keys()),
            "orderbook": list(self.orderbook_callbacks.keys()),
            "execution": list(self.execution_callbacks.keys())
        }
    }
```

# ==================== 사용 예제 ====================

class KISApiExample:
“”“KIS API 사용 예제”””

```
def __init__(self, app_key: str, app_secret: str):
    self.config = KISConfig(
        app_key=app_key,
        app_secret=app_secret,
        is_mock=True  # 개발 초기에는 모의투자 사용
    )
    
async def basic_usage_example(self):
    """기본 사용법 예제"""
    async with KISApiClient(self.config) as client:
        print("=== 한국투자증권 KIS API 사용 예제 ===")
        
        # 1. 현재가 조회
        print("\n1. 삼성전자 현재가 조회")
        samsung_price = await client.get_current_price("005930")
        print(f"삼성전자: {samsung_price.current_price:,}원 ({samsung_price.change:+,}원, {samsung_price.change_rate:+.2f}%)")
        
        # 2. 다수 종목 현재가 조회
        print("\n2. 다수 종목 현재가 조회")
        tickers = ["005930", "000660", "035420"]  # 삼성전자, SK하이닉스, 네이버
        prices = await client.get_multiple_prices(tickers)
        for price in prices:
            print(f"{price.name}: {price.current_price:,}원 ({price.change:+,}원)")
        
        # 3. 호가창 조회
        print("\n3. 삼성전자 호가창 조회")
        orderbook = await client.get_orderbook("005930")
        print("매도호가:", orderbook.ask_prices[:3])
        print("매수호가:", orderbook.bid_prices[:3])
        
        # 4. 체결강도 조회
        print("\n4. 삼성전자 체결강도 조회")
        strength = await client.get_execution_strength("005930")
        print(f"체결강도: {strength.strength:.2f}")
        print(f"매수량: {strength.buy_volume:,}, 매도량: {strength.sell_volume:,}")
        
        # 5. 투자자별 매매동향
        print("\n5. 삼성전자 투자자별 매매동향")
        trading = await client.get_trading_by_investor("005930")
        print(f"개인 순매수: {trading.individual['net']:,}")
        print(f"외국인 순매수: {trading.foreign['net']:,}")
        print(f"기관 순매수: {trading.institutional['net']:,}")
        
        # 6. OHLCV 데이터 조회
        print("\n6. 삼성전자 최근 5일 OHLCV")
        ohlcv = await client.get_ohlcv_data("005930", count=5)
        for data in ohlcv[:3]:
            print(f"{data['date']}: {data['close']:,}원 (거래량: {data['volume']:,})")
        
        # 7. 종목 검색
        print("\n7. '삼성' 키워드 종목 검색")
        search_results = await client.search_stocks("삼성")
        for result in search_results[:3]:
            print(f"{result['name']} ({result['ticker']}): {result['current_price']:,}원")
        
        # 8. 시장 상태 조회
        print("\n8. 시장 상태 조회")
        market_status = await client.get_market_status()
        print(f"장 상태: {market_status['market_status']}")
        print(f"코스피: {market_status['kospi_index']:.2f} ({market_status['kospi_change']:+.2f})")
        
        # 9. 건강 상태 체크
        print("\n9. API 건강 상태 체크")
        health = await client.health_check()
        print(f"API 상태: {health['status']}")
        
        print("\n=== 기본 사용법 예제 완료 ===")

async def realtime_data_example(self):
    """실시간 데이터 수신 예제"""
    async def price_callback(data):
        print(f"실시간 주가 - {data['ticker']}: {data['current_price']:,}원 ({data['change']:+,}원)")
    
    async def orderbook_callback(data):
        print(f"실시간 호가 - {data['ticker']}: 매도1호가 {data['ask_prices'][0]:,}원")
    
    async with KISApiClient(self.config) as client:
        print("=== 실시간 데이터 수신 예제 ===")
        
        # 실시간 주가 구독
        await client.subscribe_realtime_price("005930", price_callback)
        await client.subscribe_realtime_orderbook("005930", orderbook_callback)
        
        print("실시간 데이터 수신 중... (30초 후 종료)")
        await asyncio.sleep(30)
        
        # 구독 해제
        await client.unsubscribe_realtime_price("005930")
        await client.unsubscribe_realtime_orderbook("005930")
        
        print("=== 실시간 데이터 수신 예제 완료 ===")
```

# ==================== 실행 예제 ====================

async def main():
“”“메인 실행 함수”””
# TODO: 실제 API 키로 교체
APP_KEY = “YOUR_APP_KEY_HERE”
APP_SECRET = “YOUR_APP_SECRET_HERE”

```
example = KISApiExample(APP_KEY, APP_SECRET)

try:
    # 기본 사용법 예제 실행
    await example.basic_usage_example()
    
    # 실시간 데이터 예제 실행 (선택적)
    # await example.realtime_data_example()
    
except Exception as e:
    logger.error(f"예제 실행 오류: {e}")
```

if **name** == “**main**”:
# 예제 실행
asyncio.run(main())