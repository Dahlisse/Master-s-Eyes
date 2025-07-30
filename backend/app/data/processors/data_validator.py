# backend/app/data/processors/data_validator.py

“””
실시간 데이터 검증기

- 주가 데이터 유효성 검증
- 호가창 데이터 검증
- 이상치 탐지 및 필터링
  “””

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger(**name**)

class RealtimeDataValidator:
“”“실시간 데이터 검증기”””

```
def __init__(self):
    # 유효성 검증 규칙
    self.price_rules = {
        'min_price': 100,           # 최소 주가 (100원)
        'max_price': 10000000,      # 최대 주가 (1천만원)
        'max_change_rate': 30.0,    # 최대 등락률 (30%)
        'min_volume': 0,            # 최소 거래량
        'max_volume': 999999999     # 최대 거래량
    }
    
    self.orderbook_rules = {
        'max_spread_rate': 10.0,    # 최대 호가 스프레드 (10%)
        'min_volume': 0,            # 최소 호가 잔량
        'max_volume': 999999999     # 최대 호가 잔량
    }
    
    # 이상치 탐지를 위한 히스토리
    self.price_history: Dict[str, List[int]] = {}
    self.history_size = 10
    
def validate_price_data(self, data: Dict[str, Any]) -> bool:
    """주가 데이터 검증"""
    try:
        # 필수 필드 확인
        required_fields = ['ticker', 'current_price', 'timestamp']
        for field in required_fields:
            if field not in data:
                logger.warning(f"주가 데이터 필수 필드 누락: {field}")
                return False
                
        ticker = data['ticker']
        current_price = data.get('current_price', 0)
        change_rate = data.get('change_rate', 0.0)
        volume = data.get('volume', 0)
        
        # 기본 범위 검증
        if not self._validate_price_range(current_price):
            return False
            
        if not self._validate_change_rate(change_rate):
            return False
            
        if not self._validate_volume(volume):
            return False
            
        # 티커 형식 검증
        if not self._validate_ticker_format(ticker):
            return False
            
        # 이상치 탐지
        if not self._validate_price_anomaly(ticker, current_price):
            return False
            
        # 타임스탬프 검증
        if not self._validate_timestamp(data.get('timestamp')):
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"주가 데이터 검증 오류: {e}")
        return False
        
def validate_orderbook_data(self, data: Dict[str, Any]) -> bool:
    """호가창 데이터 검증"""
    try:
        # 필수 필드 확인
        required_fields = ['ticker', 'ask_prices', 'bid_prices', 'timestamp']
        for field in required_fields:
            if field not in data:
                logger.warning(f"호가 데이터 필수 필드 누락: {field}")
                return False
                
        ticker = data['ticker']
        ask_prices = data.get('ask_prices', [])
        bid_prices = data.get('bid_prices', [])
        ask_volumes = data.get('ask_volumes', [])
        bid_volumes = data.get('bid_volumes', [])
        
        # 호가 개수 검증 (일반적으로 5단계)
        if len(ask_prices) != 5 or len(bid_prices) != 5:
            logger.warning(f"호가 단계 수 오류: ask={len(ask_prices)}, bid={len(bid_prices)}")
            return False
            
        # 호가 가격 검증
        for price in ask_prices + bid_prices:
            if not self._validate_price_range(price):
                return False
                
        # 호가 잔량 검증
        for volume in ask_volumes + bid_volumes:
            if not self._validate_orderbook_volume(volume):
                return False
                
        # 호가 순서 검증 (매도호가는 오름차순, 매수호가는 내림차순)
        if not self._validate_orderbook_sequence(ask_prices, bid_prices):
            return False
            
        # 스프레드 검증
        if ask_prices and bid_prices:
            if not self._validate_spread(ask_prices[0], bid_prices[0]):
                return False
                
        # 티커 형식 검증
        if not self._validate_ticker_format(ticker):
            return False
            
        # 타임스탬프 검증
        if not self._validate_timestamp(data.get('timestamp')):
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"호가 데이터 검증 오류: {e}")
        return False
        
def _validate_price_range(self, price: int) -> bool:
    """주가 범위 검증"""
    if not isinstance(price, (int, float)) or price <= 0:
        return False
        
    return (self.price_rules['min_price'] <= price <= self.price_rules['max_price'])
    
def _validate_change_rate(self, change_rate: float) -> bool:
    """등락률 검증"""
    if not isinstance(change_rate, (int, float)):
        return False
        
    return abs(change_rate) <= self.price_rules['max_change_rate']
    
def _validate_volume(self, volume: int) -> bool:
    """거래량 검증"""
    if not isinstance(volume, (int, float)) or volume < 0:
        return False
        
    return (self.price_rules['min_volume'] <= volume <= self.price_rules['max_volume'])
    
def _validate_orderbook_volume(self, volume: int) -> bool:
    """호가 잔량 검증"""
    if not isinstance(volume, (int, float)) or volume < 0:
        return False
        
    return (self.orderbook_rules['min_volume'] <= volume <= self.orderbook_rules['max_volume'])
    
def _validate_ticker_format(self, ticker: str) -> bool:
    """티커 형식 검증"""
    if not isinstance(ticker, str):
        return False
        
    # 한국 주식 티커는 6자리 숫자
    pattern = r'^\d{6}$'
    return bool(re.match(pattern, ticker))
    
def _validate_timestamp(self, timestamp: Any) -> bool:
    """타임스탬프 검증"""
    try:
        if isinstance(timestamp, datetime):
            # 현재 시간과의 차이가 1시간 이내
            time_diff = abs((datetime.now() - timestamp).total_seconds())
            return time_diff <= 3600  # 1시간
            
        return False
        
    except Exception:
        return False
        
def _validate_price_anomaly(self, ticker: str, current_price: int) -> bool:
    """주가 이상치 탐지"""
    try:
        # 가격 히스토리 업데이트
        if ticker not in self.price_history:
            self.price_history[ticker] = []
            
        history = self.price_history[ticker]
        history.append(current_price)
        
        # 히스토리 크기 제한
        if len(history) > self.history_size:
            history.pop(0)
            
        # 충분한 히스토리가 없으면 통과
        if len(history) < 3:
            return True
            
        # 이동평균 대비 급격한 변화 탐지
        avg_price = sum(history[:-1]) / len(history[:-1])
        change_rate = abs(current_price - avg_price) / avg_price * 100
        
        # 30% 이상 급변하면 이상치로 판단
        if change_rate > 30.0:
            logger.warning(f"주가 이상치 탐지: {ticker}, 현재가={current_price}, 평균가={avg_price:.0f}, 변화율={change_rate:.1f}%")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"이상치 탐지 오류 ({ticker}): {e}")
        return True  # 오류 시 통과
        
def _validate_orderbook_sequence(self, ask_prices: List[int], bid_prices: List[int]) -> bool:
    """호가 순서 검증"""
    try:
        # 매도호가는 오름차순이어야 함
        for i in range(len(ask_prices) - 1):
            if ask_prices[i] > ask_prices[i + 1] and ask_prices[i + 1] > 0:
                logger.warning(f"매도호가 순서 오류: {ask_prices}")
                return False
                
        # 매수호가는 내림차순이어야 함
        for i in range(len(bid_prices) - 1):
            if bid_prices[i] < bid_prices[i + 1] and bid_prices[i + 1] > 0:
                logger.warning(f"매수호가 순서 오류: {bid_prices}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"호가 순서 검증 오류: {e}")
        return False
        
def _validate_spread(self, ask_price: int, bid_price: int) -> bool:
    """호가 스프레드 검증"""
    try:
        if ask_price <= 0 or bid_price <= 0:
            return True  # 0인 경우는 스킵
            
        if ask_price <= bid_price:
            logger.warning(f"호가 역전: 매도={ask_price}, 매수={bid_price}")
            return False
            
        # 스프레드율 계산
        mid_price = (ask_price + bid_price) / 2
        spread_rate = (ask_price - bid_price) / mid_price * 100
        
        if spread_rate > self.orderbook_rules['max_spread_rate']:
            logger.warning(f"과도한 호가 스프레드: {spread_rate:.2f}%")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"스프레드 검증 오류: {e}")
        return False
        
def get_validation_stats(self) -> Dict[str, Any]:
    """검증 통계 반환"""
    return {
        'price_rules': self.price_rules,
        'orderbook_rules': self.orderbook_rules,
        'tracked_tickers': len(self.price_history),
        'history_size': self.history_size
    }
```