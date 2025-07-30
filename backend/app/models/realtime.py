# backend/app/models/realtime.py

“””
실시간 데이터 SQLAlchemy 모델

- 실시간 주가 데이터
- 실시간 호가 데이터
- TimescaleDB 하이퍼테이블 지원
  “””

from sqlalchemy import Column, String, Integer, Float, DateTime, ARRAY, Index
from sqlalchemy.dialects.postgresql import TIMESTAMP
from datetime import datetime

from .base import Base

class RealtimePriceData(Base):
“”“실시간 주가 데이터”””
**tablename** = ‘realtime_price_data’

```
# 기본 정보
ticker = Column(String(10), nullable=False, primary_key=True)
timestamp = Column(TIMESTAMP(timezone=True), nullable=False, primary_key=True)
received_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.utcnow)

# 주가 정보
current_price = Column(Integer, nullable=False)
change = Column(Integer, nullable=True)
change_rate = Column(Float, nullable=True)
volume = Column(Integer, nullable=True)

# 추가 정보
trade_value = Column(Integer, nullable=True)

# 인덱스
__table_args__ = (
    Index('idx_realtime_price_ticker_time', 'ticker', 'timestamp'),
    Index('idx_realtime_price_received_at', 'received_at'),
    {'timescaledb_hypertable': {
        'time_column_name': 'timestamp',
        'chunk_time_interval': '1 hour'
    }}
)
```

class RealtimeOrderbookData(Base):
“”“실시간 호가 데이터”””
**tablename** = ‘realtime_orderbook_data’

```
# 기본 정보
ticker = Column(String(10), nullable=False, primary_key=True)
timestamp = Column(TIMESTAMP(timezone=True), nullable=False, primary_key=True)
received_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.utcnow)

# 매도호가 (5단계)
ask_prices = Column(ARRAY(Integer), nullable=True)
ask_volumes = Column(ARRAY(Integer), nullable=True)

# 매수호가 (5단계)
bid_prices = Column(ARRAY(Integer), nullable=True)
bid_volumes = Column(ARRAY(Integer), nullable=True)

# 총 잔량
total_ask_volume = Column(Integer, nullable=True)
total_bid_volume = Column(Integer, nullable=True)

# 인덱스
__table_args__ = (
    Index('idx_realtime_orderbook_ticker_time', 'ticker', 'timestamp'),
    Index('idx_realtime_orderbook_received_at', 'received_at'),
    {'timescaledb_hypertable': {
        'time_column_name': 'timestamp', 
        'chunk_time_interval': '1 hour'
    }}
)
```

class RealtimeExecutionData(Base):
“”“실시간 체결 데이터”””
**tablename** = ‘realtime_execution_data’

```
# 기본 정보
ticker = Column(String(10), nullable=False, primary_key=True)
timestamp = Column(TIMESTAMP(timezone=True), nullable=False, primary_key=True)
received_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.utcnow)

# 체결 정보
execution_price = Column(Integer, nullable=False)
execution_volume = Column(Integer, nullable=False)
execution_time = Column(String(20), nullable=True)  # 체결시간 문자열

# 체결 구분 (매수/매도)
execution_type = Column(String(10), nullable=True)

# 인덱스
__table_args__ = (
    Index('idx_realtime_execution_ticker_time', 'ticker', 'timestamp'),
    Index('idx_realtime_execution_received_at', 'received_at'),
    {'timescaledb_hypertable': {
        'time_column_name': 'timestamp',
        'chunk_time_interval': '1 hour'
    }}
)
```

class RealtimeCollectorStats(Base):
“”“실시간 수집기 통계”””
**tablename** = ‘realtime_collector_stats’

```
# 기본 정보
timestamp = Column(TIMESTAMP(timezone=True), nullable=False, primary_key=True, default=datetime.utcnow)

# 연결 통계
active_connections = Column(Integer, nullable=False, default=0)
subscribed_tickers = Column(Integer, nullable=False, default=0)

# 성능 통계
messages_per_second = Column(Float, nullable=True)
queue_size = Column(Integer, nullable=True)
batch_buffer_size = Column(Integer, nullable=True)

# 에러 통계
total_errors = Column(Integer, nullable=False, default=0)
reconnection_count = Column(Integer, nullable=False, default=0)

# 인덱스
__table_args__ = (
    Index('idx_collector_stats_timestamp', 'timestamp'),
    {'timescaledb_hypertable': {
        'time_column_name': 'timestamp',
        'chunk_time_interval': '1 day'
    }}
)
```