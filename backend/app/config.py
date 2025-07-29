“””
Master’s Eye - Configuration Management
환경 변수 및 설정 관리
“””

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
import os
from datetime import datetime, timezone
import secrets

class Settings(BaseSettings):
“”“애플리케이션 설정 클래스”””

```
# =================================
# Application Settings
# =================================
ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
DEBUG: bool = Field(default=True, env="DEBUG")
LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
ALLOWED_ORIGINS: List[str] = Field(default=["http://localhost:3000", "http://localhost:8000"], env="ALLOWED_ORIGINS")
SERVE_STATIC: bool = Field(default=True, env="SERVE_STATIC")

# =================================
# Database Configuration
# =================================
DATABASE_URL: str = Field(env="DATABASE_URL")
DB_HOST: str = Field(default="localhost", env="DB_HOST")
DB_PORT: int = Field(default=5432, env="DB_PORT")
DB_NAME: str = Field(default="masters_eye", env="DB_NAME")
DB_USER: str = Field(default="admin", env="DB_USER")
DB_PASSWORD: str = Field(env="DB_PASSWORD")
DB_POOL_SIZE: int = Field(default=10, env="DB_POOL_SIZE")
DB_MAX_OVERFLOW: int = Field(default=20, env="DB_MAX_OVERFLOW")

# =================================
# Redis Configuration
# =================================
REDIS_URL: str = Field(env="REDIS_URL")
REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
REDIS_DB: int = Field(default=0, env="REDIS_DB")
REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

# =================================
# Celery Configuration
# =================================
CELERY_BROKER_URL: str = Field(env="CELERY_BROKER_URL")
CELERY_RESULT_BACKEND: str = Field(env="CELERY_RESULT_BACKEND")
CELERY_TASK_SERIALIZER: str = Field(default="json", env="CELERY_TASK_SERIALIZER")
CELERY_RESULT_SERIALIZER: str = Field(default="json", env="CELERY_RESULT_SERIALIZER")

# =================================
# Korean Investment Securities API
# =================================
KIS_APP_KEY: str = Field(env="KIS_APP_KEY")
KIS_APP_SECRET: str = Field(env="KIS_APP_SECRET")
KIS_ACCOUNT_CODE: str = Field(env="KIS_ACCOUNT_CODE")
KIS_PRODUCT_CODE: str = Field(env="KIS_PRODUCT_CODE")
KIS_MOCK_MODE: bool = Field(default=True, env="KIS_MOCK_MODE")
KIS_BASE_URL: str = Field(
    default="https://openapivts.koreainvestment.com:29443",  # 모의투자
    env="KIS_BASE_URL"
)

@validator("KIS_BASE_URL")
def validate_kis_base_url(cls, v, values):
    """KIS API URL을 모의/실전에 따라 자동 설정"""
    mock_mode = values.get("KIS_MOCK_MODE", True)
    if mock_mode:
        return "https://openapivts.koreainvestment.com:29443"  # 모의투자
    else:
        return "https://openapi.koreainvestment.com:9443"      # 실전투자

# =================================
# Kiwoom Securities API
# =================================
KIWOOM_USER_ID: str = Field(env="KIWOOM_USER_ID")
KIWOOM_PASSWORD: str = Field(env="KIWOOM_PASSWORD")
KIWOOM_CERT_PASSWORD: str = Field(env="KIWOOM_CERT_PASSWORD")
KIWOOM_MOCK_MODE: bool = Field(default=True, env="KIWOOM_MOCK_MODE")
KIWOOM_AUTO_LOGIN: bool = Field(default=False, env="KIWOOM_AUTO_LOGIN")

# =================================
# Global Financial Data APIs
# =================================
YAHOO_FINANCE_ENABLED: bool = Field(default=True, env="YAHOO_FINANCE_ENABLED")
FRED_API_KEY: str = Field(env="FRED_API_KEY")
ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")

# =================================
# AI/LLM Configuration
# =================================
OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
OLLAMA_MODEL: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
OPENAI_MODEL: str = Field(default="gpt-4", env="OPENAI_MODEL")
AI_RESPONSE_TIMEOUT: int = Field(default=30, env="AI_RESPONSE_TIMEOUT")

# =================================
# Notification Services
# =================================
SMTP_HOST: str = Field(default="smtp.gmail.com", env="SMTP_HOST")
SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
SMTP_USER: Optional[str] = Field(default=None, env="SMTP_USER")
SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
SMTP_FROM_EMAIL: Optional[str] = Field(default=None, env="SMTP_FROM_EMAIL")

SENDGRID_API_KEY: Optional[str] = Field(default=None, env="SENDGRID_API_KEY")
KAKAO_REST_API_KEY: Optional[str] = Field(default=None, env="KAKAO_REST_API_KEY")
KAKAO_ADMIN_KEY: Optional[str] = Field(default=None, env="KAKAO_ADMIN_KEY")

# =================================
# Security & Authentication
# =================================
ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=1440, env="ACCESS_TOKEN_EXPIRE_MINUTES")
REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30, env="REFRESH_TOKEN_EXPIRE_DAYS")
PASSWORD_MIN_LENGTH: int = Field(default=8, env="PASSWORD_MIN_LENGTH")

# =================================
# Monitoring & Analytics
# =================================
SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
GRAFANA_ADMIN_PASSWORD: str = Field(default="masters_eye_2024", env="GRAFANA_ADMIN_PASSWORD")
ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")

# =================================
# Market Data Configuration
# =================================
MARKET_OPEN_TIME: str = Field(default="09:00", env="MARKET_OPEN_TIME")
MARKET_CLOSE_TIME: str = Field(default="15:30", env="MARKET_CLOSE_TIME")
PRE_MARKET_OPEN_TIME: str = Field(default="08:00", env="PRE_MARKET_OPEN_TIME")
AFTER_MARKET_CLOSE_TIME: str = Field(default="18:00", env="AFTER_MARKET_CLOSE_TIME")

REALTIME_DATA_INTERVAL: int = Field(default=1, env="REALTIME_DATA_INTERVAL")
PORTFOLIO_REBALANCE_INTERVAL: int = Field(default=300, env="PORTFOLIO_REBALANCE_INTERVAL")
NEWS_COLLECTION_INTERVAL: int = Field(default=60, env="NEWS_COLLECTION_INTERVAL")

# =================================
# Risk Management
# =================================
MAX_SINGLE_STOCK_WEIGHT: float = Field(default=0.10, env="MAX_SINGLE_STOCK_WEIGHT")
MAX_SECTOR_WEIGHT: float = Field(default=0.30, env="MAX_SECTOR_WEIGHT")
MAX_DAILY_LOSS_LIMIT: float = Field(default=0.05, env="MAX_DAILY_LOSS_LIMIT")
STOP_LOSS_THRESHOLD: float = Field(default=0.15, env="STOP_LOSS_THRESHOLD")

@validator("MAX_SINGLE_STOCK_WEIGHT", "MAX_SECTOR_WEIGHT", "MAX_DAILY_LOSS_LIMIT", "STOP_LOSS_THRESHOLD")
def validate_risk_percentages(cls, v):
    """리스크 관리 비율이 0-1 사이인지 확인"""
    if not 0 <= v <= 1:
        raise ValueError("Risk management percentages must be between 0 and 1")
    return v

# =================================
# 4 Masters Algorithm Weights
# =================================
BUFFETT_WEIGHT: float = Field(default=0.30, env="BUFFETT_WEIGHT")
DALIO_WEIGHT: float = Field(default=0.30, env="DALIO_WEIGHT")
FEYNMAN_WEIGHT: float = Field(default=0.20, env="FEYNMAN_WEIGHT")
SIMONS_WEIGHT: float = Field(default=0.20, env="SIMONS_WEIGHT")

@validator("BUFFETT_WEIGHT", "DALIO_WEIGHT", "FEYNMAN_WEIGHT", "SIMONS_WEIGHT")
def validate_master_weights(cls, v):
    """4대 거장 가중치가 0-1 사이인지 확인"""
    if not 0 <= v <= 1:
        raise ValueError("Master weights must be between 0 and 1")
    return v

# =================================
# Development & Testing
# =================================
PYTEST_WORKERS: int = Field(default=4, env="PYTEST_WORKERS")
TEST_DATABASE_URL: Optional[str] = Field(default=None, env="TEST_DATABASE_URL")
ENABLE_SQL_ECHO: bool = Field(default=False, env="ENABLE_SQL_ECHO")

# =================================
# News & Sentiment Analysis
# =================================
NEWS_SOURCES: List[str] = Field(default=["naver", "daum", "hankyung", "mk"], env="NEWS_SOURCES")
SENTIMENT_ANALYSIS_ENABLED: bool = Field(default=True, env="SENTIMENT_ANALYSIS_ENABLED")
NEWS_RETENTION_DAYS: int = Field(default=30, env="NEWS_RETENTION_DAYS")
MAX_NEWS_PER_DAY: int = Field(default=1000, env="MAX_NEWS_PER_DAY")

# =================================
# Backup & Data Retention
# =================================
DATA_BACKUP_ENABLED: bool = Field(default=True, env="DATA_BACKUP_ENABLED")
DATA_RETENTION_DAYS: int = Field(default=365, env="DATA_RETENTION_DAYS")
BACKUP_SCHEDULE: str = Field(default="0 2 * * *", env="BACKUP_SCHEDULE")  # Daily at 2 AM
BACKUP_STORAGE_PATH: str = Field(default="./backups", env="BACKUP_STORAGE_PATH")

# =================================
# Rate Limiting
# =================================
API_RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="API_RATE_LIMIT_PER_MINUTE")
WEBSOCKET_CONNECTION_LIMIT: int = Field(default=100, env="WEBSOCKET_CONNECTION_LIMIT")
MAX_CONCURRENT_REQUESTS: int = Field(default=50, env="MAX_CONCURRENT_REQUESTS")

# =================================
# Logging Configuration
# =================================
LOG_FILE_PATH: str = Field(default="./logs/masters_eye.log", env="LOG_FILE_PATH")
LOG_MAX_SIZE: str = Field(default="100MB", env="LOG_MAX_SIZE")
LOG_BACKUP_COUNT: int = Field(default=5, env="LOG_BACKUP_COUNT")
LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")

# =================================
# User Configuration
# =================================
DEFAULT_USER_ME: str = Field(default="me", env="DEFAULT_USER_ME")
DEFAULT_USER_MOM: str = Field(default="mom", env="DEFAULT_USER_MOM")
USER_SESSION_TIMEOUT: int = Field(default=3600, env="USER_SESSION_TIMEOUT")

# =================================
# Cache Configuration
# =================================
CACHE_TTL_SECONDS: int = Field(default=300, env="CACHE_TTL_SECONDS")  # 5 minutes
CACHE_MAX_SIZE: int = Field(default=1000, env="CACHE_MAX_SIZE")

class Config:
    env_file = ".env"
    env_file_encoding = "utf-8"
    case_sensitive = True
    extra = "ignore"

def get_current_time(self) -> str:
    """현재 시간을 ISO 형식으로 반환"""
    return datetime.now(timezone.utc).isoformat()

def is_market_open(self) -> bool:
    """현재 시장이 열려있는지 확인 (KST 기준)"""
    from datetime import datetime, time
    import pytz
    
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst).time()
    
    market_open = time.fromisoformat(self.MARKET_OPEN_TIME)
    market_close = time.fromisoformat(self.MARKET_CLOSE_TIME)
    
    # 주말 제외 (월-금만)
    weekday = datetime.now(kst).weekday()
    if weekday >= 5:  # 토요일(5), 일요일(6)
        return False
        
    return market_open <= now <= market_close

def get_database_url_for_env(self) -> str:
    """환경별 데이터베이스 URL 반환"""
    if self.ENVIRONMENT == "test":
        return self.TEST_DATABASE_URL or self.DATABASE_URL.replace(self.DB_NAME, f"{self.DB_NAME}_test")
    return self.DATABASE_URL
```

# 전역 설정 인스턴스

settings = Settings()

# 환경별 로깅 레벨 설정

LOGGING_CONFIG = {
“version”: 1,
“disable_existing_loggers”: False,
“formatters”: {
“default”: {
“format”: “[{asctime}] {levelname} {name}: {message}”,
“style”: “{”,
},
“json”: {
“()”: “pythonjsonlogger.jsonlogger.JsonFormatter”,
“format”: “%(asctime)s %(name)s %(levelname)s %(message)s”
}
},
“handlers”: {
“default”: {
“formatter”: “default”,
“class”: “logging.StreamHandler”,
“stream”: “ext://sys.stdout”,
},
“file”: {
“formatter”: “json” if settings.LOG_FORMAT == “json” else “default”,
“class”: “logging.handlers.RotatingFileHandler”,
“filename”: settings.LOG_FILE_PATH,
“maxBytes”: 100 * 1024 * 1024,  # 100MB
“backupCount”: settings.LOG_BACKUP_COUNT,
}
},
“root”: {
“level”: settings.LOG_LEVEL,
“handlers”: [“default”, “file”],
}
}