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
```