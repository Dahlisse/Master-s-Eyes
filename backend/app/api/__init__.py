# backend/app/api/**init**.py

“””
API 패키지 초기화

- 모든 API 라우터 통합
- 라우터 등록 및 관리
  “””

from .v1.router import api_router

**all** = [“api_router”]