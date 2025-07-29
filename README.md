# 📊 Master’s Eye

**4대 거장 융합 주식 AI 포트폴리오 시스템**

워렌 버핏, 레이 달리오, 리처드 파인만, 짐 사이먼스의 투자 철학을 융합한 AI 기반 자동화 투자 플랫폼

## 🎯 프로젝트 개요

Master’s Eye는 투자계의 4대 거장의 철학을 AI로 융합하여 최적의 포트폴리오를 생성하고 관리하는 혁신적인 시스템입니다.

### 🏛️ 4대 거장의 철학

- **💡 워렌 버핏**: 가치 투자 철학 (내재가치, 경제적 해자, 안전마진)
- **🌊 레이 달리오**: 거시경제 & All Weather 전략 (경제 사이클, 리스크 패리티)
- **🔬 리처드 파인만**: 과학적 사고 & 불확실성 정량화 (몬테카를로, 베이지안 추론)
- **📐 짐 사이먼스**: 퀀트 & 패턴 인식 (멀티팩터, 머신러닝, 시장 이상현상)

### ✨ 주요 기능

- 🤖 **AI 대화형 포트폴리오 조정**: “삼성전자 대신 다른 종목으로 바꿔줘”
- 📊 **실시간 시장 데이터**: 한국투자증권 API 연동
- 🔄 **자동매매 시스템**: 키움증권 API 연동
- 📈 **백테스팅 & 몬테카를로 시뮬레이션**: 과거 성과 검증 및 미래 예측
- 👥 **개인화된 UI**: 초보자(엄마)와 중급자(나) 맞춤 인터페이스

## 🚀 빠른 시작

### 📋 사전 요구사항

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- 한국투자증권 계좌 (API 키 발급용)
- 키움증권 계좌 (자동매매용, 선택사항)

### ⚡ 설치 가이드

#### 1. 저장소 클론

```bash
git clone https://github.com/your-username/masters-eye.git
cd masters-eye
```

#### 2. 환경 설정

```bash
cp .env.template .env
# .env 파일에서 API 키 설정
```

#### 3. 백엔드 실행

```bash
docker-compose up -d
curl http://localhost:8000/health
```

#### 4. 프론트엔드 실행

```bash
cd frontend
npm install
npm start
npm run electron-dev
```

## 📁 프로젝트 구조

```
masters-eye/
├── backend/           # Python FastAPI 백엔드
├── frontend/          # React + Electron 프론트엔드
├── database/          # 데이터베이스 스키마
├── docker-compose.yml # Docker 설정
└── README.md
```

## 🛠️ 기술 스택

### Backend

- **FastAPI**: 고성능 Python 웹 프레임워크
- **PostgreSQL + TimescaleDB**: 시계열 데이터베이스
- **Redis**: 캐싱 및 실시간 데이터
- **Celery**: 백그라운드 작업
- **Ollama + Llama 3.1**: 로컬 LLM

### Frontend

- **React 18 + TypeScript**: 모던 웹 프레임워크
- **Electron**: 데스크톱 앱
- **Tailwind CSS**: 유틸리티 CSS
- **TradingView Charts**: 고급 차트
- **Zustand**: 상태 관리

### APIs & Data

- **한국투자증권 KIS API**: 실시간 시장 데이터
- **키움증권 OpenAPI**: 자동매매
- **Yahoo Finance**: 글로벌 데이터
- **FRED API**: 경제 지표

## 📈 사용법

### 1. 사용자 선택

앱 실행 후 사용자 선택 (나 또는 엄마)

### 2. 포트폴리오 생성

- 투자 목표 및 위험 선호도 설정
- 4대 거장 가중치 조정
- AI가 최적 포트폴리오 생성

### 3. AI와 대화

```
"삼성전자 비중을 줄이고 SK하이닉스를 늘려줘"
"더 안전한 포트폴리오로 바꿔줘"
"이 종목을 왜 추천했나요?"
```

### 4. 백테스팅 확인

과거 10년간 성과 시뮬레이션 결과 확인

### 5. 자동매매 실행

키움증권 연동하여 실제 매매 자동화

## 🔧 개발 환경 설정

### 백엔드 개발

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 프론트엔드 개발

```bash
cd frontend
npm install
npm start
```

### 데이터베이스 마이그레이션

```bash
cd backend
alembic upgrade head
```

## 🧪 테스트

### 백엔드 테스트

```bash
cd backend
pytest
pytest --cov=app
```

### 프론트엔드 테스트

```bash
cd frontend
npm test
npm run test:coverage
```

## 📦 배포

### Electron 앱 빌드

```bash
cd frontend
npm run build-electron
```

### Docker 프로덕션 배포

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 🤝 기여하기

1. Fork the Project
1. Create Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Commit Changes (`git commit -m 'Add AmazingFeature'`)
1. Push to Branch (`git push origin feature/AmazingFeature`)
1. Open Pull Request

## 📄 라이선스

MIT License - 자세한 내용은 <LICENSE> 파일 참조

## 💬 지원 및 문의

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **이메일**: contact@masters-eye.com
- **문서**: <docs/> 폴더 참조

## 🙏 감사

- 워렌 버핏, 레이 달리오, 리처드 파인만, 짐 사이먼스의 투자 철학
- 한국투자증권 및 키움증권의 오픈 API
- 오픈소스 커뮤니티

-----

**⚠️ 면책 조항**: 이 소프트웨어는 교육 및 연구 목적으로 제공됩니다. 실제 투자 결정은 본인의 책임하에 이루어져야 하며, 투자 손실에 대한 책임을 지지 않습니다.