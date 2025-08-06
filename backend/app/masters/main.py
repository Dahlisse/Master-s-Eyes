# backend/app/main.py

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# 분석 엔진 임포트

from app.analysis.technical_indicators import TechnicalIndicators
from app.analysis.fundamental_engine import FundamentalEngine
from app.analysis.backtest_framework import BacktestEngine, BacktestConfig, buy_and_hold_strategy
from app.analysis.performance_metrics import PerformanceAnalyzer
from app.config.analysis_settings import DEFAULT_ANALYSIS_CONFIG, get_config_for_user

# 로깅 설정

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

# FastAPI 앱 생성

app = FastAPI(
title=“Masters Eye Analysis Engine API”,
description=“4대 거장 융합 주식 분석 시스템 - Week 4 기본 분석 엔진”,
version=“0.4.0”,
docs_url=”/docs”,
redoc_url=”/redoc”
)

# CORS 설정

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],  # 개발환경용, 프로덕션에서는 특정 도메인만 허용
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

# 분석 엔진 인스턴스 (싱글톤)

technical_analyzer = TechnicalIndicators()
fundamental_analyzer = FundamentalEngine()
performance_analyzer = PerformanceAnalyzer()

# 전역 예외 처리

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
logger.error(f”Unexpected error: {str(exc)}”)
return JSONResponse(
status_code=500,
content={“error”: “Internal server error”, “detail”: “시스템 오류가 발생했습니다.”}
)

# 헬스 체크

@app.get(”/”)
async def root():
“”“루트 엔드포인트 - 시스템 상태 확인”””
return {
“service”: “Masters Eye Analysis Engine”,
“version”: “0.4.0”,
“status”: “running”,
“timestamp”: datetime.now().isoformat(),
“features”: [
“기술적 지표 분석 (50+ 지표)”,
“펀더멘털 분석 (DCF, 재무비율)”,
“백테스팅 프레임워크”,
“성과 측정 시스템”,
“몬테카를로 시뮬레이션”
]
}

@app.get(”/api/v1/health”)
async def health_check():
“”“상세 헬스 체크”””
try:
# 각 분석 엔진 상태 확인
config = DEFAULT_ANALYSIS_CONFIG

```
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "engines": {
            "technical_analyzer": "ready",
            "fundamental_analyzer": "ready", 
            "performance_analyzer": "ready",
            "backtest_engine": "ready"
        },
        "configuration": {
            "technical_indicators": len(config.technical_indicators),
            "industry_benchmarks": len(config.fundamental_analysis['industry_benchmarks']),
            "default_commission_rate": config.backtest_config['commission_rate'],
            "risk_free_rate": config.performance_config['risk_free_rate']
        },
        "system_info": {
            "python_version": "3.8+",
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__
        }
    }
except Exception as e:
    logger.error(f"Health check failed: {str(e)}")
    return JSONResponse(
        status_code=503,
        content={"status": "unhealthy", "error": str(e)}
    )
```

# 기술적 분석 API

@app.post(”/api/v1/analysis/technical”)
async def analyze_technical(data: Dict[str, Any]):
“””
기술적 분석 실행

```
Body:
- price_data: 주가 데이터 (OHLCV)
- indicators: 선택적 지표 리스트
"""
try:
    # 입력 데이터 검증
    if 'price_data' not in data:
        raise HTTPException(
            status_code=400, 
            detail="price_data가 필요합니다. OHLCV 형식의 주가 데이터를 제공해주세요."
        )
    
    price_data = data['price_data']
    if not price_data:
        raise HTTPException(status_code=400, detail="price_data가 비어있습니다.")
    
    # DataFrame 변환
    df = pd.DataFrame(price_data)
    
    # 필수 컬럼 확인
    required_cols = ['close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400, 
            detail=f"필수 컬럼이 없습니다: {missing_cols}"
        )
    
    # 최소 데이터 포인트 확인
    if len(df) < 20:
        raise HTTPException(
            status_code=400,
            detail="최소 20일 이상의 데이터가 필요합니다."
        )
    
    # 기술적 지표 계산
    logger.info(f"기술적 분석 시작: {len(df)}일 데이터")
    result_df = technical_analyzer.calculate_all_indicators(df)
    signals = technical_analyzer.get_signal_summary(result_df)
    
    # 최신 값들 추출
    latest_values = {}
    if not result_df.empty:
        latest_row = result_df.iloc[-1]
        # NaN 값 제거하고 주요 지표만 반환
        key_indicators = ['SMA_20', 'RSI_14', 'MACD', 'bb_upper', 'bb_lower', 'close']
        for indicator in key_indicators:
            if indicator in latest_row and pd.notna(latest_row[indicator]):
                latest_values[indicator] = float(latest_row[indicator])
    
    logger.info("기술적 분석 완료")
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "data_points": len(df),
        "indicators_calculated": len(result_df.columns) - len(df.columns),
        "signals": signals,
        "latest_indicators": latest_values,
        "recommendations": {
            "overall_signal": signals.get('overall', {}).get('signal', 'neutral'),
            "confidence": signals.get('overall', {}).get('confidence', 0),
            "key_insights": _generate_technical_insights(signals)
        }
    }
    
except HTTPException:
    raise
except Exception as e:
    logger.error(f"기술적 분석 오류: {str(e)}")
    raise HTTPException(status_code=500, detail=f"기술적 분석 중 오류가 발생했습니다: {str(e)}")
```

def _generate_technical_insights(signals: Dict) -> List[str]:
“”“기술적 분석 인사이트 생성”””
insights = []

```
overall = signals.get('overall', {})
signal = overall.get('signal', 'neutral')
confidence = overall.get('confidence', 0)

if signal in ['strong_buy', 'buy']:
    insights.append(f"상승 신호 감지 (신뢰도: {confidence}%)")
elif signal in ['strong_sell', 'sell']:
    insights.append(f"하락 신호 감지 (신뢰도: {confidence}%)")
else:
    insights.append("중립적 신호")

# 트렌드 분석
trend = signals.get('trend', {})
if trend.get('score', 0) > 1:
    insights.append("강한 상승 트렌드")
elif trend.get('score', 0) < -1:
    insights.append("강한 하락 트렌드")

# 모멘텀 분석
momentum = signals.get('momentum', {})
if momentum.get('rsi') == 'overbought':
    insights.append("과매수 구간 - 조정 가능성")
elif momentum.get('rsi') == 'oversold':
    insights.append("과매도 구간 - 반등 가능성")

return insights[:3]  # 최대 3개
```

# 펀더멘털 분석 API

@app.post(”/api/v1/analysis/fundamental”)
async def analyze_fundamental(data: Dict[str, Any]):
“””
펀더멘털 분석 실행

```
Body:
- financial_data: 재무 데이터
- market_data: 시장 데이터
- industry: 업종 (선택)
"""
try:
    financial_data = data.get('financial_data', {})
    market_data = data.get('market_data', {})
    industry = data.get('industry', 'default')
    
    if not financial_data:
        raise HTTPException(status_code=400, detail="financial_data가 필요합니다.")
    
    # 필수 재무 데이터 확인
    required_financial = ['revenue', 'net_income', 'total_assets', 'total_equity']
    missing_financial = [field for field in required_financial if field not in financial_data]
    if missing_financial:
        raise HTTPException(
            status_code=400,
            detail=f"필수 재무 데이터가 없습니다: {missing_financial}"
        )
    
    logger.info(f"펀더멘털 분석 시작: {industry} 업종")
    
    # 종합 분석 실행
    analysis = fundamental_analyzer.comprehensive_analysis(
        financial_data, market_data, industry
    )
    
    logger.info("펀더멘털 분석 완료")
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "industry": industry,
        "analysis": {
            "financial_ratios": analysis['ratios'].__dict__,
            "valuation_metrics": analysis['valuation'].__dict__,
            "intrinsic_value": analysis['intrinsic_value'],
            "quality_analysis": analysis['quality_analysis'],
            "peer_comparison": analysis['peer_comparison']
        },
        "summary": {
            "final_grade": analysis['summary']['final_grade'],
            "recommendation": analysis['summary']['recommendation'],
            "total_score": analysis['summary']['total_score'],
            "key_strengths": analysis['summary']['key_strengths'],
            "key_concerns": analysis['summary']['key_concerns']
        }
    }
    
except HTTPException:
    raise
except Exception as e:
    logger.error(f"펀더멘털 분석 오류: {str(e)}")
    raise HTTPException(status_code=500, detail=f"펀더멘털 분석 중 오류가 발생했습니다: {str(e)}")
```

# 백테스팅 API

@app.post(”/api/v1/backtest/run”)
async def run_backtest(data: Dict[str, Any]):
“””
백테스팅 실행

```
Body:
- config: 백테스팅 설정
- market_data: 시장 데이터 (종목별)
- strategy: 전략 타입 (선택)
"""
try:
    # 설정 파싱
    config_data = data.get('config', {})
    
    # 기본값으로 BacktestConfig 생성
    config = BacktestConfig(
        start_date=config_data.get('start_date', '2023-01-01'),
        end_date=config_data.get('end_date', '2024-12-31'),
        initial_capital=config_data.get('initial_capital', 100_000_000),
        commission_rate=config_data.get('commission_rate', 0.0015),
        rebalance_frequency=config_data.get('rebalance_frequency', 'monthly')
    )
    
    # 시장 데이터 검증
    market_data_raw = data.get('market_data', {})
    if not market_data_raw:
        raise HTTPException(status_code=400, detail="market_data가 필요합니다.")
    
    # 시장 데이터 변환
    market_data = {}
    for ticker, price_data in market_data_raw.items():
        if not price_data:
            continue
        df = pd.DataFrame(price_data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        market_data[ticker] = df
    
    if not market_data:
        raise HTTPException(status_code=400, detail="유효한 시장 데이터가 없습니다.")
    
    logger.info(f"백테스팅 시작: {len(market_data)}개 종목, {config.start_date}~{config.end_date}")
    
    # 전략 선택 (현재는 균등분산 전략만 지원)
    strategy_type = data.get('strategy', 'equal_weight')
    
    def equal_weight_strategy(market_data_point, positions, date):
        if not positions:  # 첫 거래일
            tickers = [col.replace('_close', '') for col in market_data_point.index if col.endswith('_close')]
            if tickers:
                weight = 1.0 / len(tickers)
                return {ticker: weight for ticker in tickers}
        return {}
    
    # 백테스트 실행
    engine = BacktestEngine(config)
    result = engine.run_backtest(equal_weight_strategy, market_data)
    
    # 포트폴리오 히스토리가 있는지 확인
    final_value = 0
    if not result['portfolio_history'].empty:
        final_value = result['portfolio_history']['portfolio_value'].iloc[-1]
    
    logger.info("백테스팅 완료")
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "period": f"{config.start_date} ~ {config.end_date}",
            "initial_capital": config.initial_capital,
            "commission_rate": config.commission_rate,
            "strategy": strategy_type
        },
        "performance": {
            "total_return": result['performance'].total_return,
            "annualized_return": result['performance'].annualized_return,
            "volatility": result['performance'].volatility,
            "sharpe_ratio": result['performance'].sharpe_ratio,
            "max_drawdown": result['performance'].max_drawdown,
            "win_rate": result['performance'].win_rate,
            "trades_count": result['performance'].trades_count
        },
        "summary": {
            "final_portfolio_value": final_value,
            "total_profit_loss": final_value - config.initial_capital,
            "number_of_stocks": len(market_data),
            "trading_days": len(result['portfolio_history']) if not result['portfolio_history'].empty else 0
        }
    }
    
except HTTPException:
    raise
except Exception as e:
    logger.error(f"백테스팅 오류: {str(e)}")
    raise HTTPException(status_code=500, detail=f"백테스팅 중 오류가 발생했습니다: {str(e)}")
```

# 성과 분석 API

@app.post(”/api/v1/analysis/performance”)
async def analyze_performance(data: Dict[str, Any]):
“””
성과 분석 실행

```
Body:
- returns: 일별 수익률 리스트
- benchmark_returns: 벤치마크 수익률 (선택)
- portfolio_values: 포트폴리오 가치 (선택)
"""
try:
    # 수익률 데이터 검증
    returns_data = data.get('returns', [])
    if not returns_data:
        raise HTTPException(status_code=400, detail="returns 데이터가 필요합니다.")
    
    if len(returns_data) < 30:
        raise HTTPException(status_code=400, detail="최소 30일 이상의 수익률 데이터가 필요합니다.")
    
    # 벤치마크 데이터
    benchmark_data = data.get('benchmark_returns', [])
    
    # Series 변환
    returns = pd.Series(returns_data, dtype=float)
    benchmark_returns = pd.Series(benchmark_data, dtype=float) if benchmark_data else None
    portfolio_values = None
    
    if 'portfolio_values' in data:
        portfolio_values = pd.Series(data['portfolio_values'], dtype=float)
    
    logger.info(f"성과 분석 시작: {len(returns)}일 데이터")
    
    # 성과 분석 실행
    if portfolio_values is not None:
        analysis = performance_analyzer.comprehensive_analysis(returns, benchmark_returns, portfolio_values)
    else:
        analysis = performance_analyzer.analyze_returns(returns, benchmark_returns)
    
    logger.info("성과 분석 완료")
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "data_points": len(returns),
        "has_benchmark": benchmark_returns is not None,
        "analysis": {
            "return_metrics": analysis['return_metrics'].__dict__ if hasattr(analysis['return_metrics'], '__dict__') else analysis['return_metrics'],
            "risk_metrics": analysis['risk_metrics'].__dict__ if hasattr(analysis['risk_metrics'], '__dict__') else analysis['risk_metrics'],
            "risk_adjusted_metrics": analysis['risk_adjusted_metrics'].__dict__ if hasattr(analysis['risk_adjusted_metrics'], '__dict__') else analysis['risk_adjusted_metrics']
        },
        "summary": {
            "grade": analysis.get('overall_grade', analysis['summary'].grade if hasattr(analysis['summary'], 'grade') else 'C'),
            "risk_level": analysis['summary'].risk_level if hasattr(analysis['summary'], 'risk_level') else 'Medium',
            "key_metrics": {
                "total_return": analysis['summary'].total_return if hasattr(analysis['summary'], 'total_return') else 0,
                "sharpe_ratio": analysis['summary'].sharpe_ratio if hasattr(analysis['summary'], 'sharpe_ratio') else 0,
                "max_drawdown": analysis['summary'].max_drawdown if hasattr(analysis['summary'], 'max_drawdown') else 0
            }
        }
    }
    
except HTTPException:
    raise
except Exception as e:
    logger.error(f"성과 분석 오류: {str(e)}")
    raise HTTPException(status_code=500, detail=f"성과 분석 중 오류가 발생했습니다: {str(e)}")
```

# 통합 분석 API (미래 Week 5+ 준비)

@app.post(”/api/v1/analysis/comprehensive”)
async def comprehensive_analysis(data: Dict[str, Any]):
“””
통합 분석 (기술적 + 펀더멘털 + 성과)
“””
try:
results = {}

```
    # 기술적 분석
    if 'price_data' in data:
        tech_result = await analyze_technical({'price_data': data['price_data']})
        results['technical'] = tech_result
    
    # 펀더멘털 분석  
    if 'financial_data' in data:
        fund_data = {
            'financial_data': data['financial_data'],
            'market_data': data.get('market_data', {}),
            'industry': data.get('industry', 'default')
        }
        fund_result = await analyze_fundamental(fund_data)
        results['fundamental'] = fund_result
    
    # 성과 분석
    if 'returns' in data:
        perf_data = {
            'returns': data['returns'],
            'benchmark_returns': data.get('benchmark_returns', [])
        }
        perf_result = await analyze_performance(perf_data)
        results['performance'] = perf_result
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "analyses_completed": list(results.keys()),
        "results": results,
        "integrated_summary": _generate_integrated_summary(results)
    }
    
except Exception as e:
    logger.error(f"통합 분석 오류: {str(e)}")
    raise HTTPException(status_code=500, detail=f"통합 분석 중 오류가 발생했습니다: {str(e)}")
```

def _generate_integrated_summary(results: Dict) -> Dict:
“”“통합 분석 요약 생성”””
summary = {
“overall_score”: 0,
“recommendation”: “HOLD”,
“key_insights”: []
}

```
scores = []

# 기술적 분석 점수
if 'technical' in results:
    tech_confidence = results['technical'].get('recommendations', {}).get('confidence', 0)
    scores.append(tech_confidence)
    summary['key_insights'].append(f"기술적 신호 신뢰도: {tech_confidence}%")

# 펀더멘털 분석 점수
if 'fundamental' in results:
    fund_grade = results['fundamental']['summary']['final_grade']
    grade_scores = {'A+': 95, 'A': 85, 'B+': 75, 'B': 65, 'C+': 55, 'C': 45}
    fund_score = grade_scores.get(fund_grade, 50)
    scores.append(fund_score)
    summary['key_insights'].append(f"펀더멘털 등급: {fund_grade}")

# 성과 분석 점수
if 'performance' in results:
    perf_grade = results['performance']['summary']['grade']
    grade_scores = {'A+': 95, 'A': 85, 'B+': 75, 'B': 65, 'C+': 55, 'C': 45}
    perf_score = grade_scores.get(perf_grade, 50)
    scores.append(perf_score)
    summary['key_insights'].append(f"성과 등급: {perf_grade}")

# 종합 점수 계산
if scores:
    summary['overall_score'] = round(sum(scores) / len(scores), 1)
    
    if summary['overall_score'] >= 80:
        summary['recommendation'] = "STRONG_BUY"
    elif summary['overall_score'] >= 65:
        summary['recommendation'] = "BUY"
    elif summary['overall_score'] >= 35:
        summary['recommendation'] = "HOLD"
    else:
        summary['recommendation'] = "SELL"

return summary
```

if **name** == “**main**”:
import uvicorn
uvicorn.run(
“main:app”,
host=“0.0.0.0”,
port=8000,
reload=True,
log_level=“info”
)