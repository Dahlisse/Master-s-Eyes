_fundamental(data: Dict[str, Any]):
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
        financial_data, market_data, 
# 백테스팅 AP.run_backtest(equal_weight_strategy, market_data)
    
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