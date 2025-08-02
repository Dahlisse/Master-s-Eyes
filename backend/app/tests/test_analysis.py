# backend/app/tests/test_analysis.py

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 프로젝트 루트 디렉토리를 Python path에 추가

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(**file**))))

from app.analysis.technical_indicators import TechnicalIndicators
from app.analysis.fundamental_engine import FundamentalEngine, FinancialRatios
from app.analysis.backtest_framework import BacktestEngine, BacktestConfig
from app.analysis.performance_metrics import PerformanceAnalyzer

class TestTechnicalIndicators(unittest.TestCase):
“”“기술적 지표 테스트”””

```
def setUp(self):
    """테스트 데이터 설정"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    
    # 가상의 주가 데이터 생성
    close_prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 500)
    high_prices = close_prices + np.random.randint(100, 1000, len(dates))
    low_prices = close_prices - np.random.randint(100, 1000, len(dates))
    open_prices = close_prices + np.random.randint(-500, 500, len(dates))
    volumes = np.random.randint(100000, 1000000, len(dates))
    
    self.test_data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    self.ti = TechnicalIndicators()

def test_calculate_all_indicators(self):
    """모든 지표 계산 테스트"""
    result = self.ti.calculate_all_indicators(self.test_data)
    
    # 기본 검증
    self.assertGreater(len(result.columns), len(self.test_data.columns))
    
    # 주요 지표 존재 확인
    expected_indicators = ['SMA_20', 'RSI_14', 'MACD', 'bb_upper', 'ATR']
    for indicator in expected_indicators:
        self.assertIn(indicator, result.columns)
    
    # 값 검증
    self.assertFalse(result['SMA_20'].isna().all())
    self.assertTrue((result['RSI_14'] >= 0).all() and (result['RSI_14'] <= 100).all())

def test_signal_summary(self):
    """매매 신호 요약 테스트"""
    result_df = self.ti.calculate_all_indicators(self.test_data)
    signals = self.ti.get_signal_summary(result_df)
    
    # 신호 구조 검증
    self.assertIn('overall', signals)
    self.assertIn('signal', signals['overall'])
    self.assertIn('confidence', signals['overall'])
    
    # 신호 타입 검증
    valid_signals = ['strong_buy', 'buy', 'neutral', 'sell', 'strong_sell']
    self.assertIn(signals['overall']['signal'], valid_signals)

def test_empty_data_handling(self):
    """빈 데이터 처리 테스트"""
    empty_df = pd.DataFrame()
    result = self.ti.calculate_all_indicators(empty_df)
    self.assertTrue(result.empty)
```

class TestFundamentalEngine(unittest.TestCase):
“”“펀더멘털 분석 테스트”””

```
def setUp(self):
    """테스트 데이터 설정"""
    self.engine = FundamentalEngine()
    
    self.sample_financial_data = {
        'revenue': 10000,
        'gross_profit': 3000,
        'operating_income': 1500,
        'net_income': 1000,
        'total_assets': 15000,
        'total_equity': 8000,
        'total_debt': 4000,
        'current_assets': 6000,
        'current_liabilities': 3000,
        'cash': 2000,
        'operating_cash_flow': 1200,
        'free_cash_flow': 700,
        'prev_revenue': 9000,
        'prev_net_income': 800
    }
    
    self.sample_market_data = {
        'market_cap': 20000,
        'share_price': 50000,
        'shares_outstanding': 400000000,
        'enterprise_value': 22000
    }

def test_financial_ratios_calculation(self):
    """재무비율 계산 테스트"""
    ratios = self.engine.calculate_financial_ratios(self.sample_financial_data)
    
    # 기본 검증
    self.assertIsInstance(ratios, FinancialRatios)
    self.assertGreater(ratios.roe, 0)
    self.assertGreater(ratios.roa, 0)
    self.assertEqual(round(ratios.roe, 1), 12.5)  # 1000/8000 * 100

def test_intrinsic_value_calculation(self):
    """내재가치 계산 테스트"""
    result = self.engine.calculate_intrinsic_value(
        self.sample_financial_data, 
        self.sample_market_data
    )
    
    # 결과 구조 검증
    required_keys = ['intrinsic_value', 'margin_of_safety', 'recommendation']
    for key in required_keys:
        self.assertIn(key, result)
    
    self.assertIsInstance(result['intrinsic_value'], (int, float))
    self.assertIsInstance(result['margin_of_safety'], (int, float))

def test_comprehensive_analysis(self):
    """종합 분석 테스트"""
    analysis = self.engine.comprehensive_analysis(
        self.sample_financial_data,
        self.sample_market_data
    )
    
    # 분석 구성 요소 확인
    required_keys = ['ratios', 'valuation', 'intrinsic_value', 'quality_analysis', 'summary']
    for key in required_keys:
        self.assertIn(key, analysis)
    
    # 등급 검증
    valid_grades = ['A+', 'A', 'B+', 'B', 'C+', 'C']
    self.assertIn(analysis['summary']['final_grade'], valid_grades)
```

class TestBacktestFramework(unittest.TestCase):
“”“백테스팅 프레임워크 테스트”””

```
def setUp(self):
    """테스트 설정"""
    self.config = BacktestConfig(
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=10000000,
        rebalance_frequency='monthly'
    )
    
    # 가상 시장 데이터 생성
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    tickers = ['TEST001', 'TEST002']
    
    self.market_data = {}
    np.random.seed(42)
    
    for ticker in tickers:
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 50000 * np.cumprod(1 + returns)
        volumes = np.random.randint(100000, 1000000, len(dates))
        
        self.market_data[ticker] = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })

def simple_strategy(self, market_data, positions, date):
    """단순 테스트 전략"""
    if not positions:  # 첫 거래일
        return {'TEST001': 0.5, 'TEST002': 0.5}  # 50:50 분산
    return {}

def test_backtest_execution(self):
    """백테스트 실행 테스트"""
    engine = BacktestEngine(self.config)
    result = engine.run_backtest(self.simple_strategy, self.market_data)
    
    # 결과 구조 검증
    required_keys = ['performance', 'portfolio_history', 'trades']
    for key in required_keys:
        self.assertIn(key, result)
    
    # 성과 지표 확인
    performance = result['performance']
    self.assertIsNotNone(performance.total_return)
    self.assertIsNotNone(performance.sharpe_ratio)
    self.assertGreaterEqual(performance.trades_count, 0)

def test_config_validation(self):
    """설정 검증 테스트"""
    # 올바른 설정
    config = BacktestConfig(initial_capital=1000000)
    self.assertEqual(config.initial_capital, 1000000)
    
    # 기본값 확인
    self.assertEqual(config.commission_rate, 0.0015)
    self.assertEqual(config.rebalance_frequency, 'monthly')
```

class TestPerformanceMetrics(unittest.TestCase):
“”“성과 측정 테스트”””

```
def setUp(self):
    """테스트 데이터 설정"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # 가상 수익률 생성 (연 10% 수익률, 20% 변동성)
    self.returns = pd.Series(
        np.random.normal(0.0004, 0.013, len(dates)),
        index=dates
    )
    
    # 벤치마크 수익률 (연 8% 수익률, 15% 변동성)
    self.benchmark_returns = pd.Series(
        np.random.normal(0.0003, 0.01, len(dates)),
        index=dates
    )
    
    self.analyzer = PerformanceAnalyzer()

def test_comprehensive_analysis(self):
    """종합 분석 테스트"""
    analysis = self.analyzer.analyze_returns(self.returns, self.benchmark_returns)
    
    # 분석 구성 요소 확인
    required_keys = ['return_metrics', 'risk_metrics', 'risk_adjusted_metrics', 'summary']
    for key in required_keys:
        self.assertIn(key, analysis)
    
    # 기본 지표 확인
    return_metrics = analysis['return_metrics']
    self.assertIsNotNone(return_metrics.total_return)
    self.assertIsNotNone(return_metrics.annualized_return)
    
    risk_metrics = analysis['risk_metrics']
    self.assertIsNotNone(risk_metrics.volatility)
    self.assertIsNotNone(risk_metrics.max_drawdown)
    
    # 등급 검증
    summary = analysis['summary']
    valid_grades = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D']
    self.assertIn(summary.grade, valid_grades)

def test_risk_metrics_calculation(self):
    """리스크 지표 계산 테스트"""
    analysis = self.analyzer.analyze_returns(self.returns)
    risk_metrics = analysis['risk_metrics']
    
    # VaR 검증 (5% < 95%)
    self.assertLess(risk_metrics.var_95, risk_metrics.var_99)
    
    # 변동성 양수 검증
    self.assertGreater(risk_metrics.volatility, 0)
    
    # 최대 낙폭 양수 검증
    self.assertGreaterEqual(risk_metrics.max_drawdown, 0)

def test_empty_returns_handling(self):
    """빈 수익률 데이터 처리 테스트"""
    empty_returns = pd.Series(dtype=float)
    analysis = self.analyzer.analyze_returns(empty_returns)
    
    # 빈 결과 검증
    self.assertEqual(analysis['return_metrics'].total_return, 0.0)
    self.assertEqual(analysis['summary'].grade, 'C')
```

class TestIntegration(unittest.TestCase):
“”“통합 테스트”””

```
def test_full_pipeline(self):
    """전체 파이프라인 테스트"""
    # 1. 기술적 지표 계산
    ti = TechnicalIndicators()
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    np.random.seed(42)
    
    price_data = pd.DataFrame({
        'open': 50000 + np.cumsum(np.random.randn(len(dates)) * 300),
        'high': 51000 + np.cumsum(np.random.randn(len(dates)) * 300),
        'low': 49000 + np.cumsum(np.random.randn(len(dates)) * 300),
        'close': 50000 + np.cumsum(np.random.randn(len(dates)) * 300),
        'volume': np.random.randint(100000, 500000, len(dates))
    })
    
    technical_result = ti.calculate_all_indicators(price_data)
    self.assertGreater(len(technical_result.columns), 5)
    
    # 2. 펀더멘털 분석
    fe = FundamentalEngine()
    financial_data = {
        'revenue': 5000, 'net_income': 500, 'total_assets': 8000,
        'total_equity': 4000, 'free_cash_flow': 300
    }
    market_data = {
        'market_cap': 10000, 'share_price': 25000, 
        'shares_outstanding': 400000
    }
    
    fundamental_result = fe.comprehensive_analysis(financial_data, market_data)
    self.assertIn('summary', fundamental_result)
    
    # 3. 성과 분석
    pa = PerformanceAnalyzer()
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    performance_result = pa.analyze_returns(returns)
    self.assertIn('summary', performance_result)
    
    print("✅ 전체 파이프라인 테스트 통과")
```

if **name** == ‘**main**’:
# 테스트 스위트 실행
unittest.main(verbosity=2)