# ============================================================================

# backend/app/tests/test_analysis.py - 분석 모듈 테스트 (완전 분리 버전)

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

def test
```