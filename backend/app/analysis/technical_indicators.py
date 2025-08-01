# backend/app/analysis/technical_indicators.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
from scipy import stats
import warnings
warnings.filterwarnings(‘ignore’)

class TechnicalIndicators:
“”“기술적 지표 계산 라이브러리”””

```
def __init__(self):
    pass

def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """모든 기술적 지표를 한 번에 계산"""
    if df.empty or len(df) < 50:
        return df
        
    result_df = df.copy()
    
    # 트렌드 지표
    result_df = self._add_trend_indicators(result_df)
    
    # 모멘텀 지표
    result_df = self._add_momentum_indicators(result_df)
    
    # 변동성 지표
    result_df = self._add_volatility_indicators(result_df)
    
    # 거래량 지표
    result_df = self._add_volume_indicators(result_df)
    
    # 기타 지표
    result_df = self._add_others_indicators(result_df)
    
    return result_df

def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """트렌드 지표 추가"""
    # 이동평균선
    df['SMA_5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_60'] = ta.trend.sma_indicator(df['close'], window=60)
    df['SMA_120'] = ta.trend.sma_indicator(df['close'], window=120)
    
    # 지수이동평균선
    df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_histogram'] = macd.macd_diff()
    
    # ADX (Average Directional Index)
    df['ADX'] = ta.trend.adx_indicator(df['high'], df['low'], df['close'], window=14)
    df['ADX_pos'] = ta.trend.adx_pos_indicator(df['high'], df['low'], df['close'], window=14)
    df['ADX_neg'] = ta.trend.adx_neg_indicator(df['high'], df['low'], df['close'], window=14)
    
    # Parabolic SAR
    df['PSAR'] = ta.trend.psar_indicator(df['high'], df['low'], df['close'])
    
    # Ichimoku
    ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
    
    return df

def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """모멘텀 지표 추가"""
    # RSI
    df['RSI_14'] = ta.momentum.rsi_indicator(df['close'], window=14)
    df['RSI_9'] = ta.momentum.rsi_indicator(df['close'], window=9)
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Williams %R
    df['williams_r'] = ta.momentum.williams_r_indicator(df['high'], df['low'], df['close'])
    
    # CCI (Commodity Channel Index)
    df['CCI'] = ta.momentum.cci_indicator(df['high'], df['low'], df['close'])
    
    # ROC (Rate of Change)
    df['ROC'] = ta.momentum.roc_indicator(df['close'])
    
    # Ultimate Oscillator
    df['ultimate_osc'] = ta.momentum.ultimate_oscillator_indicator(
        df['high'], df['low'], df['close']
    )
    
    # Awesome Oscillator
    df['awesome_osc'] = ta.momentum.awesome_oscillator_indicator(df['high'], df['low'])
    
    return df

def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """변동성 지표 추가"""
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Average True Range
    df['ATR'] = ta.volatility.average_true_range_indicator(df['high'], df['low'], df['close'])
    
    # Donchian Channel
    donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
    df['donchian_upper'] = donchian.donchian_channel_hband()
    df['donchian_lower'] = donchian.donchian_channel_lband()
    df['donchian_middle'] = donchian.donchian_channel_mband()
    
    # Keltner Channel
    keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
    df['keltner_upper'] = keltner.keltner_channel_hband()
    df['keltner_lower'] = keltner.keltner_channel_lband()
    df['keltner_middle'] = keltner.keltner_channel_mband()
    
    return df

def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """거래량 지표 추가"""
    # Volume SMA
    df['volume_sma'] = ta.volume.volume_sma_indicator(df['close'], df['volume'])
    
    # Volume Weighted Average Price
    df['vwap'] = ta.volume.volume_weighted_average_price_indicator(
        df['high'], df['low'], df['close'], df['volume']
    )
    
    # On Balance Volume
    df['OBV'] = ta.volume.on_balance_volume_indicator(df['close'], df['volume'])
    
    # Chaikin Money Flow
    df['CMF'] = ta.volume.chaikin_money_flow_indicator(
        df['high'], df['low'], df['close'], df['volume']
    )
    
    # Accumulation/Distribution Line
    df['AD'] = ta.volume.acc_dist_index_indicator(
        df['high'], df['low'], df['close'], df['volume']
    )
    
    # Money Flow Index
    df['MFI'] = ta.volume.money_flow_index_indicator(
        df['high'], df['low'], df['close'], df['volume']
    )
    
    # Volume Rate of Change
    df['volume_roc'] = df['volume'].pct_change(periods=10)
    
    return df

def _add_others_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """기타 지표 추가"""
    # Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Volatility (20일 기준)
    df['volatility_20'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
    
    # Z-Score (가격의 표준화)
    df['price_zscore'] = stats.zscore(df['close'].fillna(method='ffill'))
    
    # Price Momentum (5일, 20일)
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # High-Low Spread
    df['hl_spread'] = (df['high'] - df['low']) / df['close']
    
    # Gap (전일 종가 대비 당일 시가)
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # 52주 신고가/신저가 대비
    df['high_52w'] = df['high'].rolling(window=252).max()
    df['low_52w'] = df['low'].rolling(window=252).min()
    df['close_to_52w_high'] = df['close'] / df['high_52w']
    df['close_to_52w_low'] = df['close'] / df['low_52w']
    
    return df

def get_signal_summary(self, df: pd.DataFrame) -> Dict:
    """기술적 지표 기반 매매 신호 요약"""
    if df.empty:
        return {}
        
    latest = df.iloc[-1]
    signals = {}
    
    # 트렌드 신호
    signals['trend'] = self._analyze_trend_signals(latest)
    
    # 모멘텀 신호
    signals['momentum'] = self._analyze_momentum_signals(latest)
    
    # 변동성 신호
    signals['volatility'] = self._analyze_volatility_signals(latest)
    
    # 거래량 신호
    signals['volume'] = self._analyze_volume_signals(latest)
    
    # 종합 신호
    signals['overall'] = self._calculate_overall_signal(signals)
    
    return signals

def _analyze_trend_signals(self, latest: pd.Series) -> Dict:
    """트렌드 신호 분석"""
    signals = {}
    score = 0
    
    # MACD 신호
    if pd.notna(latest['MACD']) and pd.notna(latest['MACD_signal']):
        if latest['MACD'] > latest['MACD_signal']:
            score += 1
            signals['macd'] = 'bullish'
        else:
            score -= 1
            signals['macd'] = 'bearish'
    
    # 이동평균 배열
    if all(pd.notna(latest[col]) for col in ['SMA_5', 'SMA_20', 'SMA_60']):
        if latest['SMA_5'] > latest['SMA_20'] > latest['SMA_60']:
            score += 2
            signals['ma_alignment'] = 'bullish'
        elif latest['SMA_5'] < latest['SMA_20'] < latest['SMA_60']:
            score -= 2
            signals['ma_alignment'] = 'bearish'
        else:
            signals['ma_alignment'] = 'neutral'
    
    # ADX 신호
    if pd.notna(latest['ADX']):
        if latest['ADX'] > 25:
            signals['trend_strength'] = 'strong'
            if pd.notna(latest['ADX_pos']) and pd.notna(latest['ADX_neg']):
                if latest['ADX_pos'] > latest['ADX_neg']:
                    score += 1
                else:
                    score -= 1
        else:
            signals['trend_strength'] = 'weak'
    
    signals['score'] = score
    return signals

def _analyze_momentum_signals(self, latest: pd.Series) -> Dict:
    """모멘텀 신호 분석"""
    signals = {}
    score = 0
    
    # RSI 신호
    if pd.notna(latest['RSI_14']):
        rsi = latest['RSI_14']
        if rsi > 70:
            signals['rsi'] = 'overbought'
            score -= 1
        elif rsi < 30:
            signals['rsi'] = 'oversold'
            score += 1
        else:
            signals['rsi'] = 'neutral'
    
    # Stochastic 신호
    if pd.notna(latest['stoch_k']) and pd.notna(latest['stoch_d']):
        if latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] < 80:
            score += 1
            signals['stochastic'] = 'bullish'
        elif latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'] > 20:
            score -= 1
            signals['stochastic'] = 'bearish'
        else:
            signals['stochastic'] = 'neutral'
    
    # Williams %R 신호
    if pd.notna(latest['williams_r']):
        wr = latest['williams_r']
        if wr > -20:
            signals['williams'] = 'overbought'
        elif wr < -80:
            signals['williams'] = 'oversold'
            score += 1
        else:
            signals['williams'] = 'neutral'
    
    signals['score'] = score
    return signals

def _analyze_volatility_signals(self, latest: pd.Series) -> Dict:
    """변동성 신호 분석"""
    signals = {}
    score = 0
    
    # Bollinger Bands 신호
    if pd.notna(latest['bb_percent']):
        bb_pos = latest['bb_percent']
        if bb_pos > 0.8:
            signals['bollinger'] = 'overbought'
            score -= 1
        elif bb_pos < 0.2:
            signals['bollinger'] = 'oversold'
            score += 1
        else:
            signals['bollinger'] = 'neutral'
    
    # BB Width (변동성 측정)
    if pd.notna(latest['bb_width']):
        signals['volatility_level'] = 'high' if latest['bb_width'] > 0.1 else 'low'
    
    signals['score'] = score
    return signals

def _analyze_volume_signals(self, latest: pd.Series) -> Dict:
    """거래량 신호 분석"""
    signals = {}
    score = 0
    
    # OBV 추세 (간단 버전)
    if pd.notna(latest['OBV']):
        signals['obv_available'] = True
    
    # MFI 신호
    if pd.notna(latest['MFI']):
        mfi = latest['MFI']
        if mfi > 80:
            signals['mfi'] = 'overbought'
            score -= 1
        elif mfi < 20:
            signals['mfi'] = 'oversold'
            score += 1
        else:
            signals['mfi'] = 'neutral'
    
    signals['score'] = score
    return signals

def _calculate_overall_signal(self, signals: Dict) -> Dict:
    """종합 신호 계산"""
    total_score = 0
    signal_count = 0
    
    for category in ['trend', 'momentum', 'volatility', 'volume']:
        if category in signals and 'score' in signals[category]:
            total_score += signals[category]['score']
            signal_count += 1
    
    if signal_count == 0:
        return {'signal': 'neutral', 'score': 0, 'confidence': 0}
    
    avg_score = total_score / signal_count
    
    if avg_score > 1:
        signal = 'strong_buy'
    elif avg_score > 0.5:
        signal = 'buy'
    elif avg_score > -0.5:
        signal = 'neutral'
    elif avg_score > -1:
        signal = 'sell'
    else:
        signal = 'strong_sell'
    
    confidence = min(abs(avg_score) * 50, 100)  # 0-100%
    
    return {
        'signal': signal,
        'score': round(avg_score, 2),
        'confidence': round(confidence, 1)
    }
```

# 사용 예시

if **name** == “**main**”:
# 테스트용 데이터 생성
dates = pd.date_range(‘2023-01-01’, ‘2024-12-31’, freq=‘D’)
np.random.seed(42)

```
# 가상의 주가 데이터
close_prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 500)
high_prices = close_prices + np.random.randint(100, 1000, len(dates))
low_prices = close_prices - np.random.randint(100, 1000, len(dates))
open_prices = close_prices + np.random.randint(-500, 500, len(dates))
volumes = np.random.randint(100000, 1000000, len(dates))

test_df = pd.DataFrame({
    'date': dates,
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volumes
})

# 지표 계산 테스트
ti = TechnicalIndicators()
result_df = ti.calculate_all_indicators(test_df)
signals = ti.get_signal_summary(result_df)

print("📊 계산된 지표 수:", len(result_df.columns) - len(test_df.columns))
print("🎯 종합 신호:", signals['overall'])
print("📈 트렌드 신호:", signals.get('trend', {}).get('score', 0))
print("⚡ 모멘텀 신호:", signals.get('momentum', {}).get('score', 0))
```