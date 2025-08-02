# backend/app/config/analysis_settings.py

from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class AnalysisConfig:
“”“분석 설정”””

```
# 기술적 지표 설정
technical_indicators: Dict[str, Any] = field(default_factory=dict)

# 펀더멘털 분석 설정  
fundamental_analysis: Dict[str, Any] = field(default_factory=dict)

# 백테스팅 설정
backtest_config: Dict[str, Any] = field(default_factory=dict)

# 성과 측정 설정
performance_config: Dict[str, Any] = field(default_factory=dict)

def __post_init__(self):
    """기본 설정값 초기화"""
    
    if not self.technical_indicators:
        self.technical_indicators = {
            # 이동평균선 설정
            'sma_windows': [5, 20, 60, 120],
            'ema_windows': [12, 26],
            
            # 모멘텀 지표 설정
            'rsi_window': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            
            # MACD 설정
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # 볼린저 밴드 설정
            'bollinger_window': 20,
            'bollinger_std': 2,
            
            # ATR 설정
            'atr_window': 14,
            
            # 스토캐스틱 설정
            'stochastic_k_window': 14,
            'stochastic_d_window': 3,
            'stochastic_overbought': 80,
            'stochastic_oversold': 20,
            
            # 신호 생성 설정
            'signal_threshold': 0.6,  # 신호 신뢰도 임계값
            'min_data_points': 50,    # 최소 데이터 포인트
        }
    
    if not self.fundamental_analysis:
        self.fundamental_analysis = {
            # DCF 모델 설정
            'dcf_growth_rate': 0.05,        # 초기 성장률 5%
            'dcf_terminal_growth': 0.025,   # 터미널 성장률 2.5%
            'dcf_discount_rate': 0.08,      # 할인율 8%
            'dcf_projection_years': 10,     # 예측 기간 10년
            
            # 안전마진 설정
            'safety_margin_threshold': 20,  # 20% 이상 안전마진
            
            # 업종별 벤치마크
            'industry_benchmarks': {
                'tech': {
                    'roe': 12.0, 'roa': 6.0, 'debt_ratio': 30.0,
                    'current_ratio': 200.0, 'per': 25.0, 'pbr': 2.5
                },
                'finance': {
                    'roe': 10.0, 'roa': 1.2, 'debt_ratio': 800.0,
                    'current_ratio': 120.0, 'per': 8.0, 'pbr': 0.8
                },
                'manufacturing': {
                    'roe': 6.0, 'roa': 3.0, 'debt_ratio': 60.0,
                    'current_ratio': 130.0, 'per': 10.0, 'pbr': 1.0
                },
                'retail': {
                    'roe': 8.0, 'roa': 4.0, 'debt_ratio': 50.0,
                    'current_ratio': 140.0, 'per': 15.0, 'pbr': 1.5
                },
                'healthcare': {
                    'roe': 14.0, 'roa': 7.0, 'debt_ratio': 25.0,
                    'current_ratio': 180.0, 'per': 20.0, 'pbr': 3.0
                }
            },
            
            # 품질 평가 가중치
            'quality_weights': {
                'profitability': 0.3,
                'growth': 0.25,
                'stability': 0.25,
                'efficiency': 0.2
            }
        }
    
    if not self.backtest_config:
        self.backtest_config = {
            # 거래 비용 설정
            'commission_rate': 0.0015,      # 수수료 0.15%
            'tax_rate': 0.0025,             # 거래세 + 농특세 0.25%
            'slippage': 0.001,              # 슬리피지 0.1%
            
            # 초기 설정
            'initial_capital': 100_000_000,  # 1억원
            'min_position_size': 1_000_000,  # 최소 포지션 100만원
            'max_position_weight': 0.2,      # 최대 종목 비중 20%
            
            # 리밸런싱 설정
            'rebalance_frequency': 'monthly',  # 월간 리밸런싱
            'rebalance_threshold': 0.05,       # 5% 이상 차이날 때 리밸런싱
            
            # 벤치마크 설정
            'benchmark': 'KOSPI',
            'risk_free_rate': 0.025,          # 무위험수익률 2.5%
            
            # 몬테카를로 설정
            'monte_carlo_simulations': 1000,   # 시뮬레이션 횟수
            'confidence_level': 0.95,          # 신뢰구간 95%
            'noise_level': 0.02                # 노이즈 수준 2%
        }
    
    if not self.performance_config:
        self.performance_config = {
            # 기본 설정
            'risk_free_rate': 0.025,        # 무위험수익률 2.5%
            'trading_days_per_year': 252,   # 연간 거래일
            
            # VaR 설정
            'confidence_levels': [0.95, 0.99],  # 95%, 99% VaR
            
            # 롤링 윈도우 설정
            'rolling_windows': {
                'short': 30,      # 단기 30일
                'medium': 60,     # 중기 60일
                'long': 252       # 장기 1년
            },
            
            # 등급 기준 (100점 만점)
            'grade_thresholds': {
                'A+': 90, 'A': 80, 'B+': 70, 
                'B': 60, 'C+': 50, 'C': 40
            },
            
            # 등급 가중치
            'grade_weights': {
                'returns': 0.3,        # 수익률 30%
                'risk': 0.25,          # 리스크 25%
                'risk_adjusted': 0.25,  # 위험조정수익률 25%
                'consistency': 0.2      # 일관성 20%
            },
            
            # 리스크 수준 기준
            'risk_level_thresholds': {
                'low': {'volatility': 15, 'max_drawdown': 10},
                'medium': {'volatility': 25, 'max_drawdown': 20},
                'high': {'volatility': 35, 'max_drawdown': 30}
            },
            
            # 벤치마크 필수 여부
            'benchmark_required': True,
            
            # 최소 데이터 요구사항
            'min_data_points': 30,           # 최소 30일 데이터
            'min_trades': 5                  # 최소 5회 거래
        }
```

@dataclass
class MastersWeights:
“”“4대 거장 가중치 설정”””
buffett: float = 0.25      # 워렌 버핏 25%
dalio: float = 0.25        # 레이 달리오 25%  
feynman: float = 0.25      # 리처드 파인만 25%
simons: float = 0.25       # 짐 사이먼스 25%

```
def __post_init__(self):
    """가중치 합계 검증"""
    total = self.buffett + self.dalio + self.feynman + self.simons
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"가중치 합계가 1.0이 아닙니다: {total}")
```

@dataclass
class RiskToleranceSettings:
“”“리스크 허용도별 설정”””

```
# 안전형 (보수적)
conservative: Dict[str, Any] = field(default_factory=lambda: {
    'max_volatility': 15,           # 최대 변동성 15%
    'max_drawdown': 10,             # 최대 낙폭 10%
    'max_single_position': 10,      # 최대 종목 비중 10%
    'min_diversification': 20,      # 최소 분산 종목 수 20개
    'masters_weights': MastersWeights(0.4, 0.4, 0.15, 0.05),  # 버핏+달리오 중심
    'cash_allocation': 20           # 현금 비중 20%
})

# 균형형 (중도적)
balanced: Dict[str, Any] = field(default_factory=lambda: {
    'max_volatility': 20,           # 최대 변동성 20%
    'max_drawdown': 15,             # 최대 낙폭 15%
    'max_single_position': 15,      # 최대 종목 비중 15%
    'min_diversification': 15,      # 최소 분산 종목 수 15개
    'masters_weights': MastersWeights(0.3, 0.3, 0.2, 0.2),   # 균형적 배분
    'cash_allocation': 10           # 현금 비중 10%
})

# 공격형 (적극적)  
aggressive: Dict[str, Any] = field(default_factory=lambda: {
    'max_volatility': 30,           # 최대 변동성 30%
    'max_drawdown': 25,             # 최대 낙폭 25%
    'max_single_position': 20,      # 최대 종목 비중 20%
    'min_diversification': 10,      # 최소 분산 종목 수 10개
    'masters_weights': MastersWeights(0.2, 0.2, 0.2, 0.4),   # 사이먼스 중심
    'cash_allocation': 5            # 현금 비중 5%
})
```

# 전역 설정 인스턴스들

DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()
DEFAULT_RISK_SETTINGS = RiskToleranceSettings()

# 설정 유틸리티 함수들

def get_config_for_user(user_type: str = “balanced”) -> AnalysisConfig:
“”“사용자 타입별 설정 반환”””
config = AnalysisConfig()

```
if user_type == "conservative":
    # 보수적 설정 조정
    config.backtest_config['max_position_weight'] = 0.1
    config.performance_config['risk_free_rate'] = 0.03
elif user_type == "aggressive":
    # 적극적 설정 조정  
    config.backtest_config['max_position_weight'] = 0.25
    config.technical_indicators['signal_threshold'] = 0.5

return config
```

def get_masters_weights(strategy_type: str = “balanced”) -> MastersWeights:
“”“전략 타입별 거장 가중치 반환”””
risk_settings = DEFAULT_RISK_SETTINGS

```
if strategy_type == "conservative":
    return MastersWeights(**risk_settings.conservative['masters_weights'].__dict__)
elif strategy_type == "aggressive":
    return MastersWeights(**risk_settings.aggressive['masters_weights'].__dict__)
else:
    return MastersWeights(**risk_settings.balanced['masters_weights'].__dict__)
```

def validate_config(config: AnalysisConfig) -> bool:
“”“설정 유효성 검증”””
try:
# 기본 검증
assert 0 < config.backtest_config[‘commission_rate’] < 0.01, “수수료율이 유효하지 않음”
assert 0 < config.backtest_config[‘initial_capital’], “초기 자본이 양수여야 함”
assert config.performance_config[‘risk_free_rate’] >= 0, “무위험수익률이 음수일 수 없음”

```
    # 기술적 지표 검증
    assert len(config.technical_indicators['sma_windows']) > 0, "SMA 윈도우가 비어있음"
    assert all(w > 0 for w in config.technical_indicators['sma_windows']), "SMA 윈도우가 양수여야 함"
    
    return True
    
except AssertionError as e:
    print(f"설정 검증 실패: {e}")
    return False
```

if **name** == “**main**”:
# 설정 테스트
print(“📊 Masters Eye 분석 설정 테스트”)
print(”=” * 50)

```
# 기본 설정 테스트
config = DEFAULT_ANALYSIS_CONFIG
print(f"✅ 기본 설정 로드 완료")
print(f"📈 기술적 지표 수: {len(config.technical_indicators)}")
print(f"💰 펀더멘털 업종 수: {len(config.fundamental_analysis['industry_benchmarks'])}")
print(f"🔄 백테스팅 초기자본: {config.backtest_config['initial_capital']:,}원")

# 설정 검증
is_valid = validate_config(config)
print(f"✅ 설정 검증: {'통과' if is_valid else '실패'}")

# 리스크 설정 테스트
conservative_weights = get_masters_weights("conservative")
print(f"🛡️ 보수형 - 버핏 가중치: {conservative_weights.buffett}")

aggressive_weights = get_masters_weights("aggressive") 
print(f"🚀 공격형 - 사이먼스 가중치: {aggressive_weights.simons}")

print("\n🎯 설정 시스템 준비 완료!")
```