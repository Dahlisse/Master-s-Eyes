# backend/app/analysis/backtest_framework.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings(‘ignore’)

@dataclass
class BacktestConfig:
“”“백테스트 설정”””
start_date: str = ‘2020-01-01’
end_date: str = ‘2024-12-31’
initial_capital: float = 100_000_000  # 1억원
commission_rate: float = 0.0015  # 0.15%
tax_rate: float = 0.0025  # 0.25% (거래세 + 농특세)
slippage: float = 0.001  # 0.1%
rebalance_frequency: str = ‘monthly’  # daily, weekly, monthly, quarterly
benchmark: str = ‘KOSPI’
risk_free_rate: float = 0.025  # 2.5%

@dataclass
class Trade:
“”“거래 기록”””
date: datetime
ticker: str
action: str  # ‘BUY’ or ‘SELL’
quantity: int
price: float
amount: float
commission: float
tax: float
total_cost: float

@dataclass
class Position:
“”“포지션 정보”””
ticker: str
quantity: int = 0
avg_price: float = 0.0
market_value: float = 0.0
unrealized_pnl: float = 0.0
weight: float = 0.0

@dataclass
class PerformanceMetrics:
“”“성과 지표”””
total_return: float = 0.0
annualized_return: float = 0.0
volatility: float = 0.0
sharpe_ratio: float = 0.0
sortino_ratio: float = 0.0
max_drawdown: float = 0.0
max_drawdown_duration: int = 0
win_rate: float = 0.0
profit_factor: float = 0.0
calmar_ratio: float = 0.0
var_95: float = 0.0
cvar_95: float = 0.0
beta: float = 0.0
alpha: float = 0.0
information_ratio: float = 0.0
trades_count: int = 0

class BacktestEngine:
“”“백테스팅 엔진”””

```
def __init__(self, config: BacktestConfig):
    self.config = config
    self.portfolio_value_history = []
    self.positions: Dict[str, Position] = {}
    self.trades: List[Trade] = []
    self.cash = config.initial_capital
    self.total_commission = 0.0
    self.total_tax = 0.0
    
def run_backtest(self, strategy_func: Callable, market_data: Dict[str, pd.DataFrame], 
                benchmark_data: pd.DataFrame = None) -> Dict:
    """백테스트 실행"""
    try:
        # 데이터 전처리
        aligned_data = self._align_market_data(market_data)
        if aligned_data.empty:
            raise ValueError("시장 데이터가 비어있습니다")
        
        # 백테스트 실행
        for date in aligned_data.index:
            current_data = aligned_data.loc[date]
            
            # 전략 신호 생성
            signals = strategy_func(current_data, self.positions, date)
            
            # 리밸런싱 실행
            if self._should_rebalance(date):
                self._execute_rebalance(signals, current_data, date)
            
            # 포트폴리오 가치 업데이트
            portfolio_value = self._calculate_portfolio_value(current_data)
            self.portfolio_value_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash
            })
        
        # 성과 분석
        performance = self._calculate_performance_metrics(benchmark_data)
        
        return {
            'performance': performance,
            'portfolio_history': pd.DataFrame(self.portfolio_value_history),
            'trades': self.trades,
            'final_positions': self.positions,
            'config': self.config
        }
        
    except Exception as e:
        print(f"백테스트 실행 오류: {e}")
        return self._get_empty_result()

def _align_market_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """시장 데이터 정렬 및 병합"""
    try:
        aligned_dfs = []
        
        for ticker, df in market_data.items():
            if df.empty:
                continue
                
            # 날짜 컬럼 정리
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # 필요한 컬럼만 선택하고 이름 변경
            required_cols = ['close', 'volume']
            available_cols = [col for col in required_cols if col in df.columns]
            
            if not available_cols:
                continue
            
            ticker_df = df[available_cols].copy()
            ticker_df.columns = [f"{ticker}_{col}" for col in ticker_df.columns]
            aligned_dfs.append(ticker_df)
        
        if not aligned_dfs:
            return pd.DataFrame()
        
        # 모든 데이터 병합
        result = pd.concat(aligned_dfs, axis=1, sort=True)
        
        # 날짜 범위 필터링
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        result = result.loc[start_date:end_date]
        
        # 결측치 전진 채우기
        result = result.fillna(method='ffill').fillna(method='bfill')
        
        return result
        
    except Exception as e:
        print(f"데이터 정렬 오류: {e}")
        return pd.DataFrame()

def _should_rebalance(self, date: datetime) -> bool:
    """리밸런싱 여부 판단"""
    if not self.portfolio_value_history:
        return True
        
    last_rebalance_date = self.portfolio_value_history[-1]['date']
    
    if self.config.rebalance_frequency == 'daily':
        return True
    elif self.config.rebalance_frequency == 'weekly':
        return (date - last_rebalance_date).days >= 7
    elif self.config.rebalance_frequency == 'monthly':
        return date.month != last_rebalance_date.month
    elif self.config.rebalance_frequency == 'quarterly':
        return date.month != last_rebalance_date.month and date.month in [1, 4, 7, 10]
    
    return False

def _execute_rebalance(self, signals: Dict[str, float], market_data: pd.Series, date: datetime):
    """리밸런싱 실행"""
    try:
        current_portfolio_value = self._calculate_portfolio_value(market_data)
        
        for ticker, target_weight in signals.items():
            if target_weight < 0 or target_weight > 1:
                continue
            
            price_col = f"{ticker}_close"
            if price_col not in market_data or pd.isna(market_data[price_col]):
                continue
            
            current_price = market_data[price_col]
            target_value = current_portfolio_value * target_weight
            
            # 현재 포지션
            current_position = self.positions.get(ticker, Position(ticker))
            current_value = current_position.quantity * current_price
            
            # 거래 필요 금액
            trade_value = target_value - current_value
            
            if abs(trade_value) > current_portfolio_value * 0.01:  # 1% 이상 차이날 때만 거래
                if trade_value > 0:  # 매수
                    self._execute_buy(ticker, trade_value, current_price, date)
                else:  # 매도
                    self._execute_sell(ticker, abs(trade_value), current_price, date)
                    
    except Exception as e:
        print(f"리밸런싱 실행 오류: {e}")

def _execute_buy(self, ticker: str, target_value: float, price: float, date: datetime):
    """매수 실행"""
    try:
        # 슬리피지 적용
        execution_price = price * (1 + self.config.slippage)
        
        # 거래 가능 수량 계산
        max_quantity = int(self.cash / execution_price)
        target_quantity = int(target_value / execution_price)
        quantity = min(max_quantity, target_quantity)
        
        if quantity <= 0:
            return
        
        # 거래 비용 계산
        gross_amount = quantity * execution_price
        commission = gross_amount * self.config.commission_rate
        total_cost = gross_amount + commission
        
        if total_cost > self.cash:
            return
        
        # 포지션 업데이트
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker)
        
        position = self.positions[ticker]
        old_value = position.quantity * position.avg_price
        new_value = old_value + gross_amount
        position.quantity += quantity
        position.avg_price = new_value / position.quantity if position.quantity > 0 else 0
        
        # 현금 차감
        self.cash -= total_cost
        self.total_commission += commission
        
        # 거래 기록
        trade = Trade(
            date=date,
            ticker=ticker,
            action='BUY',
            quantity=quantity,
            price=execution_price,
            amount=gross_amount,
            commission=commission,
            tax=0,
            total_cost=total_cost
        )
        self.trades.append(trade)
        
    except Exception as e:
        print(f"매수 실행 오류: {e}")

def _execute_sell(self, ticker: str, target_value: float, price: float, date: datetime):
    """매도 실행"""
    try:
        if ticker not in self.positions or self.positions[ticker].quantity <= 0:
            return
        
        # 슬리피지 적용
        execution_price = price * (1 - self.config.slippage)
        
        position = self.positions[ticker]
        target_quantity = int(target_value / execution_price)
        quantity = min(position.quantity, target_quantity)
        
        if quantity <= 0:
            return
        
        # 거래 비용 계산
        gross_amount = quantity * execution_price
        commission = gross_amount * self.config.commission_rate
        tax = gross_amount * self.config.tax_rate
        net_amount = gross_amount - commission - tax
        
        # 포지션 업데이트
        position.quantity -= quantity
        if position.quantity <= 0:
            del self.positions[ticker]
        
        # 현금 증가
        self.cash += net_amount
        self.total_commission += commission
        self.total_tax += tax
        
        # 거래 기록
        trade = Trade(
            date=date,
            ticker=ticker,
            action='SELL',
            quantity=quantity,
            price=execution_price,
            amount=gross_amount,
            commission=commission,
            tax=tax,
            total_cost=net_amount
        )
        self.trades.append(trade)
        
    except Exception as e:
        print(f"매도 실행 오류: {e}")

def _calculate_portfolio_value(self, market_data: pd.Series) -> float:
    """포트폴리오 가치 계산"""
    total_value = self.cash
    
    for ticker, position in self.positions.items():
        price_col = f"{ticker}_close"
        if price_col in market_data and not pd.isna(market_data[price_col]):
            current_price = market_data[price_col]
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = position.market_value - (position.quantity * position.avg_price)
            total_value += position.market_value
    
    return total_value

def _calculate_performance_metrics(self, benchmark_data: pd.DataFrame = None) -> PerformanceMetrics:
    """성과 지표 계산"""
    try:
        if not self.portfolio_value_history:
            return PerformanceMetrics()
        
        # 포트폴리오 수익률 계산
        portfolio_df = pd.DataFrame(self.portfolio_value_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)
        
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        if len(returns) == 0:
            return PerformanceMetrics()
        
        # 기본 지표
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital - 1) * 100
        trading_days = len(returns)
        years = trading_days / 252
        annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        volatility = returns.std() * np.sqrt(252) * 100
        
        # 샤프 비율
        excess_returns = returns - (self.config.risk_free_rate / 252)
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 소르티노 비율
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - self.config.risk_free_rate * 100) / downside_std if downside_std > 0 else 0
        
        # 최대 낙폭 계산
        peak = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - peak) / peak * 100
        max_drawdown = abs(drawdown.min())
        
        # 최대 낙폭 지속 기간
        max_dd_duration = 0
        current_dd_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        # 승률 계산
        winning_trades = [t for t in self.trades if t.action == 'SELL' and self._calculate_trade_pnl(t) > 0]
        total_sell_trades = [t for t in self.trades if t.action == 'SELL']
        win_rate = len(winning_trades) / len(total_sell_trades) * 100 if total_sell_trades else 0
        
        # 수익 팩터
        gross_profit = sum(self._calculate_trade_pnl(t) for t in total_sell_trades if self._calculate_trade_pnl(t) > 0)
        gross_loss = abs(sum(self._calculate_trade_pnl(t) for t in total_sell_trades if self._calculate_trade_pnl(t) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 칼마 비율
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR 및 CVaR (95% 신뢰구간)
        var_95 = np.percentile(returns, 5) * 100
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # 베타 및 알파 (벤치마크 대비)
        beta, alpha = 0.0, 0.0
        information_ratio = 0.0
        
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_returns = benchmark_data['close'].pct_change().dropna()
            aligned_returns = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns[0]) > 1:
                port_ret = aligned_returns[0]
                bench_ret = aligned_returns[1]
                
                # 베타 계산
                covariance = np.cov(port_ret, bench_ret)[0][1]
                benchmark_variance = np.var(bench_ret)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # 알파 계산
                alpha = (annualized_return - self.config.risk_free_rate * 100) - beta * (bench_ret.mean() * 252 * 100 - self.config.risk_free_rate * 100)
                
                # 정보 비율
                active_returns = port_ret - bench_ret
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        return PerformanceMetrics(
            total_return=round(total_return, 2),
            annualized_return=round(annualized_return, 2),
            volatility=round(volatility, 2),
            sharpe_ratio=round(sharpe_ratio, 3),
            sortino_ratio=round(sortino_ratio, 3),
            max_drawdown=round(max_drawdown, 2),
            max_drawdown_duration=max_dd_duration,
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            calmar_ratio=round(calmar_ratio, 3),
            var_95=round(var_95, 2),
            cvar_95=round(cvar_95, 2),
            beta=round(beta, 3),
            alpha=round(alpha, 2),
            information_ratio=round(information_ratio, 3),
            trades_count=len(self.trades)
        )
        
    except Exception as e:
        print(f"성과 지표 계산 오류: {e}")
        return PerformanceMetrics()

def _calculate_trade_pnl(self, sell_trade: Trade) -> float:
    """개별 거래 손익 계산"""
    # 간단하게 매도 거래의 실현 손익만 계산
    # 실제로는 매수/매도 쌍을 매칭해야 하지만 여기서는 단순화
    if sell_trade.action != 'SELL':
        return 0
    
    ticker = sell_trade.ticker
    if ticker in self.positions:
        avg_buy_price = self.positions[ticker].avg_price
        return (sell_trade.price - avg_buy_price) * sell_trade.quantity - sell_trade.commission - sell_trade.tax
    
    return 0

def _get_empty_result(self) -> Dict:
    """빈 결과 반환"""
    return {
        'performance': PerformanceMetrics(),
        'portfolio_history': pd.DataFrame(),
        'trades': [],
        'final_positions': {},
        'config': self.config
    }
```

class MonteCarloBacktest:
“”“몬테카를로 백테스팅”””

```
def __init__(self, base_config: BacktestConfig):
    self.base_config = base_config

def run_monte_carlo(self, strategy_func: Callable, market_data: Dict[str, pd.DataFrame],
                   num_simulations: int = 1000, confidence_level: float = 0.95) -> Dict:
    """몬테카를로 시뮬레이션 실행"""
    try:
        results = []
        
        for i in range(num_simulations):
            # 랜덤 시드 설정
            np.random.seed(i)
            
            # 노이즈가 추가된 데이터 생성
            noisy_data = self._add_market_noise(market_data)
            
            # 백테스트 실행
            config = self._get_random_config()
            engine = BacktestEngine(config)
            result = engine.run_backtest(strategy_func, noisy_data)
            
            results.append(result['performance'])
            
            if (i + 1) % 100 == 0:
                print(f"몬테카를로 시뮬레이션 진행: {i + 1}/{num_simulations}")
        
        # 통계 분석
        returns = [r.total_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        
        alpha = 1 - confidence_level
        
        return {
            'num_simulations': num_simulations,
            'confidence_level': confidence_level,
            'returns': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentile_5': np.percentile(returns, alpha/2 * 100),
                'percentile_95': np.percentile(returns, (1-alpha/2) * 100),
                'probability_positive': np.mean([r > 0 for r in returns]) * 100
            },
            'sharpe_ratios': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'percentile_5': np.percentile(sharpe_ratios, alpha/2 * 100),
                'percentile_95': np.percentile(sharpe_ratios, (1-alpha/2) * 100)
            },
            'max_drawdowns': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'percentile_5': np.percentile(max_drawdowns, alpha/2 * 100),
                'percentile_95': np.percentile(max_drawdowns, (1-alpha/2) * 100)
            },
            'all_results': results
        }
        
    except Exception as e:
        print(f"몬테카를로 시뮬레이션 오류: {e}")
        return {}

def _add_market_noise(self, market_data: Dict[str, pd.DataFrame], noise_level: float = 0.02) -> Dict[str, pd.DataFrame]:
    """시장 데이터에 노이즈 추가"""
    noisy_data = {}
    
    for ticker, df in market_data.items():
        if df.empty:
            noisy_data[ticker] = df
            continue
            
        noisy_df = df.copy()
        
        # 가격 데이터에 노이즈 추가
        for col in ['open', 'high', 'low', 'close']:
            if col in noisy_df.columns:
                noise = np.random.normal(1, noise_level, len(noisy_df))
                noisy_df[col] = noisy_df[col] * noise
        
        # 거래량에 노이즈 추가
        if 'volume' in noisy_df.columns:
            volume_noise = np.random.normal(1, noise_level * 2, len(noisy_df))
            noisy_df['volume'] = noisy_df['volume'] * np.abs(volume_noise)
        
        noisy_data[ticker] = noisy_df
    
    return noisy_data

def _get_random_config(self) -> BacktestConfig:
    """랜덤 설정 생성"""
    config = BacktestConfig()
    
    # 일부 파라미터를 랜덤하게 변경
    config.commission_rate = self.base_config.commission_rate * np.random.uniform(0.8, 1.2)
    config.slippage = self.base_config.slippage * np.random.uniform(0.5, 1.5)
    
    return config
```

# 샘플 전략 함수들

def buy_and_hold_strategy(market_data: pd.Series, positions: Dict, date: datetime) -> Dict[str, float]:
“”“바이앤홀드 전략”””
if not positions:  # 첫 거래일
# 균등 분산 투자
tickers = [col.replace(’_close’, ‘’) for col in market_data.index if col.endswith(’_close’)]
weight = 1.0 / len(tickers) if tickers else 0
return {ticker: weight for ticker in tickers}
else:
return {}  # 추가 거래 없음

def momentum_strategy(market_data: pd.Series, positions: Dict, date: datetime) -> Dict[str, float]:
“”“모멘텀 전략 (단순 버전)”””
signals = {}

```
# 각 종목별로 모멘텀 점수 계산
tickers = [col.replace('_close', '') for col in market_data.index if col.endswith('_close')]

for ticker in tickers:
    close_col = f"{ticker}_close"
    if close_col in market_data.index:
        # 여기서는 단순하게 구현 - 실제로는 과거 데이터 필요
        signals[ticker] = 0.2  # 균등 분산

return signals
```

# 사용 예시

if **name** == “**main**”:
# 테스트 설정
config = BacktestConfig(
start_date=‘2023-01-01’,
end_date=‘2024-12-31’,
initial_capital=100_000_000,
rebalance_frequency=‘monthly’
)

```
# 가상의 시장 데이터 생성
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
tickers = ['005930', '000660', '035420']  # 삼성전자, SK하이닉스, NAVER

market_data = {}
np.random.seed(42)

for ticker in tickers:
    # 가상의 주가 데이터
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 50000 * np.cumprod(1 + returns)
    volumes = np.random.randint(100000, 1000000, len(dates))
    
    market_data[ticker] = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volumes
    })

# 백테스트 실행
engine = BacktestEngine(config)
result = engine.run_backtest(buy_and_hold_strategy, market_data)

print("📊 백테스트 결과")
print(f"총 수익률: {result['performance'].total_return:.2f}%")
print(f"연환산 수익률: {result['performance'].annualized_return:.2f}%")
print(f"샤프 비율: {result['performance'].sharpe_ratio:.3f}")
print(f"최대 낙폭: {result['performance'].max_drawdown:.2f}%")
print(f"거래 횟수: {result['performance'].trades_count}")

# 몬테카를로 시뮬레이션
mc_engine = MonteCarloBacktest(config)
mc_result = mc_engine.run_monte_carlo(buy_and_hold_strategy, market_data, num_simulations=100)

if mc_result:
    print("\n🎲 몬테카를로 시뮬레이션 결과")
    print(f"평균 수익률: {mc_result['returns']['mean']:.2f}%")
    print(f"수익률 95% 신뢰구간: [{mc_result['returns']['percentile_5']:.2f}%, {mc_result['returns']['percentile_95']:.2f}%]")
    print(f"양수 수익률 확률: {mc_result['returns']['probability_positive']:.1f}%")
```