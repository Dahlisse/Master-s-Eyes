– Master’s Eye Database Schema
– PostgreSQL with TimescaleDB extensions

– Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

– =================================
– User Management
– =================================

– Users (나, 엄마)
CREATE TABLE users (
id SERIAL PRIMARY KEY,
username VARCHAR(50) UNIQUE NOT NULL,
display_name VARCHAR(100) NOT NULL,
email VARCHAR(255),
password_hash VARCHAR(255),
user_type VARCHAR(20) NOT NULL DEFAULT ‘individual’, – ‘me’, ‘mom’
risk_tolerance INTEGER DEFAULT 5 CHECK (risk_tolerance >= 1 AND risk_tolerance <= 10),
investment_experience VARCHAR(20) DEFAULT ‘beginner’, – ‘beginner’, ‘intermediate’, ‘advanced’
preferred_language VARCHAR(10) DEFAULT ‘ko’,
notification_settings JSONB DEFAULT ‘{}’,
created_at TIMESTAMPTZ DEFAULT NOW(),
updated_at TIMESTAMPTZ DEFAULT NOW(),
is_active BOOLEAN DEFAULT TRUE
);

– Insert default users
INSERT INTO users (username, display_name, user_type, risk_tolerance, investment_experience) VALUES
(‘me’, ‘나’, ‘me’, 7, ‘intermediate’),
(‘mom’, ‘엄마’, ‘mom’, 3, ‘beginner’);

– =================================
– Market Data (Time Series)
– =================================

– Real-time stock prices (hypertable)
CREATE TABLE market_data (
time TIMESTAMPTZ NOT NULL,
symbol VARCHAR(20) NOT NULL,
exchange VARCHAR(10) NOT NULL DEFAULT ‘KRX’,
open_price DECIMAL(15,2),
high_price DECIMAL(15,2),
low_price DECIMAL(15,2),
close_price DECIMAL(15,2),
volume BIGINT,
amount BIGINT,
market_cap BIGINT,
PRIMARY KEY (time, symbol)
);

– Convert to hypertable
SELECT create_hypertable(‘market_data’, ‘time’, chunk_time_interval => INTERVAL ‘1 day’);

– Create indexes for efficient queries
CREATE INDEX idx_market_data_symbol_time ON market_data (symbol, time DESC);
CREATE INDEX idx_market_data_time ON market_data (time DESC);

– Order book data (real-time bid/ask)
CREATE TABLE orderbook_data (
time TIMESTAMPTZ NOT NULL,
symbol VARCHAR(20) NOT NULL,
bid_prices DECIMAL(15,2)[10], – Array of 10 bid prices
bid_volumes BIGINT[10],       – Array of 10 bid volumes
ask_prices DECIMAL(15,2)[10], – Array of 10 ask prices
ask_volumes BIGINT[10],       – Array of 10 ask volumes
total_bid_volume BIGINT,
total_ask_volume BIGINT,
PRIMARY KEY (time, symbol)
);

SELECT create_hypertable(‘orderbook_data’, ‘time’, chunk_time_interval => INTERVAL ‘1 hour’);

– Trading by investor type
CREATE TABLE trading_by_investor (
time TIMESTAMPTZ NOT NULL,
symbol VARCHAR(20) NOT NULL,
individual_buy BIGINT DEFAULT 0,
individual_sell BIGINT DEFAULT 0,
foreign_buy BIGINT DEFAULT 0,
foreign_sell BIGINT DEFAULT 0,
institution_buy BIGINT DEFAULT 0,
institution_sell BIGINT DEFAULT 0,
PRIMARY KEY (time, symbol)
);

SELECT create_hypertable(‘trading_by_investor’, ‘time’, chunk_time_interval => INTERVAL ‘1 day’);

– =================================
– Company & Stock Information
– =================================

– Listed companies
CREATE TABLE companies (
symbol VARCHAR(20) PRIMARY KEY,
company_name VARCHAR(200) NOT NULL,
company_name_en VARCHAR(200),
exchange VARCHAR(10) NOT NULL DEFAULT ‘KRX’,
market_type VARCHAR(20), – ‘KOSPI’, ‘KOSDAQ’, ‘KONEX’
sector VARCHAR(100),
industry VARCHAR(100),
business_description TEXT,
listing_date DATE,
fiscal_year_end VARCHAR(10),
currency VARCHAR(10) DEFAULT ‘KRW’,
shares_outstanding BIGINT,
created_at TIMESTAMPTZ DEFAULT NOW(),
updated_at TIMESTAMPTZ DEFAULT NOW(),
is_active BOOLEAN DEFAULT TRUE
);

– Financial statements (quarterly)
CREATE TABLE financial_statements (
id SERIAL PRIMARY KEY,
symbol VARCHAR(20) REFERENCES companies(symbol),
report_date DATE NOT NULL,
period_type VARCHAR(10) NOT NULL, – ‘Q1’, ‘Q2’, ‘Q3’, ‘Q4’, ‘Annual’
fiscal_year INTEGER NOT NULL,

```
-- Income Statement
revenue BIGINT,
operating_income BIGINT,
net_income BIGINT,
eps DECIMAL(10,2),

-- Balance Sheet
total_assets BIGINT,
total_liabilities BIGINT,
shareholders_equity BIGINT,

-- Cash Flow
operating_cash_flow BIGINT,
investing_cash_flow BIGINT,
financing_cash_flow BIGINT,
free_cash_flow BIGINT,

-- Key Ratios
roe DECIMAL(8,4),
roa DECIMAL(8,4),
debt_to_equity DECIMAL(8,4),
current_ratio DECIMAL(8,4),

created_at TIMESTAMPTZ DEFAULT NOW(),
UNIQUE(symbol, report_date, period_type)
```

);

– =================================
– Global Economic Data
– =================================

– Global economic indicators
CREATE TABLE global_indicators (
time TIMESTAMPTZ NOT NULL,
indicator_name VARCHAR(50) NOT NULL,
value DECIMAL(15,6),
unit VARCHAR(20),
source VARCHAR(50),
PRIMARY KEY (time, indicator_name)
);

SELECT create_hypertable(‘global_indicators’, ‘time’, chunk_time_interval => INTERVAL ‘1 month’);

– Insert common indicators
INSERT INTO global_indicators (time, indicator_name, value, unit, source) VALUES
(NOW(), ‘USD_KRW’, 1300.00, ‘KRW’, ‘Yahoo Finance’),
(NOW(), ‘US_10Y_TREASURY’, 4.50, ‘Percent’, ‘FRED’),
(NOW(), ‘VIX’, 18.50, ‘Index’, ‘Yahoo Finance’),
(NOW(), ‘DXY’, 103.50, ‘Index’, ‘Yahoo Finance’),
(NOW(), ‘WTI_OIL’, 75.00, ‘USD’, ‘Yahoo Finance’);

– =================================
– News & Sentiment Analysis
– =================================

– News articles
CREATE TABLE news_articles (
id SERIAL PRIMARY KEY,
title VARCHAR(500) NOT NULL,
content TEXT,
url VARCHAR(1000),
source VARCHAR(100),
published_at TIMESTAMPTZ NOT NULL,
symbols VARCHAR(20)[], – Array of related symbols
sentiment_score DECIMAL(5,4), – -1.0 (negative) to 1.0 (positive)
sentiment_magnitude DECIMAL(5,4), – 0.0 to 1.0 (strength)
language VARCHAR(10) DEFAULT ‘ko’,
created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_news_published_at ON news_articles (published_at DESC);
CREATE INDEX idx_news_symbols ON news_articles USING GIN (symbols);

– =================================
– 4 Masters Algorithm Data
– =================================

– Buffett scoring (value investing)
CREATE TABLE buffett_scores (
time TIMESTAMPTZ NOT NULL,
symbol VARCHAR(20) NOT NULL,
intrinsic_value DECIMAL(15,2),
safety_margin DECIMAL(8,4),
moat_score DECIMAL(5,2), – 0-10 scale
management_score DECIMAL(5,2), – 0-10 scale
financial_strength DECIMAL(5,2), – 0-10 scale
total_score DECIMAL(5,2), – 0-100 scale
recommendation VARCHAR(20), – ‘strong_buy’, ‘buy’, ‘hold’, ‘sell’, ‘strong_sell’
PRIMARY KEY (time, symbol)
);

SELECT create_hypertable(‘buffett_scores’, ‘time’, chunk_time_interval => INTERVAL ‘1 day’);

– Dalio scoring (macro economic)
CREATE TABLE dalio_scores (
time TIMESTAMPTZ NOT NULL,
symbol VARCHAR(20) NOT NULL,
economic_environment VARCHAR(20), – ‘growth_up_inflation_down’, etc.
correlation_score DECIMAL(5,2),
risk_parity_weight DECIMAL(8,6),
economic_cycle_stage VARCHAR(20),
macro_score DECIMAL(5,2), – 0-100 scale
PRIMARY KEY (time, symbol)
);

SELECT create_hypertable(‘dalio_scores’, ‘time’, chunk_time_interval => INTERVAL ‘1 day’);

– Feynman scoring (scientific approach)
CREATE TABLE feynman_scores (
time TIMESTAMPTZ NOT NULL,
symbol VARCHAR(20) NOT NULL,
simplicity_score DECIMAL(5,2), – Business model simplicity
uncertainty_level DECIMAL(5,2), – Quantified uncertainty
confidence_interval_lower DECIMAL(15,2),
confidence_interval_upper DECIMAL(15,2),
monte_carlo_mean DECIMAL(15,2),
monte_carlo_std DECIMAL(15,2),
scientific_score DECIMAL(5,2), – 0-100 scale
PRIMARY KEY (time, symbol)
);

SELECT create_hypertable(‘feynman_scores’, ‘time’, chunk_time_interval => INTERVAL ‘1 day’);

– Simons scoring (quantitative)
CREATE TABLE simons_scores (
time TIMESTAMPTZ NOT NULL,
symbol VARCHAR(20) NOT NULL,
value_factor DECIMAL(8,4),
growth_factor DECIMAL(8,4),
quality_factor DECIMAL(8,4),
momentum_factor DECIMAL(8,4),
low_vol_factor DECIMAL(8,4),
alpha_score DECIMAL(8,4),
quant_score DECIMAL(5,2), – 0-100 scale
PRIMARY KEY (time, symbol)
);

SELECT create_hypertable(‘simons_scores’, ‘time’, chunk_time_interval => INTERVAL ‘1 day’);

– =================================
– Portfolio Management
– =================================

– User portfolios
CREATE TABLE portfolios (
id SERIAL PRIMARY KEY,
user_id INTEGER REFERENCES users(id),
name VARCHAR(200) NOT NULL,
strategy_type VARCHAR(20) NOT NULL, – ‘conservative’, ‘balanced’, ‘aggressive’
total_value DECIMAL(15,2) DEFAULT 0,
cash_balance DECIMAL(15,2) DEFAULT 0,
target_allocation JSONB, – JSON of symbol: weight pairs
current_allocation JSONB, – Current actual allocation

```
-- 4 Masters weights
buffett_weight DECIMAL(5,4) DEFAULT 0.25,
dalio_weight DECIMAL(5,4) DEFAULT 0.25,
feynman_weight DECIMAL(5,4) DEFAULT 0.25,
simons_weight DECIMAL(5,4) DEFAULT 0.25,

rebalance_threshold DECIMAL(5,4) DEFAULT 0.05, -- 5%
rebalance_frequency VARCHAR(20) DEFAULT 'monthly',
last_rebalanced_at TIMESTAMPTZ,

created_at TIMESTAMPTZ DEFAULT NOW(),
updated_at TIMESTAMPTZ DEFAULT NOW(),
is_active BOOLEAN DEFAULT TRUE
```

);

– Portfolio positions (current holdings)
CREATE TABLE portfolio_positions (
id SERIAL PRIMARY KEY,
portfolio_id INTEGER REFERENCES portfolios(id),
symbol VARCHAR(20) NOT NULL,
quantity INTEGER NOT NULL DEFAULT 0,
average_cost DECIMAL(15,2),
current_price DECIMAL(15,2),
current_value DECIMAL(15,2),
weight DECIMAL(8,6), – Current weight in portfolio
target_weight DECIMAL(8,6), – Target weight
pnl DECIMAL(15,2), – Profit & Loss
pnl_percent DECIMAL(8,4),
updated_at TIMESTAMPTZ DEFAULT NOW(),
UNIQUE(portfolio_id, symbol)
);

– Portfolio performance history
CREATE TABLE portfolio_performance (
time TIMESTAMPTZ NOT NULL,
portfolio_id INTEGER NOT NULL REFERENCES portfolios(id),
total_value DECIMAL(15,2),
daily_return DECIMAL(8,6),
cumulative_return DECIMAL(8,6),
benchmark_return DECIMAL(8,6), – KOSPI return
alpha DECIMAL(8,6),
beta DECIMAL(8,6),
sharpe_ratio DECIMAL(8,6),
max_drawdown DECIMAL(8,6),
PRIMARY KEY (time, portfolio_id)
);

SELECT create_hypertable(‘portfolio_performance’, ‘time’, chunk_time_interval => INTERVAL ‘1 month’);