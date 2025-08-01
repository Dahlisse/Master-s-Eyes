# backend/app/analysis/fundamental_engine.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings(‘ignore’)

@dataclass
class FinancialRatios:
“”“재무비율 데이터 클래스”””
# 수익성 지표
roe: float = 0.0  # 자기자본이익률
roa: float = 0.0  # 총자산이익률
roic: float = 0.0  # 투자자본이익률
gross_margin: float = 0.0  # 매출총이익률
operating_margin: float = 0.0  # 영업이익률
net_margin: float = 0.0  # 순이익률

```
# 성장성 지표
revenue_growth: float = 0.0  # 매출성장률
operating_growth: float = 0.0  # 영업이익성장률
net_growth: float = 0.0  # 순이익성장률
asset_growth: float = 0.0  # 자산성장률

# 안정성 지표
debt_ratio: float = 0.0  # 부채비율
debt_to_equity: float = 0.0  # 부채대비자기자본비율
current_ratio: float = 0.0  # 유동비율
quick_ratio: float = 0.0  # 당좌비율
interest_coverage: float = 0.0  # 이자보상배율

# 활동성 지표
asset_turnover: float = 0.0  # 총자산회전율
inventory_turnover: float = 0.0  # 재고자산회전율
receivable_turnover: float = 0.0  # 매출채권회전율

# 현금흐름 지표
operating_cash_flow: float = 0.0  # 영업현금흐름
free_cash_flow: float = 0.0  # 잉여현금흐름
cash_conversion_cycle: float = 0.0  # 현금전환주기
```

@dataclass
class ValuationMetrics:
“”“밸류에이션 지표 데이터 클래스”””
per: float = 0.0  # 주가수익비율
pbr: float = 0.0  # 주가순자산비율
pcr: float = 0.0  # 주가현금흐름비율
psr: float = 0.0  # 주가매출비율
ev_ebitda: float = 0.0  # EV/EBITDA
ev_sales: float = 0.0  # EV/Sales
peg_ratio: float = 0.0  # PEG 비율
dividend_yield: float = 0.0  # 배당수익률
book_value_per_share: float = 0.0  # 주당순자산가치
earnings_per_share: float = 0.0  # 주당순이익

class FundamentalEngine:
“”“펀더멘털 분석 엔진”””

```
def __init__(self):
    # 업종별 평균 지표 (한국 주식시장 기준)
    self.industry_benchmarks = {
        'default': {
            'roe': 8.0, 'roa': 4.0, 'debt_ratio': 45.0,
            'current_ratio': 150.0, 'per': 12.0, 'pbr': 1.2
        },
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
        }
    }

def calculate_financial_ratios(self, financial_data: Dict) -> FinancialRatios:
    """재무비율 계산"""
    try:
        # 필요한 재무제표 항목들
        revenue = financial_data.get('revenue', 0)
        gross_profit = financial_data.get('gross_profit', 0)
        operating_income = financial_data.get('operating_income', 0)
        net_income = financial_data.get('net_income', 0)
        total_assets = financial_data.get('total_assets', 1)
        total_equity = financial_data.get('total_equity', 1)
        total_debt = financial_data.get('total_debt', 0)
        current_assets = financial_data.get('current_assets', 0)
        current_liabilities = financial_data.get('current_liabilities', 1)
        cash = financial_data.get('cash', 0)
        inventory = financial_data.get('inventory', 0)
        accounts_receivable = financial_data.get('accounts_receivable', 0)
        operating_cash_flow = financial_data.get('operating_cash_flow', 0)
        capex = financial_data.get('capex', 0)
        interest_expense = financial_data.get('interest_expense', 0)
        
        # 전년도 데이터 (성장률 계산용)
        prev_revenue = financial_data.get('prev_revenue', revenue)
        prev_operating_income = financial_data.get('prev_operating_income', operating_income)
        prev_net_income = financial_data.get('prev_net_income', net_income)
        prev_assets = financial_data.get('prev_assets', total_assets)
        
        # 수익성 지표 계산
        roe = (net_income / total_equity * 100) if total_equity > 0 else 0
        roa = (net_income / total_assets * 100) if total_assets > 0 else 0
        invested_capital = total_equity + total_debt
        roic = (operating_income * 0.75 / invested_capital * 100) if invested_capital > 0 else 0  # 세후 영업이익 가정
        gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0
        operating_margin = (operating_income / revenue * 100) if revenue > 0 else 0
        net_margin = (net_income / revenue * 100) if revenue > 0 else 0
        
        # 성장성 지표 계산
        revenue_growth = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
        operating_growth = ((operating_income - prev_operating_income) / prev_operating_income * 100) if prev_operating_income > 0 else 0
        net_growth = ((net_income - prev_net_income) / prev_net_income * 100) if prev_net_income > 0 else 0
        asset_growth = ((total_assets - prev_assets) / prev_assets * 100) if prev_assets > 0 else 0
        
        # 안정성 지표 계산
        debt_ratio = (total_debt / total_assets * 100) if total_assets > 0 else 0
        debt_to_equity = (total_debt / total_equity * 100) if total_equity > 0 else 0
        current_ratio = (current_assets / current_liabilities * 100) if current_liabilities > 0 else 0
        quick_assets = current_assets - inventory
        quick_ratio = (quick_assets / current_liabilities * 100) if current_liabilities > 0 else 0
        interest_coverage = (operating_income / interest_expense) if interest_expense > 0 else 999
        
        # 활동성 지표 계산
        asset_turnover = (revenue / total_assets) if total_assets > 0 else 0
        inventory_turnover = (revenue / inventory) if inventory > 0 else 0
        receivable_turnover = (revenue / accounts_receivable) if accounts_receivable > 0 else 0
        
        # 현금흐름 지표 계산
        free_cash_flow = operating_cash_flow - capex
        # 현금전환주기 = 재고보유기간 + 매출채권회수기간 - 매입채무지급기간
        days_inventory = (inventory / revenue * 365) if revenue > 0 and inventory > 0 else 0
        days_receivable = (accounts_receivable / revenue * 365) if revenue > 0 and accounts_receivable > 0 else 0
        days_payable = 30  # 가정값 (실제로는 매입채무 데이터 필요)
        cash_conversion_cycle = days_inventory + days_receivable - days_payable
        
        return FinancialRatios(
            roe=round(roe, 2),
            roa=round(roa, 2),
            roic=round(roic, 2),
            gross_margin=round(gross_margin, 2),
            operating_margin=round(operating_margin, 2),
            net_margin=round(net_margin, 2),
            revenue_growth=round(revenue_growth, 2),
            operating_growth=round(operating_growth, 2),
            net_growth=round(net_growth, 2),
            asset_growth=round(asset_growth, 2),
            debt_ratio=round(debt_ratio, 2),
            debt_to_equity=round(debt_to_equity, 2),
            current_ratio=round(current_ratio, 2),
            quick_ratio=round(quick_ratio, 2),
            interest_coverage=round(interest_coverage, 2),
            asset_turnover=round(asset_turnover, 2),
            inventory_turnover=round(inventory_turnover, 2),
            receivable_turnover=round(receivable_turnover, 2),
            operating_cash_flow=operating_cash_flow,
            free_cash_flow=free_cash_flow,
            cash_conversion_cycle=round(cash_conversion_cycle, 1)
        )
        
    except Exception as e:
        print(f"재무비율 계산 오류: {e}")
        return FinancialRatios()

def calculate_valuation_metrics(self, market_data: Dict, financial_data: Dict) -> ValuationMetrics:
    """밸류에이션 지표 계산"""
    try:
        # 시장 데이터
        market_cap = market_data.get('market_cap', 0)
        share_price = market
```