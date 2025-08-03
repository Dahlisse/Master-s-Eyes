"""
Masters 모듈 초기화
4대 거장 투자 알고리즘 패키지
"""

from .base import BaseMaster, MasterScore, PortfolioRecommendation, DataQualityManager
from .buffett import BuffettValueInvestor, BuffettScore, create_buffett_portfolio
from .dalio import AllWeatherStrategy, DalioScore, EconomicMachine, create_dalio_portfolio
from .fusion import MastersFusionEngine, InvestmentProfile, MasterWeights, create_masters_fusion_portfolio

__all__ = [
    # Base classes
    'BaseMaster',
    'MasterScore', 
    'PortfolioRecommendation',
    'DataQualityManager',
    
    # Buffett
    'BuffettValueInvestor',
    'BuffettScore',
    'create_buffett_portfolio',
    
    # Dalio
    'AllWeatherStrategy',
    'DalioScore',
    'EconomicMachine',
    'create_dalio_portfolio',
    
    # Fusion
    'MastersFusionEngine',
    'InvestmentProfile',
    'MasterWeights',
    'create_masters_fusion_portfolio'
]

# 버전 정보
__version__ = "1.0.0"
__author__ = "Masters Eye Development Team"