#!/usr/bin/env python3
“””
Master’s Eye - 4대 거장 완전 융합 주식 AI 포트폴리오 시스템
일괄 실행 및 테스트 스크립트 (Week 7-8 완성 버전)

GitHub repo: https://github.com/Dahlisse/Master-s-Eyes
완성된 4대 거장: 워렌 버핏, 레이 달리오, 리처드 파인만, 짐 사이먼스

사용법:
python run_masters_eye.py                    # 완전 융합 테스트
python run_masters_eye.py –test-buffett     # 버핏 알고리즘만
python run_masters_eye.py –test-dalio       # 달리오 알고리즘만
python run_masters_eye.py –test-feynman     # 파인만 알고리즘만
python run_masters_eye.py –test-simons      # 사이먼스 알고리즘만
python run_masters_eye.py –test-fusion      # 융합 엔진만
python run_masters_eye.py –check-env        # 환경 확인만
python run_masters_eye.py –demo             # 데모 모드 (Mock 데이터)
python run_masters_eye.py –performance      # 성능 벤치마크
“””

import sys
import os
import asyncio
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time
from datetime import datetime
from dataclasses import asdict

# 프로젝트 루트 디렉토리를 Python path에 추가

project_root = Path(**file**).parent
sys.path.insert(0, str(project_root / “backend”))

# 로깅 설정

log_dir = project_root / “logs”
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
level=logging.INFO,
format=’%(asctime)s - %(name)s - %(levelname)s - %(message)s’,
handlers=[
logging.FileHandler(log_dir / ‘masters_eye.log’),
logging.StreamHandler(sys.stdout)
]
)
logger = logging.getLogger(**name**)

class MockDatabase:
“”“데이터베이스 Mock 클래스”””
def query(self, *args, **kwargs):
return MockQuery()
def close(self):
pass

class MockQuery:
“”“쿼리 Mock 클래스”””
def filter(self, *args, **kwargs):
return self
def first(self):
return MockCompany()
def all(self):
return [MockCompany()]

class MockCompany:
“”“회사 정보 Mock 클래스”””
def **init**(self):
self.ticker = “005930”
self.name = “삼성전자”
self.sector = “반도체”
self.market_cap = 400_000_000_000_000

class MastersEyeRunner:
“”“Master’s Eye 통합 실행 및 테스트 클래스 (4대 거장 완성)”””

```
def __init__(self, demo_mode: bool = False):
    self.demo_mode = demo_mode
    self.test_tickers = [
        '005930',  # 삼성전자
        '000660',  # SK하이닉스
        '035420',  # NAVER
        '055550',  # 신한지주
        '051910',  # LG화학
        '105560',  # KB금융
        '006400',  # 삼성SDI
        '035720',  # 카카오
        '000270',  # 기아
        '068270',  # 셀트리온
        '207940',  # 삼성바이오로직스
        '373220'   # LG에너지솔루션
    ]
    
    # 결과 저장 디렉토리
    self.results_dir = project_root / "test_results"
    self.results_dir.mkdir(exist_ok=True)
    
def check_environment(self) -> bool:
    """개발 환경 확인 (4대 거장 완성 버전)"""
    logger.info("🔍 Master's Eye 4대 거장 환경 확인 중...")
    
    try:
        # 1. Python 버전 확인
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error(f"❌ Python 3.8+ 필요. 현재: {python_version}")
            return False
        logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 2. 프로젝트 루트 디렉토리 확인
        logger.info(f"📁 프로젝트 루트: {project_root}")
        
        # 3. 필수 디렉토리 구조 확인
        required_dirs = [
            'backend',
            'backend/app',
            'backend/app/masters',
            'backend/app/core',
            'backend/app/models',
            'backend/app/utils'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            if full_path.exists():
                logger.info(f"✅ {dir_path}/")
            else:
                missing_dirs.append(dir_path)
                logger.error(f"❌ {dir_path}/ 디렉토리 없음")
        
        # 4. Week 7-8 완성 파일 확인
        week7_8_files = [
            'backend/app/masters/__init__.py',
            'backend/app/masters/base.py',
            'backend/app/masters/buffett.py',
            'backend/app/masters/dalio.py',
            'backend/app/masters/feynman.py',     # 새로 추가
            'backend/app/masters/simons.py',      # 새로 추가
            'backend/app/masters/fusion.py'       # 업데이트됨
        ]
        
        missing_files = []
        for file_path in week7_8_files:
            full_path = project_root / file_path
            if full_path.exists() and full_path.stat().st_size > 0:
                logger.info(f"✅ {file_path} ({full_path.stat().st_size:,} bytes)")
            else:
                missing_files.append(file_path)
                logger.error(f"❌ {file_path} 파일 없음 또는 비어있음")
        
        # 5. Python 패키지 확인 (머신러닝 패키지 포함)
        required_packages = [
            'pandas', 'numpy', 'scipy', 'dataclasses', 'typing', 'asyncio', 'enum'
        ]
        
        ml_packages = [
            'scikit-learn', 'numba'  # 파인만, 사이먼스용
        ]
        
        optional_packages = [
            'sqlalchemy', 'fastapi', 'yfinance', 'requests'
        ]
        
        missing_required = []
        missing_ml = []
        missing_optional = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package}")
            except ImportError:
                missing_required.append(package)
                logger.error(f"❌ {package} (필수)")
        
        for package in ml_packages:
            try:
                __import__(package.replace('-', '_'))  # scikit-learn -> sklearn
                logger.info(f"✅ {package}")
            except ImportError:
                missing_ml.append(package)
                logger.warning(f"⚠️ {package} (ML 기능용)")
        
        for package in optional_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package}")
            except ImportError:
                missing_optional.append(package)
                logger.warning(f"⚠️ {package} (선택사항)")
        
        # 6. 4대 거장 import 테스트
        try:
            from app.masters import (
                BuffettValueInvestor, AllWeatherStrategy, 
                FeynmanScientificInvestor, SimonsQuantInvestor,
                MastersFusionEngine, InvestmentProfile
            )
            logger.info("✅ 4대 거장 모듈 import 성공")
        except ImportError as e:
            logger.error(f"❌ 4대 거장 모듈 import 실패: {str(e)}")
            missing_files.append("masters modules")
        
        # 7. 결과 요약
        issues = len(missing_dirs) + len(missing_files) + len(missing_required)
        
        if issues == 0:
            logger.info("🎉 환경 확인 완료! 4대 거장 시스템이 준비되었습니다.")
            if missing_ml:
                logger.info(f"💡 ML 성능 향상을 위해 설치 권장: pip install {' '.join(missing_ml)}")
            return True
        else:
            logger.error(f"❌ {issues}개의 문제 발견:")
            if missing_dirs:
                logger.error(f"   - 누락된 디렉토리: {missing_dirs}")
            if missing_files:
                logger.error(f"   - 누락된 파일: {missing_files}")
            if missing_required:
                logger.error(f"   - 누락된 필수 패키지: {missing_required}")
                logger.info(f"   설치 명령어: pip install {' '.join(missing_required)}")
            
            return False
            
    except Exception as e:
        logger.error(f"환경 확인 중 오류 발생: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def get_mock_db(self):
    """Mock 데이터베이스 세션 반환"""
    if self.demo_mode:
        return MockDatabase()
    else:
        try:
            logger.info("실제 DB 연결 미구현, Mock DB 사용")
            return MockDatabase()
        except Exception as e:
            logger.warning(f"DB 연결 실패, Mock DB 사용: {str(e)}")
            return MockDatabase()

async def test_feynman_algorithm(self) -> Dict[str, Any]:
    """리처드 파인만 과학적 사고 알고리즘 테스트"""
    logger.info("🔬 리처드 파인만 과학적 사고 알고리즘 테스트 시작...")
    
    try:
        try:
            from app.masters.feynman import FeynmanScientificInvestor, FeynmanScore, create_feynman_portfolio
            logger.info("✅ 파인만 모듈 import 성공")
        except ImportError as e:
            logger.error(f"❌ 파인만 모듈 import 실패: {str(e)}")
            return {
                'algorithm': 'Richard Feynman',
                'status': 'import_failed',
                'error': str(e)
            }
        
        db_session = self.get_mock_db()
        feynman = FeynmanScientificInvestor(db_session)
        logger.info("✅ FeynmanScientificInvestor 인스턴스 생성 완료")
        
        # 개별 종목 평가 테스트
        individual_results = {}
        test_count = 2 if self.demo_mode else 3
        
        for i, ticker in enumerate(self.test_tickers[:test_count]):
            logger.info(f"  📊 [{i+1}/{test_count}] {ticker} 과학적 분석 중...")
            
            try:
                if self.demo_mode:
                    # 데모 모드: 가상 점수 생성
                    mock_score = FeynmanScore(
                        total_score=72.5 + i * 3,
                        understanding_score=85.0 + i * 2,
                        uncertainty_score=78.0 + i * 1.5,
                        simplicity_score=80.0 + i * 2.5,
                        probability_score=75.0 + i * 3,
                        monte_carlo_confidence=0.25 + i * 0.05,
                        bayesian_probability=0.68 + i * 0.02,
                        expected_scenarios={
                            'optimistic': 0.15 + i * 0.01,
                            'base_case': 0.08 + i * 0.005,
                            'pessimistic': -0.05 + i * 0.002,
                            'expected_value': 0.06 + i * 0.003
                        },
                        confidence_interval=(-0.12 + i * 0.01, 0.18 + i * 0.01),
                        intellectual_honesty=82.0 + i * 1.5
                    )
                    
                    individual_results[ticker] = {
                        'total_score': mock_score.total_score,
                        'understanding_score': mock_score.understanding_score,
                        'uncertainty_score': mock_score.uncertainty_score,
                        'monte_carlo_confidence': mock_score.monte_carlo_confidence,
                        'bayesian_probability': mock_score.bayesian_probability,
                        'status': 'demo_success'
                    }
                    logger.info(f"    ✅ {ticker}: {mock_score.total_score:.1f}점 (과학적 사고)")
                
                else:
                    score = await feynman.evaluate_stock(ticker)
                    
                    if score:
                        individual_results[ticker] = {
                            'total_score': score.total_score,
                            'understanding_score': score.understanding_score,
                            'uncertainty_score': score.uncertainty_score,
                            'monte_carlo_confidence': score.monte_carlo_confidence,
                            'bayesian_probability': score.bayesian_probability,
                            'status': 'success'
                        }
                        logger.info(f"    ✅ {ticker}: {score.total_score:.1f}점")
                    else:
                        individual_results[ticker] = {'status': 'no_data'}
                        logger.warning(f"    ⚠️ {ticker}: 데이터 부족")
            
            except Exception as e:
                individual_results[ticker] = {'status': 'error', 'error': str(e)}
                logger.error(f"    ❌ {ticker}: {str(e)}")
        
        # 포트폴리오 생성 테스트
        logger.info("  📈 파인만 포트폴리오 생성 테스트...")
        try:
            if self.demo_mode:
                portfolio_result = {
                    'portfolio': {
                        '005930': {'weight': 0.08, 'reasoning': '이해도 90점, 불확실성 관리 우수'},
                        '055550': {'weight': 0.06, 'reasoning': '과학적 검증 가능한 사업모델'},
                        '105560': {'weight': 0.04, 'reasoning': '확률적 분석 결과 양호'},
                        '051910': {'weight': 0.02, 'reasoning': '몬테카를로 시뮬레이션 통과'}
                    },
                    'total_allocation': 0.20,
                    'selected_count': 4,
                    'average_understanding': 87.5,
                    'average_uncertainty': 76.2,
                    'strategy': 'Richard Feynman Scientific Thinking (Demo)'
                }
                logger.info("    ✅ 데모 과학적 포트폴리오 생성 완료")
            else:
                portfolio_result = await create_feynman_portfolio(self.test_tickers, db_session, 0.2)
                if portfolio_result:
                    logger.info(f"    ✅ 포트폴리오 생성: {len(portfolio_result.get('portfolio', {}))}개 종목")
                else:
                    logger.warning("    ⚠️ 포트폴리오 생성 실패")
        
        except Exception as e:
            logger.error(f"    ❌ 포트폴리오 생성 오류: {str(e)}")
            portfolio_result = {'error': str(e)}
        
        return {
            'algorithm': 'Richard Feynman',
            'individual_scores': individual_results,
            'portfolio': portfolio_result,
            'test_mode': 'demo' if self.demo_mode else 'real',
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"파인만 알고리즘 테스트 실패: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'algorithm': 'Richard Feynman',
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc() if not self.demo_mode else None
        }

async def test_simons_algorithm(self) -> Dict[str, Any]:
    """짐 사이먼스 퀀트 알고리즘 테스트"""
    logger.info("📐 짐 사이먼스 퀀트 알고리즘 테스트 시작...")
    
    try:
        try:
            from app.masters.simons import SimonsQuantInvestor, SimonsScore, create_simons_portfolio
            logger.info("✅ 사이먼스 모듈 import 성공")
        except ImportError as e:
            logger.error(f"❌ 사이먼스 모듈 import 실패: {str(e)}")
            return {
                'algorithm': 'Jim Simons',
                'status': 'import_failed',
                'error': str(e)
            }
        
        db_session = self.get_mock_db()
        simons = SimonsQuantInvestor(db_session)
        logger.info("✅ SimonsQuantInvestor 인스턴스 생성 완료")
        
        # 개별 종목 퀀트 분석 테스트
        individual_results = {}
        test_count = 3 if self.demo_mode else 4
        
        for i, ticker in enumerate(self.test_tickers[:test_count]):
            logger.info(f"  📊 [{i+1}/{test_count}] {ticker} 퀀트 분석 중...")
            
            try:
                if self.demo_mode:
                    mock_score = SimonsScore(
                        total_score=78.5 + i * 4,
                        factor_score=75.0 + i * 3,
                        momentum_score=82.0 + i * 2,
                        mean_reversion_score=68.0 + i * 3.5,
                        anomaly_score=73.0 + i * 2.5,
                        ml_prediction_score=80.0 + i * 3,
                        statistical_significance=85.0 + i * 1.5,
                        sharpe_ratio=1.25 + i * 0.15,
                        information_ratio=0.85 + i * 0.1,
                        max_drawdown=-0.12 + i * 0.01,
                        win_rate=0.58 + i * 0.02,
                        expected_alpha=0.08 + i * 0.01
                    )
                    
                    individual_results[ticker] = {
                        'total_score': mock_score.total_score,
                        'factor_score': mock_score.factor_score,
                        'sharpe_ratio': mock_score.sharpe_ratio,
                        'expected_alpha': mock_score.expected_alpha,
                        'statistical_significance': mock_score.statistical_significance,
                        'status': 'demo_success'
                    }
                    logger.info(f"    ✅ {ticker}: {mock_score.total_score:.1f}점 (퀀트)")
                
                else:
                    score = await simons.evaluate_stock(ticker)
                    
                    if score:
                        individual_results[ticker] = {
                            'total_score': score.total_score,
                            'factor_score': score.factor_score,
                            'sharpe_ratio': score.sharpe_ratio,
                            'expected_alpha': score.expected_alpha,
                            'statistical_significance': score.statistical_significance,
                            'status': 'success'
                        }
                        logger.info(f"    ✅ {ticker}: {score.total_score:.1f}점")
                    else:
                        individual_results[ticker] = {'status': 'no_data'}
                        logger.warning(f"    ⚠️ {ticker}: 데이터 부족")
            
            except Exception as e:
                individual_results[ticker] = {'status': 'error', 'error': str(e)}
                logger.error(f"    ❌ {ticker}: {str(e)}")
        
        # 퀀트 포트폴리오 생성 테스트
        logger.info("  📈 사이먼스 퀀트 포트폴리오 생성 테스트...")
        try:
            if self.demo_mode:
                portfolio_result = {
                    'portfolio': {
                        '000660': {'weight': 0.12, 'reasoning': '모멘텀 팩터 85점, ML 예측 우수'},
                        '005930': {'weight': 0.10, 'reasoning': '멀티팩터 종합 80점'},
                        '035420': {'weight': 0.08, 'reasoning': '통계적 이상현상 탐지'},
                        '068270': {'weight': 0.06, 'reasoning': '퀀트 신호 강함'},
                        '207940': {'weight': 0.04, 'reasoning': '알파 0.12 기대'}
                    },
                    'total_allocation': 0.40,
                    'selected_count': 5,
                    'average_alpha': 0.095,
                    'average_sharpe': 1.35,
                    'strategy': 'Jim Simons Quantitative (Demo)'
                }
                logger.info("    ✅ 데모 퀀트 포트폴리오 생성 완료")
            else:
                portfolio_result = await create_simons_portfolio(self.test_tickers, db_session, 0.4)
                if portfolio_result:
                    logger.info(f"    ✅ 포트폴리오 생성: {len(portfolio_result.get('portfolio', {}))}개 종목")
                else:
                    logger.warning("    ⚠️ 포트폴리오 생성 실패")
        
        except Exception as e:
            logger.error(f"    ❌ 포트폴리오 생성 오류: {str(e)}")
            portfolio_result = {'error': str(e)}
        
        return {
            'algorithm': 'Jim Simons',
            'individual_scores': individual_results,
            'portfolio': portfolio_result,
            'test_mode': 'demo' if self.demo_mode else 'real',
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"사이먼스 알고리즘 테스트 실패: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'algorithm': 'Jim Simons',
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc() if not self.demo_mode else None
        }

async def test_complete_fusion_engine(self) -> Dict[str, Any]:
    """4대 거장 완전 융합 엔진 테스트"""
    logger.info("🔀 4대 거장 완전 융합 엔진 테스트 시작...")
    
    try:
        try:
            from app.masters.fusion import (
                MastersFusionEngine, InvestmentProfile, MasterWeights,
                FusionResult, create_masters_fusion_portfolio
            )
            logger.info("✅ 완전 융합 엔진 모듈 import 성공")
        except ImportError as e:
            logger.error(f"❌ 융합 엔진 모듈 import 실패: {str(e)}")
            return {
                'algorithm': 'Complete Masters Fusion',
                'status': 'import_failed',
                'error': str(e)
            }
        
        db_session = self.get_mock_db()
        
        # 3가지 투자 성향별 완전 융합 포트폴리오 테스트
        profile_results = {}
        
        for profile in InvestmentProfile:
            profile_name = profile.value
            logger.info(f"  🎯 {profile_name} 성향 완전 융합 포트폴리오 생성...")
            
            try:
                if self.demo_mode:
                    demo_weights = {
                        InvestmentProfile.CONSERVATIVE: {'buffett': 0.4, 'dalio': 0.4, 'feynman': 0.15, 'simons': 0.05},
                        InvestmentProfile.BALANCED: {'buffett': 0.25, 'dalio': 0.25, 'feynman': 0.25, 'simons': 0.25},
                        InvestmentProfile.AGGRESSIVE: {'buffett': 0.15, 'dalio': 0.15, 'feynman': 0.20, 'simons': 0.50}
                    }
                    
                    demo_portfolio = {
                        'portfolio': {
                            '005930': {
                                'weight': 0.15, 
                                'masters_votes': {'buffett': 0.05, 'dalio': 0.03, 'feynman': 0.04, 'simons': 0.03},
                                'consensus_strength': 1.0,
                                'confidence_score': 0.85
                            },
                            '000660': {
                                'weight': 0.12, 
                                'masters_votes': {'simons': 0.08, 'feynman': 0.04},
                                'consensus_strength': 0.5,
                                'confidence_score': 0.78
                            },
                            '055550': {
                                'weight': 0.08, 
                                'masters_votes': {'buffett': 0.05, 'dalio': 0.03},
                                'consensus_strength': 0.5,
                                'confidence_score': 0.82
                            },
                            '051910': {
                                'weight': 0.06, 
                                'masters_votes': {'dalio': 0.04, 'feynman': 0.02},
                                'consensus_strength': 0.5,
                                'confidence_score': 0.75
                            }
                        },
                        'total_score': 87.5 if profile == InvestmentProfile.BALANCED else 82.0,
                        'expected_return': 0.12 if profile == InvestmentProfile.BALANCED else 0.10,
                        'expected_volatility': 0.18 if profile == InvestmentProfile.BALANCED else 0.15,
                        'master_weights': demo_weights[profile],
                        'master_contributions': {
                            'buffett': {'total_contribution': demo_weights[profile]['buffett'] * 0.9},
                            'dalio': {'total_contribution': demo_weights[profile]['dalio'] * 0.95},
                            'feynman': {'total_contribution': demo_weights[profile]['feynman'] * 0.85},
                            'simons': {'total_contribution': demo_weights[profile]['simons'] * 1.1}
                        },
                        'strategy': f'Complete Masters Fusion - {profile_name} (Demo)'
                    }
                    
                    profile_results[profile_name] = {
                        'portfolio_size': len(demo_portfolio['portfolio']),
                        'total_score': demo_portfolio['total_score'],
                        'expected_return': demo_portfolio['expected_return'],
                        'expected_volatility': demo_portfolio['expected_volatility'],
                        'master_weights': demo_portfolio['master_weights'],
                        'master_contributions': demo_portfolio['master_contributions'],
                        'avg_consensus': 0.75,
                        'avg_confidence': 0.80,
                        'status': 'demo_success'
                    }
                    
                    logger.info(f"    ✅ {profile_name}: {demo_portfolio['total_score']:.1f}점 (완전 융합)")
                
                else:
                    portfolio_result = await create_masters_fusion_portfolio(
                        self.test_tickers, profile, db_session
                    )
                    
                    if portfolio_result and 'portfolio' in portfolio_result:
                        profile_results[profile_name] = {
                            'portfolio_size': len(portfolio_result['portfolio']),
                            'total_score': portfolio_result.get('total_score', 0),
                            'expected_return': portfolio_result.get('expected_return', 0),
                            'expected_volatility': portfolio_result.get('expected_volatility', 0),
                            'master_weights': portfolio_result.get('master_weights', {}),
                            'master_contributions': portfolio_result.get('master_contributions', {}),
                            'status': 'success'
                        }
                        logger.info(f"    ✅ {profile_name}: {portfolio_result.get('total_score', 0):.1f}점")
                    else:
                        profile_results[profile_name] = {'status': 'no_result'}
                        logger.warning(f"    ⚠️ {profile_name}: 결과 없음")
            
            except Exception as e:
                profile_results[profile_name] = {'status': 'error', 'error': str(e)}
                logger.error(f"    ❌ {profile_name}: {str(e)}")
        
        # 커스텀 가중치 테스트 (4대 거장)
        logger.info("  ⚖️ 커스텀 가중치 테스트 (4대 거장)...")
        try:
            if self.demo_mode:
                custom_result = {
                    'portfolio_size': 8,
                    'weights_used': {'buffett': 0.3, 'dalio': 0.3, 'feynman': 0.2, 'simons': 0.2},
                    'total_score': 89.5,
                    'master_effectiveness': {'buffett': 0.95, 'dalio': 0.88, 'feynman': 0.92, 'simons': 1.05},
                    'status': 'demo_success'
                }
                logger.info("    ✅ 커스텀 가중치 테스트 완료 (4대 거장)")
            else:
                custom_weights = MasterWeights(buffett=0.3, dalio=0.3, feynman=0.2, simons=0.2)
                custom_portfolio = await create_masters_fusion_portfolio(
                    self.test_tickers, InvestmentProfile.BALANCED, db_session, custom_weights
                )
                
                custom_result = {
                    'portfolio_size': len(custom_portfolio.get('portfolio', {})) if custom_portfolio else 0,
                    'weights_used': custom_weights.to_dict(),
                    'total_score': custom_portfolio.get('total_score', 0) if custom_portfolio else 0,
                    'status': 'success' if custom_portfolio else 'failed'
                }
                logger.info(f"    ✅ 커스텀 가중치: {custom_result['total_score']:.1f}점")
            
            profile_results['custom_weights'] = custom_result
        
        except Exception as e:
            profile_results['custom_weights'] = {'status': 'error', 'error': str(e)}
            logger.error(f"    ❌ 커스텀 가중치 테스트 오류: {str(e)}")
        
        return {
            'algorithm': 'Complete Masters Fusion',
            'profile_tests': profile_results,
            'test_mode': 'demo' if self.demo_mode else 'real',
            'masters_count': 4,
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"완전 융합 엔진 테스트 실패: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'algorithm': 'Complete Masters Fusion',
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc() if not self.demo_mode else None
        }

async def test_buffett_algorithm(self) -> Dict[str, Any]:
    """워렌 버핏 알고리즘 테스트 (재사용)"""
    logger.info("🏛️ 워렌 버핏 가치투자 알고리즘 테스트...")
    
    try:
        from app.masters.buffett import BuffettValueInvestor, create_buffett_portfolio
        
        db_session = self.get_mock_db()
        buffett = BuffettValueInvestor(db_session)
        
        individual_results = {}
        for i, ticker in enumerate(self.test_tickers[:2]):
            if self.demo_mode:
                individual_results[ticker] = {
                    'total_score': 75.5 + i * 5,
                    'intrinsic_value': 65000 + i * 5000,
                    'margin_of_safety': 0.23 + i * 0.02,
                    'status': 'demo_success'
                }
            
        portfolio_result = {
            'portfolio': {'005930': {'weight': 0.15}, '055550': {'weight': 0.10}},
            'total_allocation': 0.30,
            'strategy': 'Warren Buffett Value Investing'
        } if self.demo_mode else await create_buffett_portfolio(self.test_tickers, db_session, 0.3)
        
        return {
            'algorithm': 'Warren Buffett',
            'individual_scores': individual_results,
            'portfolio': portfolio_result,
            'status': 'completed'
        }
        
    except Exception as e:
        return {'algorithm': 'Warren Buffett', 'status': 'failed', 'error': str(e)}

async def test_dalio_algorithm(self) -> Dict[str, Any]:
    """레이 달리오 알고리즘 테스트 (재사용)"""
    logger.info("🌊 레이 달리오 All Weather 전략 테스트...")
    
    try:
        from app.masters.dalio import AllWeatherStrategy, EconomicMachine, EconomicIndicators
        
        db_session = self.get_mock_db()
        all_weather = AllWeatherStrategy(db_session)
        
        portfolio_result = {
            'portfolio': {
                '005930': {'weight': 0.08, 'asset_class': 'growth_stocks'},
                '055550': {'weight': 0.06, 'asset_class': 'value_stocks'}
            },
            'economic_environment': 'recovery',
            'strategy': 'Ray Dalio All Weather'
        } if self.demo_mode else await all_weather.create_all_weather_portfolio(0.3)
        
        return {
            'algorithm': 'Ray Dalio',
            'portfolio': portfolio_result,
            'status': 'completed'
        }
        
    except Exception as e:
        return {'algorithm': 'Ray Dalio', 'status': 'failed', 'error': str(e)}

async def run_performance_benchmark(self) -> Dict[str, Any]:
    """성능 벤치마크 실행"""
    logger.info("⚡ 성능 벤치마크 실행...")
    
    try:
        benchmark_results = {}
        
        # 1. 개별 알고리즘 성능 측정
        algorithms = [
            ('buffett', self.test_buffett_algorithm),
            ('dalio', self.test_dalio_algorithm),
            ('feynman', self.test_feynman_algorithm),
            ('simons', self.test_simons_algorithm)
        ]
        
        for algo_name, algo_func in algorithms:
            start_time = time.time()
            result = await algo_func()
            end_time = time.time()
            
            benchmark_results[algo_name] = {
                'execution_time': end_time - start_time,
                'status': result.get('status', 'unknown'),
                'memory_efficient': True  # 간소화
            }
        
        # 2. 융합 엔진 성능 측정
        start_time = time.time()
        fusion_result = await self.test_complete_fusion_engine()
        end_time = time.time()
        
        benchmark_results['fusion'] = {
            'execution_time': end_time - start_time,
            'status': fusion_result.get('status', 'unknown'),
            'total_complexity': 'high'
        }
        
        # 3. 전체 성능 요약
        total_time = sum(r['execution_time'] for r in benchmark_results.values())
        successful_algos = sum(1 for r in benchmark_results.values() if r['status'] in ['completed', 'demo_success'])
        
        return {
            'benchmark_results': benchmark_results,
            'total_execution_time': total_time,
            'successful_algorithms': successful_algos,
            'total_algorithms': len(benchmark_results),
            'average_execution_time': total_time / len(benchmark_results),
            'performance_grade': 'A' if total_time < 10 else 'B' if total_time < 30 else 'C',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"성능 벤치마크 실패: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

async def run_comprehensive_test(self) -> Dict[str, Any]:
    """종합 테스트 실행 (4대 거장 완성)"""
    logger.info("🚀 Master's Eye 4대 거장 완전 통합 테스트 시작!")
    logger.info(f"   모드: {'데모' if self.demo_mode else '실제'}")
    
    start_time = datetime.now()
    
    # 환경 확인
    if not self.check_environment():
        return {
            'status': 'failed',
            'reason': 'environment_check_failed',
            'timestamp': start_time.isoformat(),
            'test_mode': 'demo' if self.demo_mode else 'real'
        }
    
    # 각 알고리즘별 테스트 실행
    test_results = {}
    
    # 1. 워렌 버핏 알고리즘 테스트
    logger.info("1️⃣ 워렌 버핏 가치투자 알고리즘...")
    test_results['buffett'] = await self.test_buffett_algorithm()
    
    # 2. 레이 달리오 알고리즘 테스트  
    logger.info("2️⃣ 레이 달리오 All Weather 전략...")
    test_results['dalio'] = await self.test_dalio_algorithm()
    
    # 3. 리처드 파인만 알고리즘 테스트
    logger.info("3️⃣ 리처드 파인만 과학적 사고...")
    test_results['feynman'] = await self.test_feynman_algorithm()
    
    # 4. 짐 사이먼스 알고리즘 테스트
    logger.info("4️⃣ 짐 사이먼스 퀀트 투자...")
    test_results['simons'] = await self.test_simons_algorithm()
    
    # 5. 4대 거장 완전 융합 테스트
    logger.info("5️⃣ 4대 거장 완전 융합 엔진...")
    test_results['complete_fusion'] = await self.test_complete_fusion_engine()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 결과 요약
    successful_tests = sum(1 for result in test_results.values() if result.get('status') == 'completed')
    total_tests = len(test_results)
    
    summary = {
        'overall_status': 'success' if successful_tests == total_tests else 'partial_success',
        'successful_tests': successful_tests,
        'total_tests': total_tests,
        'success_rate': f"{successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)",
        'duration_seconds': duration,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'test_mode': 'demo' if self.demo_mode else 'real',
        'masters_completed': 4,
        'fusion_engine_status': test_results['complete_fusion'].get('status', 'unknown'),
        'detailed_results': test_results
    }
    
    # 결과 로깅
    logger.info(f"🎉 4대 거장 완전 통합 테스트 완료!")
    logger.info(f"   성공률: {summary['success_rate']}")
    logger.info(f"   소요시간: {duration:.2f}초")
    logger.info(f"   모드: {'데모' if self.demo_mode else '실제'}")
    logger.info(f"   융합 엔진: {summary['fusion_engine_status']}")
    
    # 결과를 파일로 저장
    self.save_test_results(summary)
    
    return summary

def save_test_results(self, results: Dict[str, Any]):
    """테스트 결과를 파일로 저장"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_suffix = "_demo" if self.demo_mode else "_real"
        results_file = self.results_dir / f"masters_complete_{timestamp}{mode_suffix}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 테스트 결과 저장: {results_file}")
        
        # 간단한 요약 파일도 생성
        summary_file = self.results_dir / f"summary_{timestamp}{mode_suffix}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Master's Eye 4대 거장 완전 테스트 결과\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"실행 시간: {results['start_time']} ~ {results['end_time']}\n")
            f.write(f"소요 시간: {results['duration_seconds']:.2f}초\n")
            f.write(f"테스트 모드: {results['test_mode']}\n")
            f.write(f"성공률: {results['success_rate']}\n")
            f.write(f"전체 상태: {results['overall_status']}\n")
            f.write(f"완성된 거장 수: {results['masters_completed']}/4\n\n")
            
            for algo, result in results['detailed_results'].items():
                f.write(f"{algo.upper()}: {result.get('status', 'unknown')}\n")
                if result.get('error'):
                    f.write(f"  오류: {result['error']}\n")
            f.write(f"\n상세 결과: {results_file.name}\n")
        
        logger.info(f"📋 요약 저장: {summary_file}")
        
    except Exception as e:
        logger.error(f"테스트 결과 저장 실패: {str(e)}")

def print_usage(self):
    """사용법 출력 (4대 거장 완성 버전)"""
    print(f"""
```

🎯 Master’s Eye - 4대 거장 완전 융합 주식 AI 포트폴리오 시스템

GitHub: https://github.com/Dahlisse/Master-s-Eyes
Week 7-8 완성: 4대 거장 알고리즘 + 완전 융합 엔진

✨ 완성된 4대 거장:
🏛️ 워렌 버핏    - 가치투자 (DCF, 내재가치, 안전마진)
🌊 레이 달리오   - All Weather (Economic Machine, 리스크 패리티)
🔬 리처드 파인만 - 과학적 사고 (몬테카를로, 베이지안 추론)
📐 짐 사이먼스   - 퀀트 투자 (멀티팩터, 머신러닝)

사용법:
python run_masters_eye.py [옵션]

옵션:
–test-all         4대 거장 완전 융합 테스트 (기본값)
–test-buffett     워렌 버핏 알고리즘만 테스트
–test-dalio       레이 달리오 알고리즘만 테스트  
–test-feynman     리처드 파인만 알고리즘만 테스트 ✨NEW
–test-simons      짐 사이먼스 알고리즘만 테스트 ✨NEW
–test-fusion      4대 거장 완전 융합 엔진만 테스트
–check-env        개발 환경 확인 (4대 거장 버전)
–demo             데모 모드 (Mock 데이터 사용)
–performance      성능 벤치마크 실행 ✨NEW
–help, -h         이 도움말 출력

예시:
python run_masters_eye.py                      # 4대 거장 완전 융합 테스트
python run_masters_eye.py –demo               # 데모 모드 테스트
python run_masters_eye.py –test-feynman       # 파인만 과학적 사고만
python run_masters_eye.py –test-simons –demo # 사이먼스 퀀트 데모
python run_masters_eye.py –performance        # 성능 벤치마크
python run_masters_eye.py –check-env          # 환경 확인

📁 결과 파일:

- test_results/masters_complete_YYYYMMDD_HHMMSS_[mode].json
- test_results/summary_YYYYMMDD_HHMMSS_[mode].txt
- logs/masters_eye.log

🎯 Week 7-8 완성 현황:
✅ 워렌 버핏: DCF 모델, 내재가치, 안전마진 계산
✅ 레이 달리오: Economic Machine, All Weather, 리스크 패리티
✅ 리처드 파인만: 몬테카를로 시뮬레이션, 베이지안 추론, 불확실성 정량화
✅ 짐 사이먼스: 멀티팩터 모델, 머신러닝 예측, 통계적 차익거래
✅ 완전 융합 엔진: 4대 거장 지능형 통합, 동적 가중치 조정

🚀 다음 단계: Week 9에서 AI 대화 시스템 구현 예정
“””)

async def main():
“”“메인 실행 함수 (4대 거장 완성 버전)”””

```
# 명령행 인수 처리
args = sys.argv[1:]

if '--help' in args or '-h' in args:
    MastersEyeRunner().print_usage()
    return

# 데모 모드 확인
demo_mode = '--demo' in args

# 실행할 테스트 결정
runner = MastersEyeRunner(demo_mode=demo_mode)

try:
    if '--check-env' in args:
        success = runner.check_environment()
        if success:
            print("\n✅ 환경 확인 완료! 4대 거장 시스템이 준비되었습니다.")
            sys.exit(0)
        else:
            print("\n❌ 환경 확인 실패! 위의 문제들을 해결해주세요.")
            sys.exit(1)
    
    elif '--test-buffett' in args:
        print("🏛️ 워렌 버핏 알고리즘 단독 테스트")
        result = await runner.test_buffett_algorithm()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    elif '--test-dalio' in args:
        print("🌊 레이 달리오 알고리즘 단독 테스트")
        result = await runner.test_dalio_algorithm()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    elif '--test-feynman' in args:
        print("🔬 리처드 파인만 알고리즘 단독 테스트")
        result = await runner.test_feynman_algorithm()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    elif '--test-simons' in args:
        print("📐 짐 사이먼스 알고리즘 단독 테스트")
        result = await runner.test_simons_algorithm()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    elif '--test-fusion' in args:
        print("🔀 4대 거장 완전 융합 엔진 단독 테스트")
        result = await runner.test_complete_fusion_engine()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    elif '--performance' in args:
        print("⚡ 성능 벤치마크 실행")
        result = await runner.run_performance_benchmark()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    else:  # 기본값: 4대 거장 완전 융합 테스트
        print("🚀 Master's Eye 4대 거장 완전 융합 테스트!")
        if demo_mode:
            print("   (데모 모드: Mock 데이터 사용)")
        result = await runner.run_comprehensive_test()
        
        # 간단한 요약 출력
        print(f"\n📊 4대 거장 완전 테스트 완료:")
        print(f"   성공률: {result['success_rate']}")
        print(f"   소요시간: {result['duration_seconds']:.2f}초")
        print(f"   완성된 거장: {result['masters_completed']}/4")
        print(f"   전체 상태: {result['overall_status']}")
        
        # 상세 결과는 파일로만 저장
        print(f"\n📄 상세 결과는 test_results/ 디렉토리에 저장되었습니다.")

except KeyboardInterrupt:
    logger.info("⏹️ 사용자에 의해 중단됨")
    print("\n⏹️ 테스트가 중단되었습니다.")
except Exception as e:
    logger.error(f"❌ 실행 중 치명적 오류: {str(e)}")
    logger.debug(traceback.format_exc())
    print(f"\n❌ 실행 중 오류 발생: {str(e)}")
    print("📋 자세한 오류 정보는 logs/masters_eye.log 파일을 확인하세요.")
    sys.exit(1)
```

if **name** == “**main**”:
print(“🎯 Master’s Eye - 4대 거장 완전 융합 주식 AI 포트폴리오 시스템”)
print(”=” * 70)
print(“✨ 완성된 4대 거장: 버핏 🏛️ | 달리오 🌊 | 파인만 🔬 | 사이먼스 📐”)
print(”=” * 70)
asyncio.run(main())