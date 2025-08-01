# backend/app/data/processors/data_quality_monitor.py

“””
데이터 품질 모니터링 시스템

- 실시간 데이터 품질 검증
- 이상 데이터 탐지 및 알림
- 데이터 완성도/정확성/일관성 모니터링
- 자동 품질 리포트 생성
  “””
  import asyncio
  import asyncpg
  import pandas as pd
  import numpy as np
  from datetime import datetime, timedelta
  from typing import Dict, List, Optional, Any, Tuple, Union
  import logging
  from dataclasses import dataclass, asdict
  import json
  import statistics
  from collections import defaultdict, Counter
  import warnings
  from pathlib import Path

from app.core.config import get_settings
from app.core.database import get_database
from app.core.redis import get_redis_client
from app.data.processors.data_validator import DataValidator

settings = get_settings()
logger = logging.getLogger(**name**)

@dataclass
class QualityMetric:
“”“품질 지표”””
metric_name: str
value: float
threshold: float
status: str  # ‘pass’, ‘warning’, ‘fail’
description: str
severity: str  # ‘low’, ‘medium’, ‘high’, ‘critical’
timestamp: datetime

@dataclass
class DataQualityReport:
“”“데이터 품질 리포트”””
table_name: str
check_timestamp: datetime
total_records: int
quality_score: float  # 0-100
completeness_score: float
accuracy_score: float
consistency_score: float
timeliness_score: float
validity_score: float
metrics: List[QualityMetric]
issues_found: List[str]
recommendations: List[str]
data_freshness_hours: float
duplicate_records: int
null_percentage: float
outliers_detected: int

@dataclass
class QualityAlert:
“”“품질 알림”””
alert_id: str
table_name: str
alert_type: str  # ‘completeness’, ‘accuracy’, ‘consistency’, ‘timeliness’
severity: str
message: str
current_value: float
threshold_value: float
first_detected: datetime
last_updated: datetime
status: str  # ‘active’, ‘resolved’, ‘acknowledged’

class DataQualityMonitor:
“”“데이터 품질 모니터링 시스템”””

```
def __init__(self):
    self.db_pool = None
    self.redis_client = get_redis_client()
    self.validator = DataValidator()
    
    # 품질 임계값 설정
    self.quality_thresholds = {
        'completeness': {
            'excellent': 98.0,
            'good': 95.0,
            'acceptable': 90.0,
            'poor': 85.0
        },
        'accuracy': {
            'excellent': 99.0,
            'good': 97.0,
            'acceptable': 94.0,
            'poor': 90.0
        },
        'consistency': {
            'excellent': 98.0,
            'good': 95.0,
            'acceptable': 92.0,
            'poor': 88.0
        },
        'timeliness': {
            'excellent': 95.0,
            'good': 90.0,
            'acceptable': 85.0,
            'poor': 80.0
        },
        'validity': {
            'excellent': 99.5,
            'good': 98.0,
            'acceptable': 95.0,
            'poor': 92.0
        }
    }
    
    # 테이블별 품질 규칙
    self.table_quality_rules = {
        'market_data': {
            'required_columns': ['ticker', 'price', 'volume', 'time'],
            'price_range': (0.01, 1000000),  # 원 단위
            'volume_range': (0, 1000000000),
            'max_price_change_pct': 30.0,  # 하루 최대 변동률
            'freshness_hours': 24,
            'unique_key_columns': ['ticker', 'time'],
            'business_rules': [
                'price > 0',
                'volume >= 0',
                'time <= NOW()'
            ]
        },
        'economic_indicators': {
            'required_columns': ['series_id', 'value', 'date'],
            'value_range': (-100, 1000),  # 대부분 지표는 이 범위
            'freshness_hours': 168,  # 7일
            'unique_key_columns': ['series_id', 'date'],
            'business_rules': [
                'date <= NOW()',
                'series_id IS NOT NULL'
            ]
        },
        'news_articles': {
            'required_columns': ['title', 'content', 'published_time', 'source'],
            'min_content_length': 50,
            'max_content_length': 50000,
            'freshness_hours': 48,
            'unique_key_columns': ['url'],
            'business_rules': [
                'LENGTH(title) > 5',
                'LENGTH(content) > 20',
                'published_time <= NOW()',
                'source IS NOT NULL'
            ]
        },
        'portfolios': {
            'required_columns': ['user_id', 'allocation', 'created_at'],
            'freshness_hours': 720,  # 30일
            'unique_key_columns': ['id'],
            'business_rules': [
                'user_id > 0',
                'created_at <= NOW()'
            ]
        },
        'stock_prices': {
            'required_columns': ['ticker', 'open_price', 'close_price', 'high_price', 'low_price', 'volume', 'timestamp'],
            'price_range': (0.01, 1000000),
            'volume_range': (0, 1000000000),
            'max_price_change_pct': 30.0,
            'freshness_hours': 24,
            'unique_key_columns': ['ticker', 'timestamp'],
            'business_rules': [
                'open_price > 0',
                'close_price > 0', 
                'high_price >= low_price',
                'high_price >= open_price',
                'high_price >= close_price',
                'low_price <= open_price',
                'low_price <= close_price',
                'volume >= 0'
            ]
        }
    }
    
    # 이상 탐지 설정
    self.anomaly_detection_config = {
        'statistical_methods': ['z_score', 'iqr', 'isolation_forest'],
        'z_score_threshold': 3.0,
        'iqr_multiplier': 1.5,
        'isolation_forest_contamination': 0.1,
        'seasonal_decomposition': True,
        'trend_analysis': True
    }
    
    # 알림 설정
    self.alert_config = {
        'critical_threshold': 70,  # 품질 점수가 70 이하면 critical
        'warning_threshold': 85,   # 85 이하면 warning
        'alert_cooldown_minutes': 30,  # 같은 알림 재발송 방지
        'max_alerts_per_hour': 10,
        'notification_channels': ['email', 'slack', 'database']
    }

async def __aenter__(self):
    """비동기 컨텍스트 매니저 진입"""
    self.db_pool = await get_database()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    """비동기 컨텍스트 매니저 종료"""
    if self.db_pool:
        await self.db_pool.close()

async def check_table_quality(self, table_name: str, detailed_analysis: bool = True) -> DataQualityReport:
    """테이블 품질 종합 검사"""
    try:
        logger.info(f"Starting quality check for table: {table_name}")
        check_start = datetime.now()
        
        # 기본 통계 수집
        basic_stats = await self._collect_basic_statistics(table_name)
        
        if basic_stats['total_records'] == 0:
            return self._empty_quality_report(table_name, "Table is empty")
        
        # 품질 지표별 검사
        metrics = []
        issues = []
        recommendations = []
        
        # 1. 완성도 (Completeness) 검사
        completeness_score, completeness_metrics, completeness_issues = await self._check_completeness(table_name, basic_stats)
        metrics.extend(completeness_metrics)
        issues.extend(completeness_issues)
        
        # 2. 정확성 (Accuracy) 검사
        accuracy_score, accuracy_metrics, accuracy_issues = await self._check_accuracy(table_name, basic_stats)
        metrics.extend(accuracy_metrics)
        issues.extend(accuracy_issues)
        
        # 3. 일관성 (Consistency) 검사
        consistency_score, consistency_metrics, consistency_issues = await self._check_consistency(table_name, basic_stats)
        metrics.extend(consistency_metrics)
        issues.extend(consistency_issues)
        
        # 4. 시의성 (Timeliness) 검사
        timeliness_score, timeliness_metrics, timeliness_issues, data_freshness = await self._check_timeliness(table_name)
        metrics.extend(timeliness_metrics)
        issues.extend(timeliness_issues)
        
        # 5. 유효성 (Validity) 검사
        validity_score, validity_metrics, validity_issues = await self._check_validity(table_name, basic_stats)
        metrics.extend(validity_metrics)
        issues.extend(validity_issues)
        
        # 상세 분석 (선택적)
        outliers_detected = 0
        if detailed_analysis:
            outliers_detected = await self._detect_outliers(table_name)
            
            # 이상 패턴 탐지
            pattern_issues = await self._detect_anomalous_patterns(table_name)
            issues.extend(pattern_issues)
        
        # 전체 품질 점수 계산 (가중 평균)
        weights = {'completeness': 0.25, 'accuracy': 0.25, 'consistency': 0.20, 'timeliness': 0.15, 'validity': 0.15}
        quality_score = (
            completeness_score * weights['completeness'] +
            accuracy_score * weights['accuracy'] +
            consistency_score * weights['consistency'] +
            timeliness_score * weights['timeliness'] +
            validity_score * weights['validity']
        )
        
        # 권장사항 생성
        recommendations = await self._generate_recommendations(table_name, metrics, issues)
        
        # 품질 리포트 생성
        report = DataQualityReport(
            table_name=table_name,
            check_timestamp=check_start,
            total_records=basic_stats['total_records'],
            quality_score=round(quality_score, 2),
            completeness_score=round(completeness_score, 2),
            accuracy_score=round(accuracy_score, 2),
            consistency_score=round(consistency_score, 2),
            timeliness_score=round(timeliness_score, 2),
            validity_score=round(validity_score, 2),
            metrics=metrics,
            issues_found=issues,
            recommendations=recommendations,
            data_freshness_hours=data_freshness,
            duplicate_records=basic_stats.get('duplicates', 0),
            null_percentage=basic_stats.get('null_percentage', 0.0),
            outliers_detected=outliers_detected
        )
        
        # 리포트 저장
        await self._save_quality_report(report)
        
        # 알림 확인 및 발송
        await self._check_and_send_alerts(report)
        
        duration = (datetime.now() - check_start).total_seconds()
        logger.info(f"Quality check completed for {table_name}: {quality_score:.1f}/100 ({duration:.1f}s)")
        
        return report
        
    except Exception as e:
        logger.error(f"Error checking quality for table {table_name}: {e}")
        return self._empty_quality_report(table_name, f"Error: {str(e)}")

async def monitor_all_tables(self, tables: Optional[List[str]] = None) -> Dict[str, DataQualityReport]:
    """모든 테이블 품질 모니터링"""
    try:
        tables_to_check = tables or list(self.table_quality_rules.keys())
        logger.info(f"Monitoring quality for {len(tables_to_check)} tables")
        
        reports = {}
        
        # 테이블별 병렬 검사 (동시성 제한)
        semaphore = asyncio.Semaphore(3)  # 동시 3개 테이블
        
        async def check_single_table(table_name: str):
            async with semaphore:
                report = await self.check_table_quality(table_name, detailed_analysis=True)
                reports[table_name] = report
                await asyncio.sleep(1)  # DB 부하 방지
        
        # 모든 테이블 동시 검사
        tasks = [check_single_table(table) for table in tables_to_check]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 전체 요약 생성
        await self._generate_system_quality_summary(reports)
        
        logger.info(f"Quality monitoring completed for {len(reports)} tables")
        return reports
        
    except Exception as e:
        logger.error(f"Error in monitor_all_tables: {e}")
        return {}

async def detect_real_time_anomalies(self, table_name: str, window_minutes: int = 30) -> List[Dict[str, Any]]:
    """실시간 이상 탐지"""
    try:
        logger.info(f"Detecting real-time anomalies for {table_name}")
        
        anomalies = []
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        # 테이블별 실시간 검사
        if table_name == 'market_data':
            anomalies.extend(await self._detect_market_data_anomalies(cutoff_time))
        elif table_name == 'news_articles':
            anomalies.extend(await self._detect_news_anomalies(cutoff_time))
        elif table_name == 'economic_indicators':
            anomalies.extend(await self._detect_economic_anomalies(cutoff_time))
        
        # 일반적인 이상 패턴 검사
        general_anomalies = await self._detect_general_anomalies(table_name, cutoff_time)
        anomalies.extend(general_anomalies)
        
        # 심각한 이상의 경우 즉시 알림
        critical_anomalies = [a for a in anomalies if a.get('severity') == 'critical']
        if critical_anomalies:
            await self._send_immediate_alerts(table_name, critical_anomalies)
        
        logger.info(f"Found {len(anomalies)} anomalies in {table_name} (last {window_minutes} minutes)")
        return anomalies
        
    except Exception as e:
        logger.error(f"Error detecting real-time anomalies for {table_name}: {e}")
        return []

async def generate_quality_dashboard_data(self) -> Dict[str, Any]:
    """품질 대시보드 데이터 생성"""
    try:
        logger.info("Generating quality dashboard data")
        
        # 최근 24시간 품질 리포트 조회
        recent_reports = await self._get_recent_quality_reports()
        
        # 전체 시스템 품질 점수
        if recent_reports:
            system_quality_score = np.mean([r['quality_score'] for r in recent_reports])
            
            # 테이블별 품질 트렌드
            table_trends = {}
            for report in recent_reports:
                table_name = report['table_name']
                if table_name not in table_trends:
                    table_trends[table_name] = []
                table_trends[table_name].append({
                    'timestamp': report['check_timestamp'],
                    'quality_score': report['quality_score']
                })
        else:
            system_quality_score = 0
            table_trends = {}
        
        # 활성 알림 조회
        active_alerts = await self._get_active_alerts()
        
        # 품질 지표 분포
        quality_distribution = await self._calculate_quality_distribution(recent_reports)
        
        # 데이터 신선도 현황
        freshness_status = await self._get_data_freshness_status()
        
        # 이슈 카테고리별 통계
        issue_categories = await self._categorize_quality_issues(recent_reports)
        
        dashboard_data = {
            'system_overview': {
                'overall_quality_score': round(system_quality_score, 1),
                'total_tables_monitored': len(self.table_quality_rules),
                'active_alerts_count': len(active_alerts),
                'last_updated': datetime.now().isoformat()
            },
            'quality_trends': table_trends,
            'quality_distribution': quality_distribution,
            'active_alerts': active_alerts[:10],  # 최근 10개
            'freshness_status': freshness_status,
            'issue_categories': issue_categories,
            'table_summaries': await self._get_table_quality_summaries(),
            'recommendations': await self._get_top_recommendations()
        }
        
        # 대시보드 데이터 캐시
        await self.redis_client.setex(
            'quality_dashboard_data',
            300,  # 5분 캐시
            json.dumps(dashboard_data, default=str)
        )
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error generating dashboard data: {e}")
        return {'error': str(e)}

# 내부 품질 검사 메서드들
async def _collect_basic_statistics(self, table_name: str) -> Dict[str, Any]:
    """기본 통계 수집"""
    try:
        async with self.db_pool.acquire() as conn:
            # 총 레코드 수
            total_records = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            
            # 컬럼 정보
            columns_info = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = $1
                ORDER BY ordinal_position
            """, table_name)
            
            columns = [col['column_name'] for col in columns_info]
            nullable_columns = [col['column_name'] for col in columns_info if col['is_nullable'] == 'YES']
            
            # NULL 값 통계
            null_counts = {}
            if total_records > 0:
                for col in nullable_columns:
                    try:
                        null_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL")
                        null_counts[col] = null_count
                    except Exception as e:
                        logger.warning(f"Error checking nulls for {col}: {e}")
            
            total_nulls = sum(null_counts.values())
            total_cells = total_records * len(columns)
            null_percentage = (total_nulls / total_cells * 100) if total_cells > 0 else 0
            
            # 중복 레코드 (가능한 경우)
            duplicates = 0
            if table_name in self.table_quality_rules:
                unique_columns = self.table_quality_rules[table_name].get('unique_key_columns', [])
                if unique_columns and total_records > 1:
                    try:
                        unique_cols_str = ', '.join(unique_columns)
                        duplicates = await conn.fetchval(f"""
                            SELECT COUNT(*) - COUNT(DISTINCT ({unique_cols_str}))
                            FROM {table_name}
                        """)
                    except Exception as e:
                        logger.warning(f"Error checking duplicates: {e}")
            
            return {
                'total_records': total_records,
                'total_columns': len(columns),
                'columns': columns,
                'nullable_columns': nullable_columns,
                'null_counts': null_counts,
                'null_percentage': null_percentage,
                'duplicates': duplicates
            }
            
    except Exception as e:
        logger.error(f"Error collecting basic statistics for {table_name}: {e}")
        return {'total_records': 0, 'error': str(e)}

async def _check_completeness(self, table_name: str, basic_stats: Dict) -> Tuple[float, List[QualityMetric], List[str]]:
    """완성도 검사"""
    try:
        metrics = []
        issues = []
        
        total_records = basic_stats['total_records']
        null_percentage = basic_stats.get('null_percentage', 0)
        
        # 전체 완성도 점수
        completeness_score = max(0, 100 - null_percentage)
        
        # NULL 값 비율 메트릭
        null_metric = QualityMetric(
            metric_name='null_percentage',
            value=null_percentage,
            threshold=5.0,  # 5% 이하 허용
            status='pass' if null_percentage <= 5.0 else 'warning' if null_percentage <= 10.0 else 'fail',
            description=f'전체 데이터의 {null_percentage:.1f}%가 NULL 값',
            severity='medium' if null_percentage > 10.0 else 'low',
            timestamp=datetime.now()
        )
        metrics.append(null_metric)
        
        # 필수 컬럼 완성도 검사
        if table_name in self.table_quality_rules:
            required_columns = self.table_quality_rules[table_name].get('required_columns', [])
            
            for col in required_columns:
                if col in basic_stats['null_counts']:
                    null_count = basic_stats['null_counts'][col]
                    null_pct = (null_count / total_records * 100) if total_records > 0 else 0
                    
                    if null_pct > 0:
                        issues.append(f"필수 컬럼 '{col}'에 {null_count}개 ({null_pct:.1f}%) NULL 값 존재")
                        
                        # 개별 컬럼 메트릭
                        col_metric = QualityMetric(
                            metric_name=f'{col}_completeness',
                            value=100 - null_pct,
                            threshold=100.0,
                            status='fail' if null_pct > 0 else 'pass',
                            description=f"필수 컬럼 '{col}' 완성도",
                            severity='high' if null_pct > 5.0 else 'medium',
                            timestamp=datetime.now()
                        )
                        metrics.append(col_metric)
        
        return completeness_score, metrics, issues
        
    except Exception as e:
        logger.error(f"Error checking completeness for {table_name}: {e}")
        return 0.0, [], [f"완성도 검사 오류: {str(e)}"]

async def _check_accuracy(self, table_name: str, basic_stats: Dict) -> Tuple[float, List[QualityMetric], List[str]]:
    """정확성 검사"""
    try:
        metrics = []
        issues = []
        accuracy_score = 100.0  # 기본값
        
        if table_name not in self.table_quality_rules:
            return accuracy_score, metrics, issues
        
        rules = self.table_quality_rules[table_name]
        total_records = basic_stats['total_records']
        
        if total_records == 0:
            return accuracy_score, metrics, issues
        
        async with self.db_pool.acquire() as conn:
            # 비즈니스 규칙 검증
            business_rules = rules.get('business_rules', [])
            rule_violations = 0
            
            for rule in business_rules:
                try:
                    # 규칙 위반 건수 조회
                    violation_query = f"SELECT COUNT(*) FROM {table_name} WHERE NOT ({rule})"
                    violations = await conn.fetchval(violation_query)
                    
                    if violations > 0:
                        violation_pct = (violations / total_records) * 100
                        rule_violations += violations
                        issues.append(f"비즈니스 규칙 위반: '{rule}' - {violations}건 ({violation_pct:.1f}%)")
                        
                        # 규칙별 메트릭
                        rule_metric = QualityMetric(
                            metric_name=f'business_rule_{hash(rule) % 1000}',
                            value=100 - violation_pct,
                            threshold=99.0,
                            status='pass' if violation_pct == 0 else 'warning' if violation_pct < 1.0 else 'fail',
                            description=f"비즈니스 규칙: {rule}",
                            severity='medium' if violation_pct > 1.0 else 'low',
                            timestamp=datetime.now()
                        )
                        metrics.append(rule_metric)
                        
                except Exception as e:
                    logger.warning(f"Error checking business rule '{rule}': {e}")
            
            # 범위 검증 (숫자 컬럼)
            if table_name == 'market_data':
                price_range = rules.get('price_range', (0, float('inf')))
                volume_range = rules.get('volume_range', (0, float('inf')))
                
                # 가격 범위 검증
                price_violations = await conn.fetchval(f"""
                    SELECT COUNT(*) FROM {table_name} 
                    WHERE price < {price_range[0]} OR price > {price_range[1]}
                """)
                
                if price_violations > 0:
                    price_violation_pct = (price_violations / total_records) * 100
                    issues.append(f"가격 범위 초과: {price_violations}건 ({price_violation_pct:.1f}%)")
                    rule_violations += price_violations
            
            # 정확성 점수 계산
            if rule_violations > 0:
                violation_rate = (rule_violations / total_records) * 100
                accuracy_score = max(0, 100 - violation_rate * 2)  # 위반 1%당 2점 감점
            else:
                accuracy_score = 100.0
            
            # 전체 정확성 메트릭
            accuracy_metric = QualityMetric(
                metric_name='overall_accuracy',
                value=accuracy_score,
                threshold=95.0,
                status='pass' if accuracy_score >= 95.0 else 'warning' if accuracy_score >= 90.0 else 'fail',
                description=f'전체 데이터 정확성: {accuracy_score:.1f}%',
                severity='low' if accuracy_score >= 95.0 else 'medium' if accuracy_score >= 85.0 else 'high',
                timestamp=datetime.now()
            )
            metrics.append(accuracy_metric)
        
        return accuracy_score, metrics, issues
        
    except Exception as e:
        logger.error(f"Error checking accuracy for {table_name}: {e}")
        return 0.0, [], [f"정확성 검사 오류: {str(e)}"]

async def _check_consistency(self, table_name: str, basic_stats: Dict) -> Tuple[float, List[QualityMetric], List[str]]:
    """일관성 검사"""
    try:
        metrics = []
        issues = []
        consistency_score = 100.0
        
        total_records = basic_stats['total_records']
        duplicates = basic_stats.get('duplicates', 0)
        
        if total_records == 0:
            return consistency_score, metrics, issues
        
        # 중복 데이터 검사
        if duplicates > 0:
            duplicate_pct = (duplicates / total_records) * 100
            issues.append(f"중복 레코드 {duplicates}건 발견 ({duplicate_pct:.1f}%)")
            consistency_score -= duplicate_pct * 5  # 중복 1%당 5점 감점
            
            duplicate_metric = QualityMetric(
                metric_name='duplicate_records',
                value=100 - duplicate_pct,
                threshold=99.0,
                status='pass' if duplicate_pct == 0 else 'warning' if duplicate_pct < 1.0 else 'fail',
                description=f'중복 레코드: {duplicates}건',
                severity='medium' if duplicate_pct > 1.0 else 'low',
                timestamp=datetime.now()
            )
            metrics.append(duplicate_metric)
        
        # 데이터 형식 일관성 검사
        async with self.db_pool.acquire() as conn:
            if table_name == 'market_data':
                # 티커 형식 일관성
                ticker_inconsistency = await conn.
```