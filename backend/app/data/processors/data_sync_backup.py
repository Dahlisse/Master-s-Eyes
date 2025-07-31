# backend/app/data/processors/data_sync_backup.py

“””
데이터 동기화 및 백업 시스템

- 실시간 데이터 동기화
- 자동 백업 및 복구
- 데이터 무결성 검증
- 장애 복구 시스템
  “””
  import asyncio
  import aiofiles
  import asyncpg
  import pandas as pd
  from datetime import datetime, timedelta
  from typing import Dict, List, Optional, Any, Union
  import logging
  from dataclasses import dataclass, asdict
  import json
  import pickle
  import gzip
  import hashlib
  import os
  from pathlib import Path
  import shutil
  import sqlite3

from app.core.config import get_settings
from app.core.database import get_database
from app.core.redis import get_redis_client
from app.data.processors.data_validator import DataValidator

settings = get_settings()
logger = logging.getLogger(**name**)

@dataclass
class BackupMetadata:
“”“백업 메타데이터”””
backup_id: str
backup_type: str  # ‘full’, ‘incremental’, ‘emergency’
created_at: datetime
file_path: str
file_size: int
checksum: str
table_counts: Dict[str, int]
compression: str
retention_days: int
status: str  # ‘creating’, ‘completed’, ‘failed’, ‘expired’

@dataclass
class SyncStatus:
“”“동기화 상태”””
last_sync_time: datetime
sync_type: str  # ‘full’, ‘incremental’
records_synced: int
errors_count: int
duration_seconds: float
status: str  # ‘running’, ‘completed’, ‘failed’

class DataSyncBackupManager:
“”“데이터 동기화 및 백업 관리자”””

```
def __init__(self):
    self.db_pool = None
    self.redis_client = get_redis_client()
    self.validator = DataValidator()
    
    # 백업 설정
    self.backup_config = {
        'backup_dir': Path(settings.BACKUP_DIR if hasattr(settings, 'BACKUP_DIR') else './backups'),
        'full_backup_interval_hours': 24,  # 전체 백업 주기
        'incremental_backup_interval_hours': 6,  # 증분 백업 주기
        'retention_days': 30,  # 백업 보관 기간
        'compression_level': 6,  # gzip 압축 레벨
        'max_concurrent_backups': 3,  # 동시 백업 작업 수
        'chunk_size': 10000,  # 배치 처리 크기
    }
    
    # 동기화할 테이블 목록
    self.sync_tables = {
        'market_data': {
            'primary_key': 'time, ticker',
            'time_column': 'time',
            'priority': 1,  # 높을수록 우선순위
            'batch_size': 5000,
            'retention_days': 90
        },
        'stock_prices': {
            'primary_key': 'id',
            'time_column': 'timestamp',
            'priority': 1,
            'batch_size': 10000,
            'retention_days': 365
        },
        'economic_indicators': {
            'primary_key': 'series_id, date',
            'time_column': 'date',
            'priority': 2,
            'batch_size': 1000,
            'retention_days': 1825  # 5년
        },
        'news_articles': {
            'primary_key': 'id',
            'time_column': 'published_time',
            'priority': 3,
            'batch_size': 2000,
            'retention_days': 365
        },
        'portfolios': {
            'primary_key': 'id',
            'time_column': 'created_at',
            'priority': 4,
            'batch_size': 1000,
            'retention_days': 1825
        },
        'backtest_results': {
            'primary_key': 'id',
            'time_column': 'created_at',
            'priority': 4,
            'batch_size': 500,
            'retention_days': 730
        },
        'chat_logs': {
            'primary_key': 'id',
            'time_column': 'created_at',
            'priority': 5,
            'batch_size': 5000,
            'retention_days': 180
        }
    }
    
    # 백업 디렉토리 생성
    self.backup_config['backup_dir'].mkdir(parents=True, exist_ok=True)

async def __aenter__(self):
    """비동기 컨텍스트 매니저 진입"""
    self.db_pool = await get_database()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    """비동기 컨텍스트 매니저 종료"""
    if self.db_pool:
        await self.db_pool.close()

async def create_full_backup(self, backup_type: str = 'scheduled') -> Optional[BackupMetadata]:
    """전체 데이터베이스 백업 생성"""
    try:
        backup_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting full backup: {backup_id}")
        
        # 백업 메타데이터 초기화
        backup_start = datetime.now()
        backup_file = self.backup_config['backup_dir'] / f"{backup_id}.sql.gz"
        
        table_counts = {}
        total_size = 0
        
        # 백업 상태를 Redis에 저장
        await self._update_backup_status(backup_id, 'creating', 0)
        
        async with aiofiles.open(backup_file, 'wb') as f:
            with gzip.GzipFile(fileobj=f.raw, mode='wb', 
                             compresslevel=self.backup_config['compression_level']) as gz_file:
                
                # 데이터베이스 스키마 백업
                schema_sql = await self._export_schema()
                gz_file.write(schema_sql.encode('utf-8'))
                gz_file.write(b'\n\n')
                
                # 테이블별 데이터 백업 (우선순위 순)
                sorted_tables = sorted(
                    self.sync_tables.items(),
                    key=lambda x: x[1]['priority']
                )
                
                for i, (table_name, config) in enumerate(sorted_tables):
                    try:
                        logger.info(f"Backing up table: {table_name}")
                        
                        # 테이블 레코드 수 확인
                        async with self.db_pool.acquire() as conn:
                            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                            table_counts[table_name] = count
                        
                        # 데이터 export
                        table_data = await self._export_table_data(table_name, config['batch_size'])
                        
                        if table_data:
                            gz_file.write(f"-- Table: {table_name}\n".encode('utf-8'))
                            gz_file.write(table_data.encode('utf-8'))
                            gz_file.write(b'\n\n')
                        
                        # 진행률 업데이트
                        progress = int((i + 1) / len(sorted_tables) * 100)
                        await self._update_backup_status(backup_id, 'creating', progress)
                        
                    except Exception as e:
                        logger.error(f"Error backing up table {table_name}: {e}")
                        table_counts[table_name] = -1  # 에러 표시
        
        # 백업 파일 무결성 검증
        file_size = backup_file.stat().st_size
        checksum = await self._calculate_file_checksum(backup_file)
        
        # 백업 메타데이터 생성
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type='full',
            created_at=backup_start,
            file_path=str(backup_file),
            file_size=file_size,
            checksum=checksum,
            table_counts=table_counts,
            compression='gzip',
            retention_days=self.backup_config['retention_days'],
            status='completed'
        )
        
        # 메타데이터 저장
        await self._save_backup_metadata(metadata)
        await self._update_backup_status(backup_id, 'completed', 100)
        
        duration = (datetime.now() - backup_start).total_seconds()
        logger.info(f"Full backup completed: {backup_id}, Size: {file_size/1024/1024:.1f}MB, Duration: {duration:.1f}s")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error creating full backup: {e}")
        await self._update_backup_status(backup_id, 'failed', 0)
        return None

async def create_incremental_backup(self, hours_back: int = 6) -> Optional[BackupMetadata]:
    """증분 백업 생성 (최근 N시간 변경사항만)"""
    try:
        backup_id = f"incr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting incremental backup: {backup_id} (last {hours_back} hours)")
        
        backup_start = datetime.now()
        cutoff_time = backup_start - timedelta(hours=hours_back)
        backup_file = self.backup_config['backup_dir'] / f"{backup_id}.sql.gz"
        
        table_counts = {}
        
        await self._update_backup_status(backup_id, 'creating', 0)
        
        async with aiofiles.open(backup_file, 'wb') as f:
            with gzip.GzipFile(fileobj=f.raw, mode='wb', 
                             compresslevel=self.backup_config['compression_level']) as gz_file:
                
                # 증분 백업 헤더
                header = f"-- Incremental Backup: {backup_id}\n"
                header += f"-- From: {cutoff_time.isoformat()}\n"
                header += f"-- To: {backup_start.isoformat()}\n\n"
                gz_file.write(header.encode('utf-8'))
                
                # 테이블별 증분 데이터
                for i, (table_name, config) in enumerate(self.sync_tables.items()):
                    try:
                        # 시간 컬럼이 있는 테이블만 증분 백업
                        if 'time_column' not in config:
                            continue
                        
                        time_column = config['time_column']
                        
                        # 변경된 레코드 수 확인
                        async with self.db_pool.acquire() as conn:
                            count_query = f"""
                                SELECT COUNT(*) FROM {table_name} 
                                WHERE {time_column} >= $1
                            """
                            count = await conn.fetchval(count_query, cutoff_time)
                            table_counts[table_name] = count
                        
                        if count > 0:
                            # 증분 데이터 export
                            incremental_data = await self._export_incremental_data(
                                table_name, time_column, cutoff_time, config['batch_size']
                            )
                            
                            if incremental_data:
                                gz_file.write(f"-- Table: {table_name} ({count} records)\n".encode('utf-8'))
                                gz_file.write(incremental_data.encode('utf-8'))
                                gz_file.write(b'\n\n')
                        
                        # 진행률 업데이트
                        progress = int((i + 1) / len(self.sync_tables) * 100)
                        await self._update_backup_status(backup_id, 'creating', progress)
                        
                    except Exception as e:
                        logger.error(f"Error in incremental backup for {table_name}: {e}")
                        table_counts[table_name] = -1
        
        # 백업 파일 정보
        file_size = backup_file.stat().st_size
        checksum = await self._calculate_file_checksum(backup_file)
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type='incremental',
            created_at=backup_start,
            file_path=str(backup_file),
            file_size=file_size,
            checksum=checksum,
            table_counts=table_counts,
            compression='gzip',
            retention_days=7,  # 증분 백업은 짧은 보관
            status='completed'
        )
        
        await self._save_backup_metadata(metadata)
        await self._update_backup_status(backup_id, 'completed', 100)
        
        total_records = sum(count for count in table_counts.values() if count > 0)
        duration = (datetime.now() - backup_start).total_seconds()
        logger.info(f"Incremental backup completed: {backup_id}, Records: {total_records}, Duration: {duration:.1f}s")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error creating incremental backup: {e}")
        await self._update_backup_status(backup_id, 'failed', 0)
        return None

async def sync_data_sources(self, force_full_sync: bool = False) -> SyncStatus:
    """외부 데이터 소스와 동기화"""
    try:
        sync_start = datetime.now()
        sync_type = 'full' if force_full_sync else 'incremental'
        
        logger.info(f"Starting {sync_type} data synchronization")
        
        # 마지막 동기화 시간 확인
        last_sync = await self._get_last_sync_time()
        cutoff_time = last_sync if not force_full_sync and last_sync else sync_start - timedelta(days=7)
        
        total_synced = 0
        total_errors = 0
        
        # 실시간 데이터 동기화 (우선순위 높은 테이블부터)
        priority_tables = sorted(
            self.sync_tables.items(),
            key=lambda x: x[1]['priority']
        )
        
        for table_name, config in priority_tables:
            try:
                logger.info(f"Syncing table: {table_name}")
                
                synced_count = await self._sync_table_data(
                    table_name, 
                    config,
                    cutoff_time,
                    force_full_sync
                )
                
                total_synced += synced_count
                
                # 배치 간 지연 (DB 부하 방지)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error syncing table {table_name}: {e}")
                total_errors += 1
        
        # Redis 캐시 무효화 (새로운 데이터 반영)
        await self._invalidate_related_caches()
        
        # 동기화 결과
        duration = (datetime.now() - sync_start).total_seconds()
        status = SyncStatus(
            last_sync_time=sync_start,
            sync_type=sync_type,
            records_synced=total_synced,
            errors_count=total_errors,
            duration_seconds=duration,
            status='completed' if total_errors == 0 else 'completed_with_errors'
        )
        
        # 동기화 상태 저장
        await self._save_sync_status(status)
        
        logger.info(f"Data sync completed: {total_synced} records, {total_errors} errors, {duration:.1f}s")
        return status
        
    except Exception as e:
        logger.error(f"Error in data synchronization: {e}")
        return SyncStatus(
            last_sync_time=datetime.now(),
            sync_type=sync_type,
            records_synced=0,
            errors_count=1,
            duration_seconds=0,
            status='failed'
        )

async def restore_from_backup(self, backup_id: str, target_tables: Optional[List[str]] = None) -> bool:
    """백업에서 데이터 복원"""
    try:
        logger.info(f"Starting restore from backup: {backup_id}")
        
        # 백업 메타데이터 로드
        metadata = await self._load_backup_metadata(backup_id)
        if not metadata:
            logger.error(f"Backup metadata not found: {backup_id}")
            return False
        
        backup_file = Path(metadata.file_path)
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        # 백업 파일 무결성 검증
        if not await self._verify_backup_integrity(metadata):
            logger.error(f"Backup integrity verification failed: {backup_id}")
            return False
        
        restore_start = datetime.now()
        
        # 백업 파일 읽기 및 복원
        async with aiofiles.open(backup_file, 'rb') as f:
            with gzip.GzipFile(fileobj=f.raw, mode='rb') as gz_file:
                sql_content = gz_file.read().decode('utf-8')
        
        # SQL 스크립트를 구문별로 분할
        sql_statements = self._parse_sql_statements(sql_content)
        
        # 트랜잭션으로 복원 실행
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                restored_count = 0
                
                for statement in sql_statements:
                    try:
                        # 테이블별 필터링 (지정된 경우)
                        if target_tables and not self._statement_affects_tables(statement, target_tables):
                            continue
                        
                        await conn.execute(statement)
                        restored_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error executing SQL statement: {e}")
                        # 치명적 에러가 아니면 계속 진행
                        if "duplicate key" not in str(e).lower():
                            raise
        
        # 복원 후 데이터 검증
        verification_result = await self._verify_restored_data(metadata, target_tables)
        
        duration = (datetime.now() - restore_start).total_seconds()
        logger.info(f"Restore completed: {backup_id}, Statements: {restored_count}, Duration: {duration:.1f}s")
        
        return verification_result
        
    except Exception as e:
        logger.error(f"Error restoring from backup {backup_id}: {e}")
        return False

async def cleanup_old_backups(self) -> int:
    """오래된 백업 파일 정리"""
    try:
        logger.info("Starting backup cleanup")
        
        cleaned_count = 0
        cutoff_date = datetime.now() - timedelta(days=self.backup_config['retention_days'])
        
        # 모든 백업 메타데이터 조회
        backup_list = await self._list_all_backups()
        
        for metadata in backup_list:
            try:
                # 보관 기간 확인
                if metadata.created_at < cutoff_date:
                    # 백업 파일 삭제
                    backup_file = Path(metadata.file_path)
                    if backup_file.exists():
                        backup_file.unlink()
                    
                    # 메타데이터 삭제
                    await self._delete_backup_metadata(metadata.backup_id)
                    
                    cleaned_count += 1
                    logger.info(f"Cleaned up backup: {metadata.backup_id}")
            
            except Exception as e:
                logger.error(f"Error cleaning backup {metadata.backup_id}: {e}")
        
        logger.info(f"Backup cleanup completed: {cleaned_count} backups removed")
        return cleaned_count
        
    except Exception as e:
        logger.error(f"Error in backup cleanup: {e}")
        return 0

async def verify_data_integrity(self, tables: Optional[List[str]] = None) -> Dict[str, Any]:
    """데이터 무결성 검증"""
    try:
        logger.info("Starting data integrity verification")
        
        verification_results = {}
        tables_to_check = tables or list(self.sync_tables.keys())
        
        for table_name in tables_to_check:
            try:
                result = await self._verify_table_integrity(table_name)
                verification_results[table_name] = result
                
            except Exception as e:
                logger.error(f"Error verifying table {table_name}: {e}")
                verification_results[table_name] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        # 전체 요약
        total_tables = len(verification_results)
        passed_tables = sum(1 for r in verification_results.values() if r.get('status') == 'passed')
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tables': total_tables,
            'passed_tables': passed_tables,
            'failed_tables': total_tables - passed_tables,
            'overall_status': 'passed' if passed_tables == total_tables else 'failed',
            'table_results': verification_results
        }
        
        logger.info(f"Data integrity verification completed: {passed_tables}/{total_tables} tables passed")
        return summary
        
    except Exception as e:
        logger.error(f"Error in data integrity verification: {e}")
        return {'status': 'error', 'message': str(e)}

# 내부 헬퍼 메서드들
async def _export_schema(self) -> str:
    """데이터베이스 스키마 export"""
    try:
        async with self.db_pool.acquire() as conn:
            # PostgreSQL 스키마 정보 조회
            schema_query = """
                SELECT 
                    schemaname, tablename, 
                    pg_get_tabledef(schemaname||'.'||tablename) as table_def
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename;
            """
            
            tables = await conn.fetch(schema_query)
            
            schema_sql = "-- Database Schema Export\n"
            schema_sql += f"-- Generated: {datetime.now().isoformat()}\n\n"
            
            for row in tables:
                schema_sql += f"-- Table: {row['tablename']}\n"
                schema_sql += f"{row['table_def']};\n\n"
            
            return schema_sql
            
    except Exception as e:
        logger.error(f"Error exporting schema: {e}")
        return "-- Schema export failed\n"

async def _export_table_data(self, table_name: str, batch_size: int) -> str:
    """테이블 데이터 export"""
    try:
        async with self.db_pool.acquire() as conn:
            # 테이블 구조 확인
            columns_query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = $1 
                ORDER BY ordinal_position;
            """
            columns = await conn.fetch(columns_query, table_name)
            column_names = [col['column_name'] for col in columns]
            
            # 데이터 조회 및 INSERT 문 생성
            data_sql = ""
            offset = 0
            
            while True:
                query = f"""
                    SELECT * FROM {table_name} 
                    ORDER BY {column_names[0]} 
                    LIMIT {batch_size} OFFSET {offset}
                """
                
                rows = await conn.fetch(query)
                if not rows:
                    break
                
                for row in rows:
                    values = []
                    for value in row.values():
                        if value is None:
                            values.append('NULL')
                        elif isinstance(value, str):
                            values.append(f"'{value.replace(\"'\", \"''\")}'")
                        elif isinstance(value, datetime):
                            values.append(f"'{value.isoformat()}'")
                        else:
                            values.append(str(value))
                    
                    data_sql += f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(values)});\n"
                
                offset += batch_size
            
            return data_sql
            
    except Exception as e:
        logger.error(f"Error exporting table data for {table_name}: {e}")
        return f"-- Error exporting {table_name}: {e}\n"

async def _export_incremental_data(self, table_name: str, time_column: str, cutoff_time: datetime, batch_size: int) -> str:
    """증분 데이터 export"""
    try:
        async with self.db_pool.acquire() as conn:
            # 컬럼 정보
            columns_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = $1 
                ORDER BY ordinal_position;
            """
            columns = await conn.fetch(columns_query, table_name)
            column_names = [col['column_name'] for col in columns]
            
            # 증분 데이터 조회
            query = f"""
                SELECT * FROM {table_name} 
                WHERE {time_column} >= $1 
                ORDER BY {time_column}
            """
            
            rows = await conn.fetch(query, cutoff_time)
            
            data_sql = ""
            for row in rows:
                values = []
                for value in row.values():
                    if value is None:
                        values.append('NULL')
                    elif isinstance(value, str):
                        values.append(f"'{value.replace(\"'\", \"''\")}'")
                    elif isinstance(value, datetime):
                        values.append(f"'{value.isoformat()}'")
                    else:
                        values.append(str(value))
                
                data_sql += f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(values)}) ON CONFLICT DO NOTHING;\n"
            
            return data_sql
            
    except Exception as e:
        logger.error(f"Error exporting incremental data for {table_name}: {e}")
        return f"-- Error exporting incremental data for {table_name}: {e}\n"

async def _calculate_file_checksum(self, file_path: Path) -> str:
    """파일 체크섬 계산"""
    try:
        hash_sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
        
    except Exception as e:
        logger.error(f"Error calculating checksum for {file_path}: {e}")
        return ""

async def _save_backup_metadata(self, metadata: BackupMetadata):
    """백업 메타데이터 저장"""
    try:
        # Redis에 메타데이터 저장
        cache_key = f"backup_metadata:{metadata.backup_id}"
        metadata_json = json.dumps(asdict(metadata), default=str)
        
        await self.redis_client.setex(cache_key, 86400 * 30, metadata_json)  # 30일 TTL
        
        # 데이터베이스에도 저장 (영구 보관)
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO backup_metadata 
                (backup_id, backup_type, created_at, file_path, file_size, 
                 checksum, table_counts, compression, retention_days, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (backup_id) DO UPDATE SET
                status = EXCLUDED.status,
                file_size = EXCLUDED.file_size,
                checksum = EXCLUDED.checksum
            """, 
            metadata.backup_id, metadata.backup_type, metadata.created_at,
            metadata.file_path, metadata.file_size, metadata.checksum,
            json.dumps(metadata.table_counts), metadata.compression,
            metadata.retention_days, metadata.status)
        
    except Exception as e:
        logger.error(f"Error saving backup metadata: {e}")

async def _update_backup_status(self, backup_id: str, status: str, progress: int):
    """백업 상태 업데이트"""
    try:
        status_data = {
            'backup_id': backup_id,
            'status': status,
            'progress': progress,
            'updated_at': datetime.now().isoformat()
        }
        
        cache_key = f"backup_status:{backup_id}"
        await self.redis_client.setex(cache_key, 3600, json.dumps(status_data))
        
    except Exception as e:
        logger.error(f"Error updating backup status: {e}")

async def _sync_table_data(self, table_name: str, config: Dict, cutoff_time: datetime, force_full: bool) -> int:
    """테이블 데이터 동기화"""
    try:
        synced_count = 0
        batch_size = config['batch_size']
        
        async with self.db_pool.acquire() as conn:
            if force_full:
                # 전체 동기화 - 데이터 검증 및 누락 보완
                query = f"SELECT COUNT(*) FROM {table_name}"
                total_count = await conn.fetchval(query)
                
                # 배치별로 데이터 검증
                for offset in range(0, total_count, batch_size):
                    verification_query = f"""
                        SELECT * FROM {table_name} 
                        ORDER BY {config.get('primary_key', 'id')}
                        LIMIT {batch_size} OFFSET {offset}
                    """
                    
                    rows = await conn.fetch(verification_query)
                    
                    # 데이터 검증 및 복구 로직
                    for row in rows:
                        if await self.validator.validate_row_data(dict(row)):
                            synced_count += 1
                        else:
                            # 데이터 복구 시도
                            await self._repair_corrupted_data(table_name, dict(row))
            else:
                # 증분 동기화 - 최근 변경사항만
                if 'time_column' in config:
                    time_column = config['time_column']
                    sync_query = f"""
                        SELECT * FROM {table_name} 
                        WHERE {time_column} >= $1 
                        ORDER BY {time_column}
                    """
                    
                    rows = await conn.fetch(sync_query, cutoff_time)
                    synced_count = len(rows)
                    
                    # 중복 데이터 제거 및 최신화
                    await self._deduplicate_table_data(table_name, config)
        
        logger.info(f"Synced {synced_count} records for table {table_name}")
        return synced_count
        
    except Exception as e:
        logger.error(f"Error syncing table {table_name}: {e}")
        return 0

async def _get_last_sync_time(self) -> Optional[datetime]:
    """마지막 동기화 시간 조회"""
    try:
        cache_key = "last_data_sync_time"
        cached_time = await self.redis_client.get(cache_key)
        
        if cached_time:
            return datetime.fromisoformat(cached_time.decode('utf-8'))
        
        # DB에서 조회
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT MAX(last_sync_time) FROM sync_status 
                WHERE status = 'completed'
            """)
            
            return result
            
    except Exception as e:
        logger.error(f"Error getting last sync time: {e}")
        return None

async def _save_sync_status(self, status: SyncStatus):
    """동기화 상태 저장"""
    try:
        # Redis 캐시
        cache_key = "last_data_sync_status"
        status_json = json.dumps(asdict(status), default=str)
        await self.redis_client.setex(cache_key, 86400, status_json)
        
        # DB 저장
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sync_status 
                (last_sync_time, sync_type, records_synced, errors_count, 
                 duration_seconds, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
            status.last_sync_time, status.sync_type, status.records_synced,
            status.errors_count, status.duration_seconds, status.status,
            datetime.now())
        
        # 마지막 동기화 시간 업데이트
        await self.redis_client.set(
            "last_data_sync_time", 
            status.last_sync_time.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error saving sync status: {e}")

async def _invalidate_related_caches(self):
    """관련 캐시 무효화"""
    try:
        # 패턴별 캐시 삭제
        cache_patterns = [
            "yahoo_*",
            "fred_*", 
            "news_*",
            "market_data_*",
            "portfolio_*"
        ]
        
        for pattern in cache_patterns:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        
        logger.info("Cache invalidation completed")
        
    except Exception as e:
        logger.error(f"Error invalidating caches: {e}")

async def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
    """백업 메타데이터 로드"""
    try:
        # Redis에서 먼저 조회
        cache_key = f"backup_metadata:{backup_id}"
        cached_data = await self.redis_client.get(cache_key)
        
        if cached_data:
            metadata_dict = json.loads(cached_data)
            metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
            return BackupMetadata(**metadata_dict)
        
        # DB에서 조회
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM backup_metadata WHERE backup_id = $1
            """, backup_id)
            
            if row:
                return BackupMetadata(
                    backup_id=row['backup_id'],
                    backup_type=row['backup_type'],
                    created_at=row['created_at'],
                    file_path=row['file_path'],
                    file_size=row['file_size'],
                    checksum=row['checksum'],
                    table_counts=json.loads(row['table_counts']),
                    compression=row['compression'],
                    retention_days=row['retention_days'],
                    status=row['status']
                )
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading backup metadata for {backup_id}: {e}")
        return None

async def _verify_backup_integrity(self, metadata: BackupMetadata) -> bool:
    """백업 파일 무결성 검증"""
    try:
        backup_file = Path(metadata.file_path)
        
        # 파일 존재 확인
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        # 파일 크기 확인
        actual_size = backup_file.stat().st_size
        if actual_size != metadata.file_size:
            logger.error(f"File size mismatch: expected {metadata.file_size}, got {actual_size}")
            return False
        
        # 체크섬 검증
        actual_checksum = await self._calculate_file_checksum(backup_file)
        if actual_checksum != metadata.checksum:
            logger.error(f"Checksum mismatch: expected {metadata.checksum}, got {actual_checksum}")
            return False
        
        logger.info(f"Backup integrity verified: {metadata.backup_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying backup integrity: {e}")
        return False

def _parse_sql_statements(self, sql_content: str) -> List[str]:
    """SQL 스크립트를 개별 구문으로 분할"""
    try:
        # 주석 제거 및 구문 분할
        statements = []
        current_statement = ""
        
        for line in sql_content.split('\n'):
            line = line.strip()
            
            # 주석 라인 건너뛰기
            if line.startswith('--') or not line:
                continue
            
            current_statement += line + '\n'
            
            # 구문 종료 감지
            if line.endswith(';'):
                if current_statement.strip():
                    statements.append(current_statement.strip())
                current_statement = ""
        
        # 마지막 구문 처리
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements
        
    except Exception as e:
        logger.error(f"Error parsing SQL statements: {e}")
        return []

def _statement_affects_tables(self, statement: str, target_tables: List[str]) -> bool:
    """SQL 구문이 대상 테이블에 영향을 주는지 확인"""
    try:
        statement_upper = statement.upper()
        
        for table in target_tables:
            if f" {table.upper()} " in statement_upper or \
               f" {table.upper()}(" in statement_upper or \
               statement_upper.startswith(f"INSERT INTO {table.upper()}") or \
               statement_upper.startswith(f"CREATE TABLE {table.upper()}"):
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking statement table affection: {e}")
        return True  # 에러시 안전하게 true 반환

async def _verify_restored_data(self, metadata: BackupMetadata, target_tables: Optional[List[str]]) -> bool:
    """복원된 데이터 검증"""
    try:
        verification_passed = True
        tables_to_verify = target_tables or list(metadata.table_counts.keys())
        
        async with self.db_pool.acquire() as conn:
            for table_name in tables_to_verify:
                try:
                    # 테이블 존재 확인
                    exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = $1
                        )
                    """, table_name)
                    
                    if not exists:
                        logger.error(f"Table {table_name} not found after restore")
                        verification_passed = False
                        continue
                    
                    # 레코드 수 확인 (완전 복원의 경우)
                    if metadata.backup_type == 'full':
                        expected_count = metadata.table_counts.get(table_name, 0)
                        if expected_count > 0:
                            actual_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                            
                            if actual_count < expected_count * 0.95:  # 5% 오차 허용
                                logger.warning(f"Table {table_name}: expected ~{expected_count}, got {actual_count}")
                    
                except Exception as e:
                    logger.error(f"Error verifying table {table_name}: {e}")
                    verification_passed = False
        
        return verification_passed
        
    except Exception as e:
        logger.error(f"Error verifying restored data: {e}")
        return False

async def _list_all_backups(self) -> List[BackupMetadata]:
    """모든 백업 목록 조회"""
    try:
        backup_list = []
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM backup_metadata 
                ORDER BY created_at DESC
            """)
            
            for row in rows:
                metadata = BackupMetadata(
                    backup_id=row['backup_id'],
                    backup_type=row['backup_type'],
                    created_at=row['created_at'],
                    file_path=row['file_path'],
                    file_size=row['file_size'],
                    checksum=row['checksum'],
                    table_counts=json.loads(row['table_counts']),
                    compression=row['compression'],
                    retention_days=row['retention_days'],
                    status=row['status']
                )
                backup_list.append(metadata)
        
        return backup_list
        
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        return []

async def _delete_backup_metadata(self, backup_id: str):
    """백업 메타데이터 삭제"""
    try:
        # Redis에서 삭제
        cache_key = f"backup_metadata:{backup_id}"
        await self.redis_client.delete(cache_key)
        
        # DB에서 삭제
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM backup_metadata WHERE backup_id = $1
            """, backup_id)
        
    except Exception as e:
        logger.error(f"Error deleting backup metadata for {backup_id}: {e}")

async def _verify_table_integrity(self, table_name: str) -> Dict[str, Any]:
    """테이블 무결성 검증"""
    try:
        result = {
            'table_name': table_name,
            'status': 'passed',
            'checks': {},
            'issues': []
        }
        
        async with self.db_pool.acquire() as conn:
            # 1. 테이블 존재 확인
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
            """, table_name)
            
            result['checks']['table_exists'] = exists
            
            if not exists:
                result['status'] = 'failed'
                result['issues'].append('Table does not exist')
                return result
            
            # 2. 레코드 수 확인
            record_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            result['checks']['record_count'] = record_count
            
            # 3. NULL 값 검증 (필수 컬럼)
            if table_name in self.sync_tables:
                config = self.sync_tables[table_name]
                if 'time_column' in config:
                    time_column = config['time_column']
                    null_count = await conn.fetchval(f"""
                        SELECT COUNT(*) FROM {table_name} 
                        WHERE {time_column} IS NULL
                    """)
                    
                    result['checks']['null_time_values'] = null_count
                    
                    if null_count > 0:
                        result['issues'].append(f'{null_count} records with NULL time values')
            
            # 4. 중복 데이터 확인
            if table_name in self.sync_tables:
                config = self.sync_tables[table_name]
                primary_key = config.get('primary_key', 'id')
                
                duplicate_count = await conn.fetchval(f"""
                    SELECT COUNT(*) - COUNT(DISTINCT ({primary_key})) 
                    FROM {table_name}
                """)
                
                result['checks']['duplicate_records'] = duplicate_count
                
                if duplicate_count > 0:
                    result['issues'].append(f'{duplicate_count} duplicate records found')
            
            # 5. 최근 데이터 확인 (시계열 데이터)
            if table_name in self.sync_tables and 'time_column' in self.sync_tables[table_name]:
                time_column = self.sync_tables[table_name]['time_column']
                latest_time = await conn.fetchval(f"""
                    SELECT MAX({time_column}) FROM {table_name}
                """)
                
                result['checks']['latest_data_time'] = latest_time.isoformat() if latest_time else None
                
                if latest_time:
                    hours_since_latest = (datetime.now() - latest_time).total_seconds() / 3600
                    if hours_since_latest > 48:  # 48시간 이상 오래된 데이터
                        result['issues'].append(f'Latest data is {hours_since_latest:.1f} hours old')
        
        # 이슈가 있으면 실패로 마크
        if result['issues']:
            result['status'] = 'failed' if len(result['issues']) > 2 else 'warning'
        
        return result
        
    except Exception as e:
        logger.error(f"Error verifying table integrity for {table_name}: {e}")
        return {
            'table_name': table_name,
            'status': 'error',
            'message': str(e)
        }

async def _repair_corrupted_data(self, table_name: str, row_data: Dict):
    """손상된 데이터 복구"""
    try:
        # 간단한 데이터 복구 로직
        # 실제 환경에서는 더 정교한 복구 전략 필요
        
        if table_name == 'market_data':
            # 시장 데이터 복구
            if 'price' in row_data and (row_data['price'] is None or row_data['price'] <= 0):
                # 이전 가격으로 대체
                previous_price = await self._get_previous_price(row_data['ticker'])
                if previous_price:
                    row_data['price'] = previous_price
                    await self._update_corrupted_row(table_name, row_data)
        
        elif table_name == 'news_articles':
            # 뉴스 데이터 복구
            if 'content' in row_data and not row_data['content']:
                # 제목으로 내용 대체
                row_data['content'] = row_data.get('title', 'Content unavailable')
                await self._update_corrupted_row(table_name, row_data)
        
        logger.info(f"Repaired corrupted data in {table_name}")
        
    except Exception as e:
        logger.error(f"Error repairing corrupted data in {table_name}: {e}")

async def _deduplicate_table_data(self, table_name: str, config: Dict):
    """테이블 중복 데이터 제거"""
    try:
        primary_key = config.get('primary_key', 'id')
        
        async with self.db_pool.acquire() as conn:
            # 중복 레코드 찾기 및 제거
            dedupe_query = f"""
                DELETE FROM {table_name} a USING {table_name} b 
                WHERE a.ctid < b.ctid 
                AND a.{primary_key.split(',')[0]} = b.{primary_key.split(',')[0]}
            """
            
            result = await conn.execute(dedupe_query)
            
            if result != "DELETE 0":
                logger.info(f"Removed duplicate records from {table_name}: {result}")
        
    except Exception as e:
        logger.error(f"Error deduplicating table {table_name}: {e}")

async def _get_previous_price(self, ticker: str) -> Optional[float]:
    """이전 가격 조회 (데이터 복구용)"""
    try:
        async with self.db_pool.acquire() as conn:
            price = await conn.fetchval("""
                SELECT price FROM market_data 
                WHERE ticker = $1 AND price > 0 
                ORDER BY time DESC 
                LIMIT 1
            """, ticker)
            
            return float(price) if price else None
            
    except Exception as e:
        logger.error(f"Error getting previous price for {ticker}: {e}")
        return None

async def _update_corrupted_row(self, table_name: str, row_data: Dict):
    """손상된 행 업데이트"""
    try:
        # 동적 UPDATE 쿼리 생성
        set_clauses = []
        values = []
        
        for key, value in row_data.items():
            if key != 'id':  # ID는 업데이트하지 않음
                set_clauses.append(f"{key} = ${len(values) + 1}")
                values.append(value)
        
        if set_clauses:
            update_query = f"""
                UPDATE {table_name} 
                SET {', '.join(set_clauses)} 
                WHERE id = ${len(values) + 1}
            """
            values.append(row_data['id'])
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(update_query, *values)
        
    except Exception as e:
        logger.error(f"Error updating corrupted row in {table_name}: {e}")
```

# 비동기 함수들 (스케줄러/API용)

async def create_scheduled_backup(backup_type: str = ‘incremental’) -> Optional[BackupMetadata]:
“”“스케줄된 백업 생성”””
async with DataSyncBackupManager() as manager:
if backup_type == ‘full’:
return await manager.create_full_backup(‘scheduled’)
else:
return await manager.create_incremental_backup()

async def sync_all_data_sources(force_full: bool = False) -> SyncStatus:
“”“모든 데이터 소스 동기화”””
async with DataSyncBackupManager() as manager:
return await manager.sync_data_sources(force_full)

async def cleanup_old_backup_files() -> int:
“”“오래된 백업 파일 정리”””
async with DataSyncBackupManager() as manager:
return await manager.cleanup_old_backups()

async def verify_system_data_integrity() -> Dict[str, Any]:
“”“시스템 전체 데이터 무결성 검증”””
async with DataSyncBackupManager() as manager:
return await manager.verify_data_integrity()

# 동기 래퍼 함수들 (Celery 태스크용)

def sync_create_backup(backup_type: str = ‘incremental’):
“”“동기 방식 백업 생성”””
return asyncio.run(create_scheduled_backup(backup_type))

def sync_data_synchronization(force_full: bool = False):
“”“동기 방식 데이터 동기화”””
return asyncio.run(sync_all_data_sources(force_full))

def sync_cleanup_backups():
“”“동기 방식 백업 정리”””
return asyncio.run(cleanup_old_backup_files())

def sync_verify_data_integrity():
“”“동기 방식 데이터 무결성 검증”””
return asyncio.run(verify_system_data_integrity())

if **name** == “**main**”:
# 테스트 실행
async def main():
async with DataSyncBackupManager() as manager:
print(”=== 데이터 동기화 및 백업 시스템 테스트 ===”)

```
        # 1. 데이터 무결성 검증 테스트
        print("\n1. 데이터 무결성 검증...")
        integrity_result = await manager.verify_data_integrity(['market_data', 'portfolios'])
        print(f"검증 결과: {integrity_result['overall_status']}")
        print(f"통과: {integrity_result['passed_tables']}/{integrity_result['total_tables']} 테이블")
        
        # 2. 증분 백업 테스트
        print("\n2. 증분 백업 생성...")
        backup_metadata = await manager.create_incremental_backup(hours_back=24)
        if backup_metadata:
            print(f"백업 ID: {backup_metadata.backup_id}")
            print(f"파일 크기: {backup_metadata.file_size / 1024 / 1024:.1f}MB")
            print(f"테이블 수: {len(backup_metadata.table_counts)}")
        
        # 3. 데이터 동기화 테스트
        print("\n3. 데이터 동기화...")
        sync_result = await manager.sync_data_sources(force_full_sync=False)
        print(f"동기화 상태: {sync_result.status}")
        print(f"동기화된 레코드: {sync_result.records_synced}")
        print(f"소요 시간: {sync_result.duration_seconds:.1f}초")
        
        # 4. 백업 정리 테스트
        print("\n4. 백업 정리...")
        cleaned_count = await manager.cleanup_old_backups()
        print(f"정리된 백업: {cleaned_count}개")

asyncio.run(main())
```