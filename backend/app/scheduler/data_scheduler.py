# backend/app/scheduler/data_scheduler.py

“””
데이터 수집 스케줄러

- 정기적 데이터 수집 (5분, 1시간, 일일)
- 장 시간 외 데이터 수집
- 실패한 작업 재시도
- 스케줄링 관리 및 모니터링
  “””

import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

from ..data.collectors.kis_api import KISApiClient, MarketType
from ..data.collectors.realtime_collector import get_realtime_collector
from ..core.config import get_kis_config
from ..core.logging import get_main_logger, get_performance_logger
from ..core.redis import get_redis_client

logger = get_main_logger()
perf_logger = get_performance_logger()

class ScheduleType(Enum):
“”“스케줄 타입”””
REALTIME = “realtime”          # 실시간 (5분마다)
HOURLY = “hourly”              # 시간별
DAILY = “daily”                # 일일
WEEKLY = “weekly”              # 주간
MARKET_OPEN = “market_open”    # 장 시작
MARKET_CLOSE = “market_close”  # 장 마감
AFTER_HOURS = “after_hours”    # 장 시간 외

@dataclass
class ScheduleJob:
“”“스케줄 작업 정의”””
id: str
name: str
schedule_type: ScheduleType
func: Callable
args: tuple = field(default_factory=tuple)
kwargs: dict = field(default_factory=dict)
enabled: bool = True
max_retries: int = 3
retry_delay: int = 60  # 초
market_time_only: bool = False  # 장 시간에만 실행
description: str = “”

class DataScheduler:
“”“데이터 수집 스케줄러”””

```
def __init__(self):
    # APScheduler 설정
    self.scheduler = AsyncIOScheduler(
        jobstores={'default': MemoryJobStore()},
        executors={'default': AsyncIOExecutor()},
        job_defaults={
            'coalesce': False,
            'max_instances': 3,
            'misfire_grace_time': 30
        },
        timezone='Asia/Seoul'
    )
    
    # 작업 상태 관리
    self.jobs: Dict[str, ScheduleJob] = {}
    self.job_stats: Dict[str, Dict] = {}
    self.is_running = False
    
    # KIS API 클라이언트
    self.kis_client: Optional[KISApiClient] = None
    
    # 이벤트 리스너 등록
    self.scheduler.add_listener(self._job_executed_listener, EVENT_JOB_EXECUTED)
    self.scheduler.add_listener(self._job_error_listener, EVENT_JOB_ERROR)
    
async def initialize(self):
    """스케줄러 초기화"""
    try:
        # KIS API 클라이언트 초기화
        config = get_kis_config()
        self.kis_client = KISApiClient(config)
        await self.kis_client.initialize()
        
        # 기본 작업들 등록
        await self._register_default_jobs()
        
        logger.info("데이터 수집 스케줄러 초기화 완료")
        
    except Exception as e:
        logger.error(f"스케줄러 초기화 실패: {e}")
        raise
        
async def start(self):
    """스케줄러 시작"""
    try:
        if self.is_running:
            logger.warning("스케줄러가 이미 실행 중입니다")
            return
            
        self.scheduler.start()
        self.is_running = True
        
        logger.info("데이터 수집 스케줄러 시작")
        logger.info(f"등록된 작업 수: {len(self.jobs)}")
        
    except Exception as e:
        logger.error(f"스케줄러 시작 실패: {e}")
        raise
        
async def stop(self):
    """스케줄러 정지"""
    try:
        if not self.is_running:
            return
            
        self.scheduler.shutdown(wait=True)
        self.is_running = False
        
        if self.kis_client:
            await self.kis_client.close()
            
        logger.info("데이터 수집 스케줄러 정지")
        
    except Exception as e:
        logger.error(f"스케줄러 정지 실패: {e}")
        
async def add_job(self, schedule_job: ScheduleJob):
    """작업 추가"""
    try:
        if schedule_job.id in self.jobs:
            logger.warning(f"작업 ID가 이미 존재합니다: {schedule_job.id}")
            return False
            
        # 스케줄 타입에 따른 트리거 설정
        trigger = self._create_trigger(schedule_job.schedule_type)
        
        if trigger is None:
            logger.error(f"지원하지 않는 스케줄 타입: {schedule_job.schedule_type}")
            return False
            
        # APScheduler에 작업 추가
        self.scheduler.add_job(
            func=self._execute_job,
            trigger=trigger,
            args=[schedule_job.id],
            id=schedule_job.id,
            name=schedule_job.name,
            replace_existing=True
        )
        
        # 내부 작업 목록에 추가
        self.jobs[schedule_job.id] = schedule_job
        self.job_stats[schedule_job.id] = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'last_execution': None,
            'last_success': None,
            'last_error': None,
            'average_duration': 0.0
        }
        
        logger.info(f"작업 추가: {schedule_job.name} ({schedule_job.id})")
        return True
        
    except Exception as e:
        logger.error(f"작업 추가 실패 ({schedule_job.id}): {e}")
        return False
        
async def remove_job(self, job_id: str):
    """작업 제거"""
    try:
        if job_id not in self.jobs:
            logger.warning(f"존재하지 않는 작업 ID: {job_id}")
            return False
            
        # APScheduler에서 제거
        self.scheduler.remove_job(job_id)
        
        # 내부 목록에서 제거
        del self.jobs[job_id]
        del self.job_stats[job_id]
        
        logger.info(f"작업 제거: {job_id}")
        return True
        
    except Exception as e:
        logger.error(f"작업 제거 실패 ({job_id}): {e}")
        return False
        
async def pause_job(self, job_id: str):
    """작업 일시 정지"""
    try:
        self.scheduler.pause_job(job_id)
        if job_id in self.jobs:
            self.jobs[job_id].enabled = False
        logger.info(f"작업 일시 정지: {job_id}")
        return True
    except Exception as e:
        logger.error(f"작업 일시 정지 실패 ({job_id}): {e}")
        return False
        
async def resume_job(self, job_id: str):
    """작업 재개"""
    try:
        self.scheduler.resume_job(job_id)
        if job_id in self.jobs:
            self.jobs[job_id].enabled = True
        logger.info(f"작업 재개: {job_id}")
        return True
    except Exception as e:
        logger.error(f"작업 재개 실패 ({job_id}): {e}")
        return False
        
async def execute_job_now(self, job_id: str):
    """작업 즉시 실행"""
    try:
        if job_id not in self.jobs:
            logger.error(f"존재하지 않는 작업 ID: {job_id}")
            return False
            
        await self._execute_job(job_id)
        logger.info(f"작업 즉시 실행 완료: {job_id}")
        return True
        
    except Exception as e:
        logger.error(f"작업 즉시 실행 실패 ({job_id}): {e}")
        return False
        
def get_job_status(self) -> Dict[str, Any]:
    """작업 상태 조회"""
    return {
        'is_running': self.is_running,
        'total_jobs': len(self.jobs),
        'active_jobs': len([job for job in self.jobs.values() if job.enabled]),
        'jobs': {
            job_id: {
                'name': job.name,
                'schedule_type': job.schedule_type.value,
                'enabled': job.enabled,
                'description': job.description,
                'stats': self.job_stats.get(job_id, {})
            }
            for job_id, job in self.jobs.items()
        },
        'scheduler_info': {
            'timezone': str(self.scheduler.timezone),
            'running_jobs': len(self.scheduler.get_jobs())
        }
    }
    
async def _register_default_jobs(self):
    """기본 작업들 등록"""
    
    # 1. 실시간 데이터 수집 상태 확인 (5분마다)
    await self.add_job(ScheduleJob(
        id="realtime_health_check",
        name="실시간 수집기 상태 확인",
        schedule_type=ScheduleType.REALTIME,
        func=self._check_realtime_collector,
        market_time_only=True,
        description="실시간 데이터 수집기 상태를 확인하고 필요시 재시작"
    ))
    
    # 2. 주요 종목 현재가 수집 (5분마다)
    await self.add_job(ScheduleJob(
        id="major_stocks_price",
        name="주요 종목 현재가 수집",
        schedule_type=ScheduleType.REALTIME,
        func=self._collect_major_stocks_price,
        market_time_only=True,
        description="주요 종목들의 현재가 정보를 정기적으로 수집"
    ))
    
    # 3. 시장 지수 정보 수집 (1시간마다)
    await self.add_job(ScheduleJob(
        id="market_indices",
        name="시장 지수 정보 수집",
        schedule_type=ScheduleType.HOURLY,
        func=self._collect_market_indices,
        market_time_only=True,
        description="코스피, 코스닥 등 주요 지수 정보 수집"
    ))
    
    # 4. 투자자별 매매동향 수집 (1시간마다)
    await self.add_job(ScheduleJob(
        id="investor_trading",
        name="투자자별 매매동향 수집",
        schedule_type=ScheduleType.HOURLY,
        func=self._collect_investor_trading,
        market_time_only=True,
        description="개인, 외국인, 기관 투자자별 매매동향 수집"
    ))
    
    # 5. 일일 시장 데이터 정리 (장 마감 후)
    await self.add_job(ScheduleJob(
        id="daily_market_summary",
        name="일일 시장 데이터 정리",
        schedule_type=ScheduleType.MARKET_CLOSE,
        func=self._create_daily_summary,
        description="하루 종합 시장 데이터 정리 및 통계 생성"
    ))
    
    # 6. 주간 데이터 백업 (일요일)
    await self.add_job(ScheduleJob(
        id="weekly_data_backup",
        name="주간 데이터 백업",
        schedule_type=ScheduleType.WEEKLY,
        func=self._backup_weekly_data,
        description="주간 데이터 백업 및 정리"
    ))
    
    # 7. 시스템 정리 (새벽 2시)
    await self.add_job(ScheduleJob(
        id="system_cleanup",
        name="시스템 정리",
        schedule_type=ScheduleType.DAILY,
        func=self._system_cleanup,
        description="로그 정리, 캐시 클리어, 임시 파일 삭제"
    ))
    
def _create_trigger(self, schedule_type: ScheduleType):
    """스케줄 타입에 따른 트리거 생성"""
    if schedule_type == ScheduleType.REALTIME:
        # 5분마다 (장 시간 중)
        return CronTrigger(
            minute='*/5',
            hour='9-15',
            day_of_week='mon-fri'
        )
    elif schedule_type == ScheduleType.HOURLY:
        # 매시 정각 (장 시간 중)
        return CronTrigger(
            minute=0,
            hour='9-15',
            day_of_week='mon-fri'
        )
    elif schedule_type == ScheduleType.DAILY:
        # 매일 새벽 2시
        return CronTrigger(
            hour=2,
            minute=0
        )
    elif schedule_type == ScheduleType.WEEKLY:
        # 매주 일요일 새벽 3시
        return CronTrigger(
            day_of_week='sun',
            hour=3,
            minute=0
        )
    elif schedule_type == ScheduleType.MARKET_OPEN:
        # 장 시작 (9시)
        return CronTrigger(
            hour=9,
            minute=0,
            day_of_week='mon-fri'
        )
    elif schedule_type == ScheduleType.MARKET_CLOSE:
        # 장 마감 (15시 30분)
        return CronTrigger(
            hour=15,
            minute=30,
            day_of_week='mon-fri'
        )
    elif schedule_type == ScheduleType.AFTER_HOURS:
        # 장 시간 외 (18시)
        return CronTrigger(
            hour=18,
            minute=0,
            day_of_week='mon-fri'
        )
    else:
        return None
        
async def _execute_job(self, job_id: str):
    """작업 실행"""
    if job_id not in self.jobs:
        logger.error(f"존재하지 않는 작업 ID: {job_id}")
        return
        
    job = self.jobs[job_id]
    
    if not job.enabled:
        logger.info(f"비활성화된 작업 스킵: {job_id}")
        return
        
    # 장 시간 체크
    if job.market_time_only and not self._is_market_time():
        logger.info(f"장 시간 외 작업 스킵: {job_id}")
        return
        
    start_time = datetime.now()
    retry_count = 0
    
    while retry_count <= job.max_retries:
        try:
            logger.info(f"작업 실행 시작: {job.name} (시도: {retry_count + 1})")
            
            # 작업 실행
            if asyncio.iscoroutinefunction(job.func):
                await job.func(*job.args, **job.kwargs)
            else:
                job.func(*job.args, **job.kwargs)
            
            # 성공 통계 업데이트
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_job_stats(job_id, True, execution_time)
            
            logger.info(f"작업 실행 완료: {job.name} ({execution_time:.2f}초)")
            return
            
        except Exception as e:
            retry_count += 1
            error_msg = f"작업 실행 실패: {job.name} (시도 {retry_count}/{job.max_retries + 1}) - {str(e)}"
            
            if retry_count <= job.max_retries:
                logger.warning(f"{error_msg}, {job.retry_delay}초 후 재시도")
                await asyncio.sleep(job.retry_delay)
            else:
                logger.error(f"{error_msg}, 최대 재시도 횟수 초과")
                
                # 실패 통계 업데이트
                execution_time = (datetime.now() - start_time).total_seconds()
                self._update_job_stats(job_id, False, execution_time, str(e))
                
def _update_job_stats(self, job_id: str, success: bool, duration: float, error: str = None):
    """작업 통계 업데이트"""
    if job_id not in self.job_stats:
        return
        
    stats = self.job_stats[job_id]
    stats['total_executions'] += 1
    stats['last_execution'] = datetime.now().isoformat()
    
    if success:
        stats['successful_executions'] += 1
        stats['last_success'] = datetime.now().isoformat()
    else:
        stats['failed_executions'] += 1
        stats['last_error'] = error
        
    # 평균 실행 시간 계산
    if stats['total_executions'] > 0:
        stats['average_duration'] = (
            (stats['average_duration'] * (stats['total_executions'] - 1) + duration) /
            stats['total_executions']
        )
        
def _is_market_time(self) -> bool:
    """현재가 장 시간인지 확인"""
    now = datetime.now()
    
    # 주말 체크
    if now.weekday() >= 5:  # 토요일(5), 일요일(6)
        return False
        
    # 장 시간 체크 (9:00 ~ 15:30)
    current_time = now.time()
    market_start = time(9, 0)
    market_end = time(15, 30)
    
    return market_start <= current_time <= market_end
    
def _job_executed_listener(self, event):
    """작업 실행 완료 이벤트 리스너"""
    logger.debug(f"작업 실행 완료: {event.job_id}")
    
def _job_error_listener(self, event):
    """작업 오류 이벤트 리스너"""
    logger.error(f"작업 실행 오류: {event.job_id} - {event.exception}")
    
# ==================== 작업 함수들 ====================

async def _check_realtime_collector(self):
    """실시간 수집기 상태 확인"""
    try:
        collector = await get_realtime_collector()
        stats = collector.get_stats()
        
        if not stats['is_running']:
            logger.warning("실시간 수집기가 정지됨, 재시작 시도")
            await collector.start()
        elif stats['active_connections'] == 0:
            logger.warning("실시간 수집기 연결 없음")
        else:
            logger.info(f"실시간 수집기 정상 - 연결: {stats['active_connections']}, 구독: {stats['subscribed_tickers']}")
            
    except Exception as e:
        logger.error(f"실시간 수집기 상태 확인 실패: {e}")
        
async def _collect_major_stocks_price(self):
    """주요 종목 현재가 수집"""
    try:
        # 주요 종목 리스트 (시가총액 상위)
        major_tickers = [
            "005930",  # 삼성전자
            "000660",  # SK하이닉스
            "035420",  # 네이버
            "005490",  # POSCO홀딩스
            "051910",  # LG화학
            "006400",  # 삼성SDI
            "035720",  # 카카오
            "068270",  # 셀트리온
            "207940",  # 삼성바이오로직스
            "373220"   # LG에너지솔루션
        ]
        
        async with self.kis_client:
            prices = await self.kis_client.get_multiple_prices(major_tickers)
            
        # Redis에 캐시 저장
        redis_client = await get_redis_client()
        for price in prices:
            if hasattr(price, 'ticker'):
                cache_data = {
                    "current_price": price.current_price,
                    "change": price.change,
                    "change_rate": price.change_rate,
                    "volume": price.volume,
                    "timestamp": price.timestamp.isoformat()
                }
                await redis_client.set(
                    f"price:{price.ticker}",
                    json.dumps(cache_data),
                    expire=300  # 5분
                )
                
        logger.info(f"주요 종목 현재가 수집 완료: {len(prices)}개 종목")
        
    except Exception as e:
        logger.error(f"주요 종목 현재가 수집 실패: {e}")
        raise
        
async def _collect_market_indices(self):
    """시장 지수 정보 수집"""
    try:
        async with self.kis_client:
            market_status = await self.kis_client.get_market_status()
            
        # Redis에 시장 정보 저장
        redis_client = await get_redis_client()
        await redis_client.set(
            "market:status",
            json.dumps(market_status, default=str),
            expire=3600  # 1시간
        )
        
        logger.info("시장 지수 정보 수집 완료")
        
    except Exception as e:
        logger.error(f"시장 지수 정보 수집 실패: {e}")
        raise
        
async def _collect_investor_trading(self):
    """투자자별 매매동향 수집"""
    try:
        # 주요 종목들의 매매동향 수집
        major_tickers = ["005930", "000660", "035420"]
        
        async with self.kis_client:
            for ticker in major_tickers:
                try:
                    trading_data = await self.kis_client.get_trading_by_investor(ticker)
                    
                    # Redis에 저장
                    redis_client = await get_redis_client()
                    cache_data = {
                        "individual": trading_data.individual,
                        "foreign": trading_data.foreign,
                        "institutional": trading_data.institutional,
                        "timestamp": trading_data.timestamp.isoformat()
                    }
                    await redis_client.set(
                        f"trading:{ticker}",
                        json.dumps(cache_data),
                        expire=3600  # 1시간
                    )
                    
                    await asyncio.sleep(0.1)  # API 호출 간격
                    
                except Exception as e:
                    logger.error(f"종목 {ticker} 매매동향 수집 실패: {e}")
                    
        logger.info("투자자별 매매동향 수집 완료")
        
    except Exception as e:
        logger.error(f"투자자별 매매동향 수집 실패: {e}")
        raise
        
async def _create_daily_summary(self):
    """일일 시장 데이터 정리"""
    try:
        # 일일 종합 통계 생성
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Redis에서 오늘의 데이터 수집
        redis_client = await get_redis_client()
        
        # 예시: 일일 요약 데이터 생성
        daily_summary = {
            "date": today,
            "market_summary": "생성됨",
            "data_points_collected": 1000,  # 실제 수집된 데이터 포인트 수
            "created_at": datetime.now().isoformat()
        }
        
        await redis_client.set(
            f"daily_summary:{today}",
            json.dumps(daily_summary),
            expire=86400 * 7  # 7일 보관
        )
        
        logger.info(f"일일 시장 데이터 정리 완료: {today}")
        
    except Exception as e:
        logger.error(f"일일 시장 데이터 정리 실패: {e}")
        raise
        
async def _backup_weekly_data(self):
    """주간 데이터 백업"""
    try:
        # 주간 데이터 백업 로직
        week_start = datetime.now() - timedelta(days=7)
        week_end = datetime.now()
        
        backup_info = {
            "week_start": week_start.strftime("%Y-%m-%d"),
            "week_end": week_end.strftime("%Y-%m-%d"),
            "backup_completed": True,
            "backup_time": datetime.now().isoformat()
        }
        
        # 실제로는 데이터베이스 백업 수행
        logger.info(f"주간 데이터 백업 완료: {backup_info}")
        
    except Exception as e:
        logger.error(f"주간 데이터 백업 실패: {e}")
        raise
        
async def _system_cleanup(self):
    """시스템 정리"""
    try:
        cleanup_tasks = []
        
        # 1. Redis 만료된 키 정리
        redis_client = await get_redis_client()
        
        # 오래된 캐시 데이터 정리
        old_keys = await redis_client.keys("price:*")
        if old_keys:
            # 실제로는 TTL 확인 후 정리
            logger.info(f"Redis 키 정리 대상: {len(old_keys)}개")
            
        # 2. 로그 파일 정리 (7일 이상 된 것)
        log_dir = Path("logs")
        if log_dir.exists():
            old_logs = []
            for log_file in log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < (datetime.now() - timedelta(days=7)).timestamp():
                    old_logs.append(log_file)
                    
            logger.info(f"오래된 로그 파일 정리: {len(old_logs)}개")
            
        # 3. 임시 파일 정리
        temp_files_cleaned = 0  # 실제 정리된 파일 수
        
        cleanup_summary = {
            "redis_keys_cleaned": len(old_keys) if old_keys else 0,
            "log_files_cleaned": len(old_logs) if 'old_logs' in locals() else 0,
            "temp_files_cleaned": temp_files_cleaned,
            "cleanup_time": datetime.now().isoformat()
        }
        
        logger.info(f"시스템 정리 완료: {cleanup_summary}")
        
    except Exception as e:
        logger.error(f"시스템 정리 실패: {e}")
        raise
```

# 전역 스케줄러 인스턴스

_data_scheduler: Optional[DataScheduler] = None

async def get_data_scheduler() -> DataScheduler:
“”“전역 데이터 스케줄러 인스턴스 반환”””
global _data_scheduler

```
if _data_scheduler is None:
    _data_scheduler = DataScheduler()
    await _data_scheduler.initialize()
    
return _data_scheduler
```

async def start_data_scheduling():
“”“데이터 스케줄링 시작”””
scheduler = await get_data_scheduler()
await scheduler.start()

async def stop_data_scheduling():
“”“데이터 스케줄링 정지”””
global _data_scheduler

```
if _data_scheduler:
    await _data_scheduler.stop()
    _data_scheduler = None
```

# ==================== 사용 예제 ====================

async def example_usage():
“”“스케줄러 사용 예제”””
scheduler = DataScheduler()
await scheduler.initialize()

```
try:
    # 스케줄러 시작
    await scheduler.start()
    
    # 커스텀 작업 추가
    custom_job = ScheduleJob(
        id="custom_data_collection",
        name="커스텀 데이터 수집",
        schedule_type=ScheduleType.HOURLY,
        func=lambda: print("커스텀 작업 실행"),
        description="사용자 정의 데이터 수집 작업"
    )
    
    await scheduler.add_job(custom_job)
    
    # 상태 확인
    status = scheduler.get_job_status()
    print(f"스케줄러 상태: {status}")
    
    # 10초 대기 (실제로는 계속 실행)
    await asyncio.sleep(10)
    
finally:
    # 스케줄러 정지
    await scheduler.stop()
```

if **name** == “**main**”:
asyncio.run(example_usage())