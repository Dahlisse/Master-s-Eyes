# backend/app/data/collectors/news_crawler.py

“””
네이버 금융 뉴스 크롤링 시스템
실시간 뉴스 수집 및 종목별 뉴스 분석
“””
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import logging
from dataclasses import dataclass
import json
import re
import urllib.parse
from urllib.robotparser import RobotFileParser

from app.core.config import get_settings
from app.core.redis import get_redis_client
from app.data.processors.data_validator import DataValidator

settings = get_settings()
logger = logging.getLogger(**name**)

@dataclass
class NewsArticle:
“”“뉴스 기사 데이터 구조”””
title: str
content: str
url: str
published_time: datetime
source: str
category: str
stock_codes: List[str]  # 관련 종목 코드들
sentiment_score: Optional[float] = None
importance_score: Optional[float] = None
keywords: List[str] = None

@dataclass
class MarketNews:
“”“시장 뉴스 요약”””
date: datetime
total_articles: int
positive_sentiment: float
negative_sentiment: float
neutral_sentiment: float
top_keywords: List[str]
major_events: List[str]
affected_stocks: Dict[str, int]  # 종목별 언급 횟수

class NaverFinanceNewsCrawler:
“”“네이버 금융 뉴스 크롤러”””

```
def __init__(self):
    self.base_url = "https://finance.naver.com"
    self.redis_client = get_redis_client()
    self.validator = DataValidator()
    self.session = None
    
    # 크롤링 설정
    self.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    # 뉴스 카테고리별 URL
    self.news_categories = {
        'market': '/news/mainnews.naver?mode=LSS2D&mid=A05&sid1=101&sid2=258',  # 증시
        'industry': '/news/mainnews.naver?mode=LSS2D&mid=A05&sid1=101&sid2=261',  # 산업/재계
        'economy': '/news/mainnews.naver?mode=LSS2D&mid=A05&sid1=101&sid2=259',  # 경제일반
        'global': '/news/mainnews.naver?mode=LSS2D&mid=A05&sid1=101&sid2=262',  # 글로벌경제
        'finance': '/news/mainnews.naver?mode=LSS2D&mid=A05&sid1=101&sid2=771',  # 금융
        'real_estate': '/news/mainnews.naver?mode=LSS2D&mid=A05&sid1=101&sid2=260'  # 부동산
    }
    
    # 주요 종목 코드 및 키워드 매핑
    self.stock_keywords = {
        # 대형주
        '005930': ['삼성전자', '삼성', 'SK하이닉스와 함께', '반도체', '메모리'],
        '000660': ['SK하이닉스', 'SK하이', '하이닉스', '메모리반도체'],
        '035420': ['NAVER', '네이버', '라인', '웹툰', '클라우드'],
        '005380': ['현대차', '현대자동차', '아이오닉', '전기차'],
        '051910': ['LG화학', 'LG배터리', 'LG에너지솔루션'],
        '006400': ['삼성SDI', '배터리', '전기차배터리'],
        '207940': ['삼성바이오로직스', '바이오', 'CMO'],
        '035720': ['카카오', '카톡', '카카오페이', '카카오뱅크'],
        '003670': ['포스코', 'POSCO', '철강', '2차전지소재'],
        '068270': ['셀트리온', '바이오시밀러', '항체의약품'],
        
        # 중형주 
        '012330': ['현대모비스', '모비스', '자동차부품'],
        '011200': ['HMM', '해운', '컨테이너'],
        '028260': ['삼성물산', '건설', '상사'],
        '009540': ['HD한국조선해양', '조선', '해양플랜트'],
        '015760': ['한국전력', '한전', '전력', '원전'],
        
        # 테마주
        '091990': ['셀트리온헬스케어', '바이오'],
        '326030': ['SK바이오팜', '신약', 'CNS'],
        '377300': ['카카오페이', '핀테크', '간편결제'],
        '251270': ['넷마블', '게임', '모바일게임'],
        '352820': ['하이브', 'BTS', '엔터테인먼트']
    }
    
    # 중요 키워드 가중치
    self.keyword_weights = {
        # 정책/규제
        '금리': 0.9, '기준금리': 0.9, '통화정책': 0.8,
        '양적완화': 0.8, 'QE': 0.8,
        '증시': 0.7, '코스피': 0.8, '코스닥': 0.7,
        
        # 실적 관련
        '실적': 0.8, '영업이익': 0.7, '매출': 0.6,
        '어닝쇼크': 0.9, '어닝서프라이즈': 0.8,
        '배당': 0.6, '배당수익률': 0.6,
        
        # 산업 트렌드
        '반도체': 0.8, '메모리': 0.7, 'AI': 0.8,
        '전기차': 0.8, '배터리': 0.7, '2차전지': 0.7,
        '바이오': 0.7, '신약': 0.8,
        '메타버스': 0.6, 'NFT': 0.5,
        
        # 리스크 요인
        '인플레이션': 0.9, '물가': 0.8,
        '공급망': 0.7, '원자재': 0.7,
        '지정학적': 0.8, '우크라이나': 0.7,
        '중국': 0.8, '미중갈등': 0.8,
        
        # 긍정 키워드
        '상승': 0.6, '급등': 0.8, '강세': 0.7,
        '호재': 0.7, '개선': 0.6, '성장': 0.6,
        
        # 부정 키워드  
        '하락': 0.6, '급락': 0.8, '약세': 0.7,
        '악재': 0.7, '우려': 0.6, '위험': 0.7
    }

async def __aenter__(self):
    """비동기 컨텍스트 매니저 진입"""
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    self.session = aiohttp.ClientSession(
        headers=self.headers,
        connector=connector,
        timeout=timeout
    )
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    """비동기 컨텍스트 매니저 종료"""
    if self.session:
        await self.session.close()

async def crawl_category_news(
    self, 
    category: str, 
    max_pages: int = 3,
    hours_back: int = 24
) -> List[NewsArticle]:
    """카테고리별 뉴스 크롤링"""
    try:
        if category not in self.news_categories:
            logger.error(f"Unknown category: {category}")
            return []
        
        logger.info(f"Crawling {category} news, max_pages: {max_pages}")
        
        articles = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for page in range(1, max_pages + 1):
            try:
                # 페이지별 URL 구성
                page_url = f"{self.base_url}{self.news_categories[category]}&page={page}"
                
                async with self.session.get(page_url) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch page {page} for {category}: {response.status}")
                        continue
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 뉴스 목록 파싱
                    news_items = soup.select('.newsList .item')
                    
                    if not news_items:
                        logger.warning(f"No news items found on page {page} for {category}")
                        break
                    
                    page_articles = []
                    for item in news_items:
                        try:
                            article = await self._parse_news_item(item, category)
                            if article and article.published_time >= cutoff_time:
                                page_articles.append(article)
                            elif article and article.published_time < cutoff_time:
                                # 시간 cutoff에 도달하면 크롤링 중단
                                logger.info(f"Reached time cutoff at page {page}")
                                articles.extend(page_articles)
                                return articles
                        except Exception as e:
                            logger.error(f"Error parsing news item: {e}")
                            continue
                    
                    articles.extend(page_articles)
                    
                    # 요청 간 지연 (서버 부하 방지)
                    await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error crawling page {page} for {category}: {e}")
                continue
        
        logger.info(f"Crawled {len(articles)} articles from {category}")
        return articles
        
    except Exception as e:
        logger.error(f"Error in crawl_category_news for {category}: {e}")
        return []

async def _parse_news_item(self, item, category: str) -> Optional[NewsArticle]:
    """개별 뉴스 아이템 파싱"""
    try:
        # 제목과 링크 추출
        title_elem = item.select_one('.title a')
        if not title_elem:
            return None
        
        title = title_elem.get_text(strip=True)
        news_url = title_elem.get('href')
        
        if not news_url.startswith('http'):
            news_url = self.base_url + news_url
        
        # 발행시간 추출
        time_elem = item.select_one('.date')
        if not time_elem:
            return None
        
        time_text = time_elem.get_text(strip=True)
        published_time = self._parse_time(time_text)
        
        # 언론사 추출
        source_elem = item.select_one('.press')
        source = source_elem.get_text(strip=True) if source_elem else "Unknown"
        
        # 본문 내용 가져오기 (별도 요청)
        content = await self._fetch_article_content(news_url)
        
        # 관련 종목 코드 추출
        stock_codes = self._extract_stock_codes(title + " " + content)
        
        # 키워드 추출
        keywords = self._extract_keywords(title + " " + content)
        
        # 중요도 점수 계산
        importance_score = self._calculate_importance_score(title, content, keywords)
        
        article = NewsArticle(
            title=title,
            content=content,
            url=news_url,
            published_time=published_time,
            source=source,
            category=category,
            stock_codes=stock_codes,
            keywords=keywords,
            importance_score=importance_score
        )
        
        return article
        
    except Exception as e:
        logger.error(f"Error parsing news item: {e}")
        return None

async def _fetch_article_content(self, url: str) -> str:
    """기사 본문 내용 가져오기"""
    try:
        # 캐시 확인
        cache_key = f"news_content:{hash(url)}"
        cached_content = await self.redis_client.get(cache_key)
        if cached_content:
            return cached_content.decode('utf-8')
        
        # 본문 페이지 요청
        async with self.session.get(url) as response:
            if response.status != 200:
                return ""
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # 네이버 뉴스 본문 파싱
            content_selectors = [
                '.newsct_article ._article_body_contents',  # 일반 뉴스
                '.news_end .end_body_wrp',  # 일부 뉴스
                '#newsct_article',  # 대안 선택자
                '.article_body'  # 다른 형태
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # 광고, 스크립트 등 제거
                    for unwanted in content_elem.select('script, style, .ad, .advertisement'):
                        unwanted.decompose()
                    
                    content = content_elem.get_text(strip=True, separator=' ')
                    break
            
            # 내용이 너무 짧으면 제목에서 대체
            if len(content) < 50:
                title_elem = soup.select_one('h1, .news_title, .title')
                if title_elem:
                    content = title_elem.get_text(strip=True)
            
            # 캐시 저장 (1시간)
            if content:
                await self.redis_client.setex(cache_key, 3600, content.encode('utf-8'))
            
            return content[:5000]  # 최대 5000자로 제한
            
    except Exception as e:
        logger.error(f"Error fetching article content from {url}: {e}")
        return ""

def _parse_time(self, time_text: str) -> datetime:
    """시간 문자열 파싱"""
    try:
        now = datetime.now()
        
        # "N분 전", "N시간 전" 형태
        if '분 전' in time_text:
            minutes = int(re.search(r'(\d+)분 전', time_text).group(1))
            return now - timedelta(minutes=minutes)
        elif '시간 전' in time_text:
            hours = int(re.search(r'(\d+)시간 전', time_text).group(1))
            return now - timedelta(hours=hours)
        elif '일 전' in time_text:
            days = int(re.search(r'(\d+)일 전', time_text).group(1))
            return now - timedelta(days=days)
        
        # "MM.DD HH:mm" 형태
        elif re.match(r'\d{2}\.\d{2} \d{2}:\d{2}', time_text):
            month_day, time_part = time_text.split(' ')
            month, day = month_day.split('.')
            hour, minute = time_part.split(':')
            
            return datetime(now.year, int(month), int(day), int(hour), int(minute))
        
        # "YYYY.MM.DD HH:mm" 형태
        elif re.match(r'\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}', time_text):
            date_part, time_part = time_text.split(' ')
            year, month, day = date_part.split('.')
            hour, minute = time_part.split(':')
            
            return datetime(int(year), int(month), int(day), int(hour), int(minute))
        
        else:
            # 파싱 실패시 현재 시간
            return now
            
    except Exception as e:
        logger.error(f"Error parsing time '{time_text}': {e}")
        return datetime.now()

def _extract_stock_codes(self, text: str) -> List[str]:
    """텍스트에서 관련 종목 코드 추출"""
    found_codes = []
    
    for code, keywords in self.stock_keywords.items():
        for keyword in keywords:
            if keyword in text:
                found_codes.append(code)
                break
    
    return list(set(found_codes))  # 중복 제거

def _extract_keywords(self, text: str) -> List[str]:
    """중요 키워드 추출"""
    found_keywords = []
    
    for keyword in self.keyword_weights.keys():
        if keyword in text:
            found_keywords.append(keyword)
    
    return found_keywords

def _calculate_importance_score(self, title: str, content: str, keywords: List[str]) -> float:
    """뉴스 중요도 점수 계산 (0-1)"""
    score = 0.0
    
    # 키워드 기반 점수
    for keyword in keywords:
        weight = self.keyword_weights.get(keyword, 0.1)
        # 제목에 있으면 가중치 2배
        if keyword in title:
            score += weight * 2
        else:
            score += weight
    
    # 텍스트 길이 보정 (너무 짧거나 긴 기사는 감점)
    content_length = len(content)
    if 100 <= content_length <= 2000:
        length_bonus = 1.0
    elif content_length < 100:
        length_bonus = 0.5
    else:
        length_bonus = 0.8
    
    score *= length_bonus
    
    # 0-1 범위로 정규화
    return min(1.0, score / 5.0)

async def crawl_all_categories(self, hours_back: int = 24) -> Dict[str, List[NewsArticle]]:
    """모든 카테고리 뉴스 크롤링"""
    try:
        logger.info(f"Crawling all categories, hours_back: {hours_back}")
        
        all_news = {}
        
        # 카테고리별 병렬 크롤링 (동시 실행 제한)
        semaphore = asyncio.Semaphore(3)  # 동시 3개 카테고리
        
        async def crawl_single_category(category: str):
            async with semaphore:
                articles = await self.crawl_category_news(category, max_pages=5, hours_back=hours_back)
                all_news[category] = articles
                await asyncio.sleep(2)  # 카테고리 간 지연
        
        # 모든 카테고리 병렬 실행
        tasks = [
            crawl_single_category(category)
            for category in self.news_categories.keys()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 통계 로깅
        total_articles = sum(len(articles) for articles in all_news.values())
        logger.info(f"Total crawled articles: {total_articles}")
        
        for category, articles in all_news.items():
            logger.info(f"{category}: {len(articles)} articles")
        
        return all_news
        
    except Exception as e:
        logger.error(f"Error in crawl_all_categories: {e}")
        return {}

async def get_stock_related_news(self, stock_code: str, hours_back: int = 24) -> List[NewsArticle]:
    """특정 종목 관련 뉴스 수집"""
    try:
        # 캐시 확인
        cache_key = f"stock_news:{stock_code}:{hours_back}"
        cached_news = await self.redis_client.get(cache_key)
        if cached_news:
            cached_data = json.loads(cached_news)
            return [
                NewsArticle(**article_data) 
                for article_data in cached_data
            ]
        
        # 모든 뉴스에서 해당 종목 관련 뉴스 필터링
        all_news = await self.crawl_all_categories(hours_back)
        
        stock_news = []
        for category_news in all_news.values():
            for article in category_news:
                if stock_code in article.stock_codes:
                    stock_news.append(article)
        
        # 중요도 순 정렬
        stock_news.sort(key=lambda x: x.importance_score or 0, reverse=True)
        
        # 캐시 저장 (30분)
        cache_data = [
            {
                'title': article.title,
                'content': article.content,
                'url': article.url,
                'published_time': article.published_time.isoformat(),
                'source': article.source,
                'category': article.category,
                'stock_codes': article.stock_codes,
                'keywords': article.keywords,
                'importance_score': article.importance_score
            }
            for article in stock_news
        ]
        
        await self.redis_client.setex(cache_key, 1800, json.dumps(cache_data))
        
        logger.info(f"Found {len(stock_news)} articles for stock {stock_code}")
        return stock_news
        
    except Exception as e:
        logger.error(f"Error getting stock news for {stock_code}: {e}")
        return []

async def analyze_market_sentiment(self, hours_back: int = 24) -> MarketNews:
    """시장 전체 뉴스 심리 분석"""
    try:
        all_news = await self.crawl_all_categories(hours_back)
        
        all_articles = []
        for category_news in all_news.values():
            all_articles.extend(category_news)
        
        if not all_articles:
            return self._empty_market_news()
        
        # 키워드 빈도 분석
        keyword_counter = {}
        stock_mentions = {}
        
        for article in all_articles:
            # 키워드 카운트
            for keyword in article.keywords or []:
                keyword_counter[keyword] = keyword_counter.get(keyword, 0) + 1
            
            # 종목 언급 카운트
            for stock_code in article.stock_codes:
                stock_mentions[stock_code] = stock_mentions.get(stock_code, 0) + 1
        
        # 상위 키워드 추출
        top_keywords = sorted(keyword_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        top_keywords = [keyword for keyword, count in top_keywords]
        
        # 감성 분석 (간단한 키워드 기반)
        positive_count = sum(1 for article in all_articles 
                           if any(kw in ['상승', '급등', '강세', '호재', '개선', '성장'] 
                                 for kw in (article.keywords or [])))
        
        negative_count = sum(1 for article in all_articles 
                           if any(kw in ['하락', '급락', '약세', '악재', '우려', '위험'] 
                                 for kw in (article.keywords or [])))
        
        neutral_count = len(all_articles) - positive_count - negative_count
        
        total = len(all_articles)
        positive_sentiment = positive_count / total if total > 0 else 0
        negative_sentiment = negative_count / total if total > 0 else 0
        neutral_sentiment = neutral_count / total if total > 0 else 0
        
        # 주요 이벤트 추출 (중요도 높은 뉴스)
        major_events = [
            article.title for article in 
            sorted(all_articles, key=lambda x: x.importance_score or 0, reverse=True)[:5]
        ]
        
        market_news = MarketNews(
            date=datetime.now(),
            total_articles=total,
            positive_sentiment=positive_sentiment,
            negative_sentiment=negative_sentiment,
            neutral_sentiment=neutral_sentiment,
            top_keywords=top_keywords,
            major_events=major_events,
            affected_stocks=stock_mentions
        )
        
        # 결과 캐시
        await self._cache_market_news(market_news)
        
        return market_news
        
    except Exception as e:
        logger.error(f"Error in analyze_market_sentiment: {e}")
        return self._empty_market_news()

def _empty_market_news(self) -> MarketNews:
    """빈 시장 뉴스 객체"""
    return MarketNews(
        date=datetime.now(),
        total_articles=0,
        positive_sentiment=0.0,
        negative_sentiment=0.0,
        neutral_sentiment=0.0,
        top_keywords=[],
        major_events=[],
        affected_stocks={}
    )

async def _cache_market_news(self, market_news: MarketNews):
    """시장 뉴스 캐시"""
    try:
        cache_data = {
            'date': market_news.date.isoformat(),
            'total_articles': market_news.total_articles,
            'positive_sentiment': market_news.positive_sentiment,
            'negative_sentiment': market_news.negative_sentiment,
            'neutral_sentiment': market_news.neutral_sentiment,
            'top_keywords': market_news.top_keywords,
            'major_events': market_news.major_events,
            'affected_stocks': market_news.affected_stocks
        }
        
        await self.redis_client.setex(
            'market_news_analysis',
            1800,  # 30분 TTL
            json.dumps(cache_data)
        )
        
    except Exception as e:
        logger.error(f"Error caching market news: {e}")
```

# 비동기 함수들 (스케줄러/API용)

async def collect_all_financial_news(hours_back: int = 24) -> Dict[str, Any]:
“”“모든 금융 뉴스 수집”””
async with NaverFinanceNewsCrawler() as crawler:
all_news = await crawler.crawl_all_categories(hours_back)
market_sentiment = await crawler.analyze_market_sentiment(hours_back)

```
    return {
        'news_by_category': {
            category: [article.__dict__ for article in articles]
            for category, articles in all_news.items()
        },
        'market_sentiment': market_sentiment.__dict__,
        'timestamp': datetime.now().isoformat()
    }
```

async def get_stock_news_analysis(stock_code: str, hours_back: int = 24) -> Dict[str, Any]:
“”“특정 종목 뉴스 분석”””
async with NaverFinanceNewsCrawler() as crawler:
stock_news = await crawler.get_stock_related_news(stock_code, hours_back)

```
    return {
        'stock_code': stock_code,
        'articles': [article.__dict__ for article in stock_news],
        'total_articles': len(stock_news),
        'avg_importance': sum(article.importance_score or 0 for article in stock_news) / len(stock_news) if stock_news else 0,
        'timestamp': datetime.now().isoformat()
    }
```

# 동기 래퍼 함수들 (Celery 태스크용)

def sync_collect_financial_news(hours_back: int = 24):
“”“동기 방식 금융 뉴스 수집”””
return asyncio.run(collect_all_financial_news(hours_back))

def sync_get_stock_news(stock_code: str, hours_back: int = 24):
“”“동기 방식 종목 뉴스 수집”””
return asyncio.run(get_stock_news_analysis(stock_code, hours_back))

if **name** == “**main**”:
# 테스트 실행
async def main():
async with NaverFinanceNewsCrawler() as crawler:
print(”=== 네이버 금융 뉴스 크롤링 테스트 ===”)

```
        # 증시 뉴스만 테스트 (1페이지)
        market_news = await crawler.crawl_category_news('market', max_pages=1, hours_back=12)
        
        print(f"수집된 증시 뉴스: {len(market_news)}개")
        
        for i, article in enumerate(market_news[:3]):  # 상위 3개만 출력
            print(f"\n[{i+1}] {article.title}")
            print(f"    발행시간: {article.published_time}")
            print(f"    언론사: {article.source}")
            print(f"    관련종목: {article.stock_codes}")
            print(f"    키워드: {article.keywords}")
            print(f"    중요도: {article.importance_score:.2f}")
            print(f"    내용 미리보기: {article.content[:100]}...")
        
        # 시장 심리 분석 테스트
        print("\n=== 시장 심리 분석 ===")
        sentiment = await crawler.analyze_market_sentiment(hours_back=12)
        print(f"전체 기사: {sentiment.total_articles}개")
        print(f"긍정: {sentiment.positive_sentiment:.1%}")
        print(f"부정: {sentiment.negative_sentiment:.1%}")
        print(f"중립: {sentiment.neutral_sentiment:.1%}")
        print(f"주요 키워드: {sentiment.top_keywords[:5]}")
        
        # 삼성전자 관련 뉴스 테스트
        print("\n=== 삼성전자 관련 뉴스 ===")
        samsung_news = await crawler.get_stock_related_news('005930', hours_back=24)
        print(f"삼성전자 관련 뉴스: {len(samsung_news)}개")
        
        for article in samsung_news[:2]:  # 상위 2개만 출력
            print(f"- {article.title} (중요도: {article.importance_score:.2f})")

asyncio.run(main())
```