# backend/app/data/processors/sentiment_analyzer.py

“””
감성분석 모듈 (로컬 모델)

- 한국어 금융 특화 감성분석
- 뉴스, 리포트, 소셜미디어 텍스트 분석
- 로컬 모델 사용 (비용 절약 + 프라이버시)
- 실시간 시장 심리 분석
  “””
  import asyncio
  import re
  import pandas as pd
  import numpy as np
  from datetime import datetime, timedelta
  from typing import Dict, List, Optional, Any, Tuple, Union
  import logging
  from dataclasses import dataclass
  import json
  import pickle
  import os
  from pathlib import Path
  import joblib

# 로컬 NLP 라이브러리들

try:
from transformers import (
AutoTokenizer, AutoModelForSequenceClassification,
pipeline, Pipeline
)
from sentence_transformers import SentenceTransformer
import torch
TRANSFORMERS_AVAILABLE = True
except ImportError:
TRANSFORMERS_AVAILABLE = False
logger.warning(“Transformers not available, using basic sentiment analysis”)

# 한국어 처리

try:
from konlpy.tag import Okt, Mecab
from soynlp.normalizer import repeat_normalize
KONLPY_AVAILABLE = True
except ImportError:
KONLPY_AVAILABLE = False
logger.warning(“KoNLPy not available, using basic text processing”)

from app.core.config import get_settings
from app.core.redis import get_redis_client
from app.data.processors.data_validator import DataValidator

settings = get_settings()
logger = logging.getLogger(**name**)

@dataclass
class SentimentResult:
“”“감성분석 결과”””
text: str
sentiment: str  # ‘positive’, ‘negative’, ‘neutral’
confidence: float  # 0-1
scores: Dict[str, float]  # {‘positive’: 0.1, ‘negative’: 0.8, ‘neutral’: 0.1}
keywords: List[str]
market_impact: str  # ‘bullish’, ‘bearish’, ‘neutral’
impact_score: float  # -1 to 1
processed_at: datetime

@dataclass
class MarketSentiment:
“”“시장 전체 감성”””
period: str
total_texts: int
overall_sentiment: str
sentiment_distribution: Dict[str, float]
market_mood: str  # ‘fear’, ‘greed’, ‘neutral’, ‘uncertainty’
volatility_indicator: float  # 0-1 (높을수록 변동성 클 것으로 예상)
key_themes: List[str]
bullish_signals: List[str]
bearish_signals: List[str]
confidence_level: float
timestamp: datetime

class KoreanFinancialSentimentAnalyzer:
“”“한국어 금융 특화 감성분석기”””

```
def __init__(self):
    self.redis_client = get_redis_client()
    self.validator = DataValidator()
    
    # 모델 설정
    self.model_config = {
        'model_dir': Path(settings.MODEL_DIR if hasattr(settings, 'MODEL_DIR') else './models'),
        'cache_dir': Path('./cache/sentiment'),
        'batch_size': 32,
        'max_length': 512,
        'confidence_threshold': 0.6
    }
    
    # 디렉토리 생성
    self.model_config['model_dir'].mkdir(parents=True, exist_ok=True)
    self.model_config['cache_dir'].mkdir(parents=True, exist_ok=True)
    
    # 모델 초기화
    self.tokenizer = None
    self.sentiment_model = None
    self.embedding_model = None
    self.korean_analyzer = None
    
    # 금융 특화 사전
    self.financial_lexicon = {
        # 긍정 키워드 (bullish)
        'positive': {
            '상승': 0.8, '급등': 0.9, '강세': 0.8, '상한가': 1.0,
            '호재': 0.8, '개선': 0.7, '성장': 0.7, '증가': 0.6,
            '확대': 0.6, '회복': 0.7, '반등': 0.8, '돌파': 0.8,
            '신고가': 0.9, '최고치': 0.8, '플러스': 0.6,
            '매수': 0.7, '투자': 0.5, '기대': 0.6, '전망': 0.5,
            '긍정': 0.7, '낙관': 0.8, '호황': 0.8, '부양': 0.7,
            '실적개선': 0.8, '수익증가': 0.8, '배당': 0.6
        },
        
        # 부정 키워드 (bearish) 
        'negative': {
            '하락': 0.8, '급락': 0.9, '약세': 0.8, '하한가': 1.0,
            '악재': 0.8, '악화': 0.7, '감소': 0.6, '축소': 0.6,
            '우려': 0.7, '위험': 0.8, '위축': 0.7, '침체': 0.8,
            '폭락': 0.9, '최저치': 0.8, '마이너스': 0.6,
            '매도': 0.7, '손실': 0.8, '적자': 0.8, '부진': 0.7,
            '부정': 0.7, '비관': 0.8, '불황': 0.9, '경기침체': 0.9,
            '실적악화': 0.8, '수익감소': 0.8, '위기': 0.9
        },
        
        # 중립 키워드
        'neutral': {
            '보합': 0.0, '횡보': 0.0, '관망': 0.0, '대기': 0.0,
            '유지': 0.0, '현상': 0.0, '안정': 0.0, '균형': 0.0,
            '분석': 0.0, '검토': 0.0, '모니터링': 0.0, '추이': 0.0
        }
    }
    
    # 시장 영향도 키워드
    self.market_impact_keywords = {
        'high_impact': [
            '금리', '기준금리', '연준', '한국은행', 'Fed', 'FOMC',
            '인플레이션', '물가', 'GDP', '고용지표', '실업률',
            '무역전쟁', '지정학적', '코로나', '변이', '봉쇄',
            '삼성전자', 'SK하이닉스', '네이버', '카카오', '현대차'
        ],
        'medium_impact': [
            '반도체', '자동차', '바이오', '게임', '배터리',
            '조선', '철강', '화학', '건설', '금융',
            '실적발표', '분기실적', '연간실적'
        ],
        'low_impact': [
            '공시', '보고서', '애널리스트', '목표주가',
            '투자의견', '리포트', '전망', '예상'
        ]
    }
    
    # 감정 강도 수식어
    self.intensity_modifiers = {
        '매우': 1.5, '극도로': 1.8, '대폭': 1.6, '크게': 1.3,
        '급': 1.4, '폭': 1.5, '심각하게': 1.7, '현저히': 1.4,
        '소폭': 0.7, '약간': 0.6, '다소': 0.6, '미미하게': 0.4,
        '살짝': 0.5, '조금': 0.6, '어느정도': 0.8
    }

async def initialize_models(self):
    """모델 초기화 및 로드"""
    try:
        logger.info("Initializing sentiment analysis models...")
        
        if TRANSFORMERS_AVAILABLE:
            await self._load_transformer_models()
        else:
            logger.warning("Using fallback dictionary-based sentiment analysis")
        
        if KONLPY_AVAILABLE:
            await self._initialize_korean_nlp()
        
        # 커스텀 모델이 있다면 로드
        await self._load_custom_financial_model()
        
        logger.info("Sentiment analysis models initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        # 폴백 모드로 실행
        logger.info("Running in fallback mode with dictionary-based analysis")

async def _load_transformer_models(self):
    """Transformer 모델 로드"""
    try:
        # 한국어 금융 특화 모델들 (실제 환경에서는 더 적절한 모델 사용)
        models_to_try = [
            "snunlp/KR-FinBert-SC",  # 한국어 금융 BERT
            "klue/roberta-base",     # KLUE RoBERTa
            "monologg/kobert",       # KoBERT
            "beomi/KcELECTRA-base"   # KcELECTRA
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying to load model: {model_name}")
                
                # 토크나이저 로드
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.model_config['cache_dir'])
                )
                
                # 감성분석 모델 로드
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    model_kwargs={
                        "cache_dir": str(self.model_config['cache_dir'])
                    }
                )
                
                logger.info(f"Successfully loaded model: {model_name}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        # 임베딩 모델 (유사도 분석용)
        try:
            self.embedding_model = SentenceTransformer(
                'jhgan/ko-sroberta-multitask',
                cache_folder=str(self.model_config['cache_dir'])
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            
    except Exception as e:
        logger.error(f"Error loading transformer models: {e}")
        self.tokenizer = None
        self.sentiment_model = None

async def _initialize_korean_nlp(self):
    """한국어 NLP 도구 초기화"""
    try:
        # Okt 형태소 분석기 (가벼움)
        self.korean_analyzer = Okt()
        logger.info("Korean NLP analyzer initialized")
        
    except Exception as e:
        logger.error(f"Error initializing Korean NLP: {e}")
        self.korean_analyzer = None

async def _load_custom_financial_model(self):
    """커스텀 금융 모델 로드 (있다면)"""
    try:
        custom_model_path = self.model_config['model_dir'] / 'financial_sentiment_model.pkl'
        
        if custom_model_path.exists():
            with open(custom_model_path, 'rb') as f:
                self.custom_model = pickle.load(f)
            logger.info("Custom financial sentiment model loaded")
        
    except Exception as e:
        logger.warning(f"No custom financial model found: {e}")

async def analyze_text(self, text: str, context: str = "general") -> SentimentResult:
    """단일 텍스트 감성분석"""
    try:
        if not text or len(text.strip()) < 3:
            return self._empty_sentiment_result(text)
        
        # 텍스트 전처리
        processed_text = await self._preprocess_text(text)
        
        # 캐시 확인
        cache_key = f"sentiment:{hash(processed_text)}:{context}"
        cached_result = await self.redis_client.get(cache_key)
        
        if cached_result:
            cached_data = json.loads(cached_result)
            cached_data['processed_at'] = datetime.fromisoformat(cached_data['processed_at'])
            return SentimentResult(**cached_data)
        
        # 감성분석 실행
        sentiment_scores = await self._calculate_sentiment_scores(processed_text)
        
        # 키워드 추출
        keywords = await self._extract_financial_keywords(processed_text)
        
        # 시장 영향도 계산
        market_impact, impact_score = await self._calculate_market_impact(processed_text, keywords)
        
        # 최종 감성 결정
        primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        confidence = sentiment_scores[primary_sentiment]
        
        result = SentimentResult(
            text=text,
            sentiment=primary_sentiment,
            confidence=confidence,
            scores=sentiment_scores,
            keywords=keywords,
            market_impact=market_impact,
            impact_score=impact_score,
            processed_at=datetime.now()
        )
        
        # 결과 캐시 (1시간)
        await self._cache_sentiment_result(cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing text sentiment: {e}")
        return self._empty_sentiment_result(text)

async def analyze_batch(self, texts: List[str], context: str = "general") -> List[SentimentResult]:
    """배치 텍스트 감성분석"""
    try:
        if not texts:
            return []
        
        logger.info(f"Analyzing batch of {len(texts)} texts")
        
        # 배치 크기로 분할
        batch_size = self.model_config['batch_size']
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 배치 내 병렬 처리
            batch_tasks = [
                self.analyze_text(text, context)
                for text in batch_texts
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 에러 처리
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in batch analysis {i+j}: {result}")
                    results.append(self._empty_sentiment_result(batch_texts[j]))
                else:
                    results.append(result)
            
            # 배치 간 지연 (모델 부하 방지)
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        logger.info(f"Batch analysis completed: {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {e}")
        return [self._empty_sentiment_result(text) for text in texts]

async def analyze_market_sentiment(
    self, 
    texts: List[str], 
    period: str = "1d",
    include_social: bool = False
) -> MarketSentiment:
    """시장 전체 감성분석"""
    try:
        logger.info(f"Analyzing market sentiment for period: {period}")
        
        if not texts:
            return self._empty_market_sentiment(period)
        
        # 배치 감성분석
        sentiment_results = await self.analyze_batch(texts, "market")
        
        # 전체 통계 계산
        total_texts = len(sentiment_results)
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        impact_scores = []
        all_keywords = []
        bullish_signals = []
        bearish_signals = []
        
        for result in sentiment_results:
            # 감성 분포
            sentiment_counts[result.sentiment] += 1
            
            # 영향도 점수
            impact_scores.append(result.impact_score)
            
            # 키워드 수집
            all_keywords.extend(result.keywords)
            
            # 신호 수집
            if result.market_impact == 'bullish' and result.confidence > 0.7:
                bullish_signals.append(result.text[:100])
            elif result.market_impact == 'bearish' and result.confidence > 0.7:
                bearish_signals.append(result.text[:100])
        
        # 감성 분포 계산
        sentiment_distribution = {
            sentiment: count / total_texts 
            for sentiment, count in sentiment_counts.items()
        }
        
        # 전체 감성 결정
        overall_sentiment = max(sentiment_distribution, key=sentiment_distribution.get)
        
        # 시장 분위기 판단
        market_mood = await self._determine_market_mood(sentiment_distribution, impact_scores)
        
        # 변동성 지표 계산
        volatility_indicator = np.std(impact_scores) if impact_scores else 0.0
        
        # 주요 테마 추출
        key_themes = await self._extract_market_themes(all_keywords)
        
        # 신뢰도 계산
        confidence_level = await self._calculate_market_confidence(sentiment_results)
        
        market_sentiment = MarketSentiment(
            period=period,
            total_texts=total_texts,
            overall_sentiment=overall_sentiment,
            sentiment_distribution=sentiment_distribution,
            market_mood=market_mood,
            volatility_indicator=min(1.0, volatility_indicator),
            key_themes=key_themes[:10],  # 상위 10개
            bullish_signals=bullish_signals[:5],  # 상위 5개
            bearish_signals=bearish_signals[:5],  # 상위 5개
            confidence_level=confidence_level,
            timestamp=datetime.now()
        )
        
        # 결과 캐시
        await self._cache_market_sentiment(market_sentiment)
        
        logger.info(f"Market sentiment analysis completed: {overall_sentiment} ({confidence_level:.2f})")
        return market_sentiment
        
    except Exception as e:
        logger.error(f"Error analyzing market sentiment: {e}")
        return self._empty_market_sentiment(period)

async def get_stock_sentiment(self, stock_code: str, texts: List[str]) -> Dict[str, Any]:
    """특정 종목 감성분석"""
    try:
        if not texts:
            return {'stock_code': stock_code, 'sentiment': 'neutral', 'confidence': 0.0}
        
        # 종목 관련 텍스트만 필터링
        relevant_texts = [
            text for text in texts 
            if stock_code in text or self._is_stock_related(text, stock_code)
        ]
        
        if not relevant_texts:
            relevant_texts = texts  # 모든 텍스트 사용
        
        # 감성분석
        results = await self.analyze_batch(relevant_texts, f"stock_{stock_code}")
        
        if not results:
            return {'stock_code': stock_code, 'sentiment': 'neutral', 'confidence': 0.0}
        
        # 종목별 감성 집계
        positive_scores = [r.scores.get('positive', 0) for r in results if r.scores]
        negative_scores = [r.scores.get('negative', 0) for r in results if r.scores]
        neutral_scores = [r.scores.get('neutral', 0) for r in results if r.scores]
        
        avg_positive = np.mean(positive_scores) if positive_scores else 0.0
        avg_negative = np.mean(negative_scores) if negative_scores else 0.0
        avg_neutral = np.mean(neutral_scores) if neutral_scores else 0.0
        
        # 최종 감성 결정
        scores = {'positive': avg_positive, 'negative': avg_negative, 'neutral': avg_neutral}
        final_sentiment = max(scores, key=scores.get)
        confidence = scores[final_sentiment]
        
        # 주요 키워드
        all_keywords = []
        for result in results:
            all_keywords.extend(result.keywords)
        
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'stock_code': stock_code,
            'sentiment': final_sentiment,
            'confidence': confidence,
            'scores': scores,
            'total_mentions': len(results),
            'relevant_texts': len(relevant_texts),
            'top_keywords': [kw[0] for kw in top_keywords],
            'bullish_ratio': len([r for r in results if r.market_impact == 'bullish']) / len(results),
            'bearish_ratio': len([r for r in results if r.market_impact == 'bearish']) / len(results),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing stock sentiment for {stock_code}: {e}")
        return {'stock_code': stock_code, 'sentiment': 'neutral', 'confidence': 0.0}

# 내부 헬퍼 메서드들
async def _preprocess_text(self, text: str) -> str:
    """텍스트 전처리"""
    try:
        # 기본 정리
        processed = text.strip()
        
        # 반복 문자 정규화 (soynlp 사용 가능시)
        if 'repeat_normalize' in globals():
            processed = repeat_normalize(processed, num_repeats=2)
        
        # URL, 이메일 제거
        processed = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', processed)
        processed = re.sub(r'\S+@\S+', '', processed)
        
        # 특수문자 정리 (일부 보존)
        processed = re.sub(r'[^\w\s\-+%↑↓]', ' ', processed)
        
        # 공백 정리
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
        
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return text

async def _calculate_sentiment_scores(self, text: str) -> Dict[str, float]:
    """감성 점수 계산"""
    try:
        scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Transformer 모델 사용 (가능한 경우)
        if self.sentiment_model:
            try:
                model_result = self.sentiment_model(text)
                
                if isinstance(model_result, list) and len(model_result) > 0:
                    result = model_result[0]
                    label = result.get('label', '').lower()
                    score = result.get('score', 0.0)
                    
                    if 'pos' in label or label == 'positive':
                        scores['positive'] = score
                        scores['negative'] = 1 - score
                    elif 'neg' in label or label == 'negative':
                        scores['negative'] = score  
                        scores['positive'] = 1 - score
                    else:
                        scores['neutral'] = score
                        scores['positive'] = (1 - score) / 2
                        scores['negative'] = (1 - score) / 2
                    
                    return scores
                    
            except Exception as e:
                logger.warning(f"Transformer model failed, using dictionary: {e}")
        
        # 사전 기반 감성분석 (폴백)
        words = text.split()
        positive_score = 0.0
        negative_score = 0.0
        total_words = len(words)
        
        for i, word in enumerate(words):
            # 긍정 키워드
            if word in self.financial_lexicon['positive']:
                base_score = self.financial_lexicon['positive'][word]
                
                # 수식어 적용
                if i > 0 and words[i-1] in self.intensity_modifiers:
                    base_score *= self.intensity_modifiers[words[i-1]]
                
                positive_score += base_score
            
            # 부정 키워드
            elif word in self.financial_lexicon['negative']:
                base_score = self.financial_lexicon['negative'][word]
                
                # 수식어 적용
                if i > 0 and words[i-1] in self.intensity_modifiers:
                    base_score *= self.intensity_modifiers[words[i-1]]
                
                negative_score += base_score
        
        # 정규화
        if total_words > 0:
            positive_score = min(1.0, positive_score / total_words * 10)
            negative_score = min(1.0, negative_score / total_words * 10)
        
        # 중립 점수 계산
        neutral_score = max(0.0, 1.0 - positive_score - negative_score)
        
        # 정규화 (합계 = 1.0)
        total = positive_score + negative_score + neutral_score
        if total > 0:
            scores['positive'] = positive_score / total
            scores['negative'] = negative_score / total
            scores['neutral'] = neutral_score / total
        
        return scores
        
    except Exception as e:
        logger.error(f"Error calculating sentiment scores: {e}")
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

async def _extract_financial_keywords(self, text: str) -> List[str]:
    """금융 키워드 추출"""
    try:
        keywords = []
        
        # 한국어 형태소 분석 (가능한 경우)
        if self.korean_analyzer:
            try:
                # 명사만 추출
                nouns = self.korean_analyzer.nouns(text)
                keywords.extend([noun for noun in nouns if len(noun) > 1])
            except Exception as e:
                logger.warning(f"Korean analysis failed: {e}")
        
        # 사전 기반 키워드 추출
        words = text.split()
        for word in words:
            # 금융 전문용어
            if (word in self.financial_lexicon['positive'] or 
                word in self.financial_lexicon['negative'] or
                any(keyword in word for keyword_list in self.market_impact_keywords.values() 
                    for keyword in keyword_list)):
                keywords.append(word)
        
        # 중복 제거 및 정리
        unique_keywords = list(set(keywords))
        
        # 길이 필터링 (너무 짧거나 긴 것 제외)
        filtered_keywords = [kw for kw in unique_keywords if 2 <= len(kw) <= 10]
        
        return filtered_keywords[:20]  # 상위 20개만
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

async def _calculate_market_impact(self, text: str, keywords: List[str]) -> Tuple[str, float]:
    """시장 영향도 계산"""
    try:
        impact_score = 0.0
        
        # 키워드 기반 영향도
        for keyword in keywords:
            if keyword in self.market_impact_keywords['high_impact']:
                impact_score += 0.3
            elif keyword in self.market_impact_keywords['medium_impact']:
                impact_score += 0.2
            elif keyword in self.market_impact_keywords['low_impact']:
                impact_score += 0.1
        
        # 감성 방향 적용
        positive_words = sum(1 for word in text.split() if word in self.financial_lexicon['positive'])
        negative_words = sum(1 for word in text.split() if word in self.financial_lexicon['negative'])
        
        if positive_words > negative_words:
            market_impact = 'bullish'
            impact_score = min(1.0, impact_score)
        elif negative_words > positive_words:
            market_impact = 'bearish'
            impact_score = -min(1.0, impact_score)
        else:
            market_impact = 'neutral'
            impact_score = 0.0
        
        return market_impact, impact_score
        
    except Exception as e
```