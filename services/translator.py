#!/usr/bin/env python3
"""
Translation service for articles
Translates articles from various languages to English for better processing
"""

import logging
import time
import os
from typing import List, Dict, Any, Optional
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationService:
    """Translation service using multiple APIs or fallback methods"""
    
    def __init__(self, api_key: Optional[str] = None, api_type: str = "google"):
        self.api_key = api_key
        self.api_type = api_type
        self.google_url = "https://translation.googleapis.com/language/translate/v2"
        self.libretranslate_url = "https://libretranslate.de/translate"  # Free API
        self.mymemory_url = "https://api.mymemory.translated.net/get"
        
    def detect_language(self, text: str) -> str:
        """Detect the language of the given text"""
        if not text or len(text.strip()) < 3:
            return "en"  # Default to English for very short texts
            
        # Simple language detection based on character sets
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff')
        korean_chars = sum(1 for char in text if '\uac00' <= char <= '\ud7af')
        
        # Calculate percentages
        total_chars = len(text)
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        japanese_ratio = japanese_chars / total_chars if total_chars > 0 else 0
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        
        logger.info(f"Language detection for text '{text[:20]}...': Chinese={chinese_ratio:.2f}, Japanese={japanese_ratio:.2f}, Korean={korean_ratio:.2f}")
        
        if chinese_ratio > 0.1:  # Lower threshold for Chinese
            return "zh"
        elif japanese_ratio > 0.1:
            return "ja"
        elif korean_ratio > 0.1:
            return "ko"
        else:
            return "en"  # Default to English
    
    def translate_text(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """Translate text using available APIs"""
        if not text or len(text.strip()) < 10:
            return text
            
        # If already English, return as is
        if source_lang == "en" or (source_lang == "auto" and self.detect_language(text) == "en"):
            return text
            
        try:
            if self.api_key and self.api_type == "google":
                return self._translate_with_google_api(text, source_lang, target_lang)
            elif self.api_type == "libretranslate":
                return self._translate_with_libretranslate(text, source_lang, target_lang)
            elif self.api_type == "mymemory":
                return self._translate_with_mymemory(text, source_lang, target_lang)
            else:
                logger.error(f"No valid translation API configured for {self.api_type}")
                return text
        except Exception as e:
            logger.error(f"Translation failed: {e}, returning original text")
            return text
    
    def _translate_with_google_api(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Google Translate API"""
        try:
            url = f"{self.google_url}?key={self.api_key}"
            data = {
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text"
            }
            
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result["data"]["translations"][0]["translatedText"]
            
            logger.info(f"✅ Google API: Translated {len(text)} chars from {source_lang} to {target_lang}")
            return translated_text
            
        except Exception as e:
            logger.error(f"Google API translation failed: {e}")
            return text
    
    def _translate_with_libretranslate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using LibreTranslate (free API)"""
        try:
            data = {
                "q": text,
                "source": source_lang,
                "target": target_lang
            }
            
            response = requests.post(self.libretranslate_url, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result["translatedText"]
            
            logger.info(f"✅ LibreTranslate: Translated {len(text)} chars from {source_lang} to {target_lang}")
            return translated_text
            
        except Exception as e:
            logger.error(f"LibreTranslate failed: {e}")
            return text
    
    def _translate_with_mymemory(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using MyMemory (free API) with chunking for long content"""
        # MyMemory has a 500 character limit
        MAX_CHUNK_SIZE = 450  # Leave some buffer
        
        if len(text) <= MAX_CHUNK_SIZE:
            return self._translate_mymemory_single(text, source_lang, target_lang)
        
        # Split text into chunks
        chunks = self._split_text_into_chunks(text, MAX_CHUNK_SIZE)
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                translated_chunk = self._translate_mymemory_single(chunk, source_lang, target_lang)
                translated_chunks.append(translated_chunk)
                logger.info(f"✅ MyMemory: Translated chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            except Exception as e:
                logger.error(f"Failed to translate chunk {i+1}: {e}")
                translated_chunks.append(chunk)  # Keep original if translation fails
        
        # Recombine chunks
        return ' '.join(translated_chunks)
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into chunks, trying to break at sentence boundaries"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first, then by words if needed
        sentences = text.split('. ')
        
        for sentence in sentences:
            # Add period back except for the last sentence
            if not sentence.endswith('.') and sentence != sentences[-1]:
                sentence += '. '
            elif sentence != sentences[-1]:
                sentence += ' '
            
            # If adding this sentence would exceed the limit
            if len(current_chunk + sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    for word in words:
                        if len(current_chunk + word + ' ') > max_chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = word + ' '
                            else:
                                # Single word is too long, truncate
                                chunks.append(word[:max_chunk_size])
                        else:
                            current_chunk += word + ' '
            else:
                current_chunk += sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _translate_mymemory_single(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single chunk using MyMemory API"""
        try:
            params = {
                "q": text,
                "langpair": f"{source_lang}|{target_lang}"
            }
            
            response = requests.get(self.mymemory_url, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result["responseData"]["translatedText"]
            
            logger.info(f"✅ MyMemory: Translated {len(text)} chars from {source_lang} to {target_lang}")
            return translated_text
            
        except Exception as e:
            logger.error(f"MyMemory failed: {e}")
            return text
    

    
    def translate_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Translate an article's title and content"""
        try:
            # Detect language
            title = article.get('title', '')
            content = article.get('body', article.get('content', ''))
            
            # Detect source language
            title_lang = self.detect_language(title)
            content_lang = self.detect_language(content)
            
            # Translate title if needed
            if title_lang != "en":
                translated_title = self.translate_text(title, title_lang, "en")
                article['title_translated'] = translated_title
                article['title_original'] = title
                article['title_language'] = title_lang
            else:
                article['title_translated'] = title
                article['title_original'] = title
                article['title_language'] = "en"
            
            # Translate content if needed
            if content_lang != "en":
                translated_content = self.translate_text(content, content_lang, "en")
                article['content_translated'] = translated_content
                article['content_original'] = content
                article['content_language'] = content_lang
            else:
                article['content_translated'] = content
                article['content_original'] = content
                article['content_language'] = "en"
            
            # Update cleaned_text for downstream processing
            article['cleaned_text'] = article['content_translated']
            
            logger.info(f"✅ Translated article {article.get('article_id', 'unknown')}: {title_lang}→en, {content_lang}→en")
            return article
            
        except Exception as e:
            logger.error(f"❌ Translation failed for article {article.get('article_id', 'unknown')}: {e}")
            # Return original article if translation fails
            return article
    
    def translate_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate a batch of articles"""
        logger.info(f"🔄 Starting batch translation of {len(articles)} articles")
        start_time = time.time()
        
        translated_articles = []
        for i, article in enumerate(articles):
            try:
                translated_article = self.translate_article(article)
                translated_articles.append(translated_article)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Translated {i + 1}/{len(articles)} articles")
                    
            except Exception as e:
                logger.error(f"❌ Failed to translate article {i}: {e}")
                translated_articles.append(article)  # Keep original if translation fails
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Batch translation completed in {elapsed:.2f}s: {len(translated_articles)} articles")
        
        return translated_articles

def get_translation_service(api_key: Optional[str] = None, api_type: str = "google") -> TranslationService:
    """Get a translation service instance"""
    return TranslationService(api_key, api_type)

def translate_article_simple(article: Dict[str, Any]) -> Dict[str, Any]:
    """Simple translation function for single article"""
    service = get_translation_service()
    return service.translate_article(article)

def translate_articles_batch_simple(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple batch translation function"""
    service = get_translation_service()
    return service.translate_articles_batch(articles) 