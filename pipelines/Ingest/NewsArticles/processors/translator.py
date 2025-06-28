#!/usr/bin/env python3
"""
Translation Processor

Handles translation of articles from various languages to English
using multiple translation backends with fallbacks.
"""

import logging
import re
from typing import Dict, Optional, Any

# Translation services
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from googletrans import Translator
    HAS_GOOGLETRANS = True
except ImportError:
    HAS_GOOGLETRANS = False

try:
    import argostranslate.package
    import argostranslate.translate
    HAS_ARGOS = True
except ImportError:
    HAS_ARGOS = False

logger = logging.getLogger(__name__)

class TranslationProcessor:
    """Handles article translation with multiple backends."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize translation processor."""
        self.openai_api_key = openai_api_key
        
        # Initialize Google Translator
        self.google_translator = None
        if HAS_GOOGLETRANS:
            try:
                self.google_translator = Translator()
                logger.info("✅ Google Translator initialized")
            except Exception as e:
                logger.warning(f"⚠️ Google Translator failed to initialize: {e}")
        
        # Initialize OpenAI
        if HAS_OPENAI and openai_api_key:
            try:
                openai.api_key = openai_api_key
                logger.info("✅ OpenAI translator initialized")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI initialization failed: {e}")
        
        # Language detection patterns
        self.language_patterns = self._build_language_patterns()
        
    def _build_language_patterns(self) -> Dict[str, list]:
        """Build patterns for language detection."""
        return {
            'chinese': [
                r'[\u4e00-\u9fff]+',  # Chinese characters
                r'[\u3400-\u4dbf]+',  # CJK Extension A
            ],
            'japanese': [
                r'[\u3040-\u309f]+',  # Hiragana
                r'[\u30a0-\u30ff]+',  # Katakana
            ],
            'korean': [
                r'[\uac00-\ud7af]+',  # Hangul
            ],
            'arabic': [
                r'[\u0600-\u06ff]+',  # Arabic
            ],
            'russian': [
                r'[\u0400-\u04ff]+',  # Cyrillic
            ]
        }
        
    def detect_language(self, text: str) -> str:
        """Detect language of text using patterns."""
        if not text:
            return 'en'
            
        text_sample = text[:1000]  # Use first 1000 chars for detection
        
        # Check for non-Latin scripts
        for language, patterns in self.language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_sample):
                    return language
                    
        # If mostly ASCII, assume English
        ascii_ratio = sum(1 for c in text_sample if ord(c) < 128) / len(text_sample)
        if ascii_ratio > 0.8:
            return 'en'
            
        # Default to unknown
        return 'unknown'
        
    def needs_translation(self, text: str) -> bool:
        """Check if text needs translation to English."""
        if not text:
            return False
            
        detected_lang = self.detect_language(text)
        return detected_lang != 'en'
        
    def translate_with_openai(self, text: str, source_lang: str = 'auto') -> Optional[str]:
        """Translate text using OpenAI GPT."""
        if not HAS_OPENAI or not self.openai_api_key:
            return None
            
        try:
            # Limit text length for API
            text_chunk = text[:3000] if len(text) > 3000 else text
            
            prompt = f"""Translate the following text to English. Preserve the original meaning and tone. Only return the translation, no explanations:

{text_chunk}"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"OpenAI translation failed: {e}")
            return None
            
    def translate_with_google(self, text: str, source_lang: str = 'auto') -> Optional[str]:
        """Translate text using Google Translate."""
        if not self.google_translator:
            return None
            
        try:
            # Limit text length
            text_chunk = text[:4000] if len(text) > 4000 else text
            
            result = self.google_translator.translate(
                text_chunk, 
                src=source_lang if source_lang != 'auto' else None,
                dest='en'
            )
            
            return result.text
            
        except Exception as e:
            logger.warning(f"Google translation failed: {e}")
            return None
            
    def translate_with_argos(self, text: str, source_lang: str) -> Optional[str]:
        """Translate text using Argos Translate."""
        if not HAS_ARGOS:
            return None
            
        try:
            # Map language codes
            lang_map = {
                'chinese': 'zh',
                'japanese': 'ja', 
                'korean': 'ko',
                'russian': 'ru',
                'arabic': 'ar'
            }
            
            source_code = lang_map.get(source_lang, source_lang)
            if source_code == 'en':
                return text
                
            # Limit text length
            text_chunk = text[:2000] if len(text) > 2000 else text
            
            # Translate
            translated = argostranslate.translate.translate(text_chunk, source_code, 'en')
            return translated
            
        except Exception as e:
            logger.warning(f"Argos translation failed: {e}")
            return None
            
    def translate_text(self, text: str, source_lang: str = 'auto') -> Dict[str, Any]:
        """Translate text with fallback methods."""
        result = {
            'original_text': text,
            'translated_text': text,
            'source_language': 'en',
            'translation_method': 'none',
            'translation_confidence': 1.0,
            'needs_translation': False
        }
        
        if not text:
            return result
            
        # Detect language if auto
        if source_lang == 'auto':
            detected_lang = self.detect_language(text)
            result['source_language'] = detected_lang
        else:
            detected_lang = source_lang
            result['source_language'] = source_lang
            
        # Check if translation needed
        if not self.needs_translation(text):
            return result
            
        result['needs_translation'] = True
        
        # Try translation methods in order of preference
        translated = None
        method_used = 'none'
        
        # 1. Try OpenAI (best quality)
        if not translated and HAS_OPENAI and self.openai_api_key:
            translated = self.translate_with_openai(text, detected_lang)
            if translated:
                method_used = 'openai'
                result['translation_confidence'] = 0.9
                
        # 2. Try Google Translate
        if not translated:
            translated = self.translate_with_google(text, detected_lang)
            if translated:
                method_used = 'google'
                result['translation_confidence'] = 0.7
                
        # 3. Try Argos (local, private)
        if not translated:
            translated = self.translate_with_argos(text, detected_lang)
            if translated:
                method_used = 'argos'
                result['translation_confidence'] = 0.6
                
        # Update result
        if translated:
            result['translated_text'] = translated
            result['translation_method'] = method_used
        else:
            # Translation failed, use original
            result['translation_method'] = 'failed'
            result['translation_confidence'] = 0.0
            
        return result
        
    def translate_article(self, title: str, content: str) -> Dict[str, Any]:
        """Translate full article (title + content)."""
        
        # Translate title
        title_result = self.translate_text(title)
        
        # Translate content  
        content_result = self.translate_text(content)
        
        # Combined result
        result = {
            'original_title': title,
            'translated_title': title_result['translated_text'],
            'title_language': title_result['source_language'],
            'title_translation_method': title_result['translation_method'],
            
            'original_content': content,
            'translated_content': content_result['translated_text'],
            'content_language': content_result['source_language'],
            'content_translation_method': content_result['translation_method'],
            
            'overall_confidence': (title_result['translation_confidence'] + 
                                 content_result['translation_confidence']) / 2,
            'needs_translation': title_result['needs_translation'] or content_result['needs_translation']
        }
        
        return result 