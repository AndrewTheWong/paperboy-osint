import re
import html
from typing import Optional, Dict, Any
from services.language_detector import detect_language
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and normalize article text
    """
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Chinese characters
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def clean_html_text(text: str) -> str:
    return clean_text(text)

def normalize_text(text: str) -> str:
    return clean_text(text).lower()

def extract_title_from_html(html_content: str) -> Optional[str]:
    """
    Extract title from HTML content
    """
    if not html_content:
        return None
    
    # Look for title tag
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE)
    if title_match:
        return clean_text(title_match.group(1))
    
    # Look for h1 tag
    h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html_content, re.IGNORECASE)
    if h1_match:
        return clean_text(h1_match.group(1))
    
    return None

def extract_meta_description(html_content: str) -> Optional[str]:
    """
    Extract meta description from HTML content
    """
    if not html_content:
        return None
    
    # Look for meta description
    meta_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', html_content, re.IGNORECASE)
    if meta_match:
        return clean_text(meta_match.group(1))
    
    return None

def clean_and_translate_text(text: str, source_lang: str = "auto", target_lang: str = "en") -> Dict[str, Any]:
    """Clean text and translate if needed"""
    try:
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Detect language if not specified
        if source_lang == "auto":
            detected_lang = detect_language(cleaned_text)
            source_lang = detected_lang
            logger.info(f"Language detection for text '{cleaned_text[:50]}...': {detected_lang}")
        
        # Determine if translation is needed
        needs_translation = source_lang not in ['en', 'english'] and len(cleaned_text) > 10
        
        translated_text = cleaned_text
        translation_status = "not_needed"
        
        if needs_translation:
            try:
                from services.translator import translate_text
                translated_text = translate_text(cleaned_text, source_lang, target_lang)
                translation_status = "translated"
                logger.info(f"✅ Translated text from {source_lang} to {target_lang}")
            except Exception as e:
                logger.warning(f"⚠️ Translation failed for {source_lang} text: {e}")
                # Keep original text if translation fails
                translated_text = cleaned_text
                translation_status = "failed"
        
        return {
            'cleaned_text': cleaned_text,
            'translated_text': translated_text,
            'original_language': source_lang,
            'translation_status': translation_status,
            'needs_translation': needs_translation
        }
        
    except Exception as e:
        logger.error(f"❌ Error in clean_and_translate_text: {e}")
        return {
            'cleaned_text': text,
            'translated_text': text,
            'original_language': 'unknown',
            'translation_status': 'error',
            'needs_translation': False
        } 

__all__ = [
    'clean_text',
    'clean_html_text',
    'normalize_text',
    'extract_title_from_html',
    'extract_meta_description',
    'clean_and_translate_text',
] 