import re
from typing import Dict, Tuple

def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns
    Returns language code like 'en', 'zh-cn', 'ja', 'ko'
    """
    if not text:
        return 'en'
    
    # Count Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # Count Japanese characters (Hiragana, Katakana, Kanji)
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))
    # Count Korean characters
    korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
    # Count total characters
    total_chars = len(text)
    
    if total_chars == 0:
        return 'en'
    
    # Calculate ratios
    chinese_ratio = chinese_chars / total_chars
    japanese_ratio = japanese_chars / total_chars
    korean_ratio = korean_chars / total_chars
    
    # Determine language based on highest ratio
    if chinese_ratio > 0.1:
        return 'zh-cn'
    elif japanese_ratio > 0.1:
        return 'ja'
    elif korean_ratio > 0.1:
        return 'ko'
    else:
        return 'en'

def detect_language_with_confidence(text: str) -> Dict[str, float]:
    """
    Detect language with confidence scores
    """
    if not text:
        return {'en': 1.0}
    
    # Count characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))
    korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
    total_chars = len(text)
    
    if total_chars == 0:
        return {'en': 1.0}
    
    # Calculate ratios
    chinese_ratio = chinese_chars / total_chars
    japanese_ratio = japanese_chars / total_chars
    korean_ratio = korean_chars / total_chars
    english_ratio = 1.0 - (chinese_ratio + japanese_ratio + korean_ratio)
    
    return {
        'zh-cn': chinese_ratio,
        'ja': japanese_ratio,
        'ko': korean_ratio,
        'en': english_ratio
    } 