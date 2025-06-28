#!/usr/bin/env python3
"""
Translation pipeline for news articles.
"""
import json
import logging
import argparse
from typing import List, Dict, Optional, Union
from pathlib import Path
import torch
from langdetect import detect_langs, LangDetectException
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('translation_pipeline')

# Translation model settings
DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEFAULT_BATCH_SIZE = 8

# Language code mapping (ISO 639-1 to NLLB model codes)
LANG_CODE_MAP = {
    'en': 'eng_Latn',  # English
    'zh': 'zho_Hans',  # Chinese (Simplified)
    'zh-tw': 'zho_Hant', # Chinese (Traditional)
    'ja': 'jpn_Jpan',  # Japanese
    'ko': 'kor_Hang',  # Korean
    'fr': 'fra_Latn',  # French
    'de': 'deu_Latn',  # German
    'es': 'spa_Latn',  # Spanish
    'pt': 'por_Latn',  # Portuguese
    'ru': 'rus_Cyrl',  # Russian
    'ar': 'arb_Arab',  # Arabic
    'th': 'tha_Thai',  # Thai
    'vi': 'vie_Latn',  # Vietnamese
    'ms': 'msa_Latn',  # Malay
    'id': 'ind_Latn',  # Indonesian
}

# Default source and target languages
DEFAULT_TARGET_LANG = 'en'

class ArticleTranslator:
    """Class to handle translation of articles using the NLLB model."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, target_lang: str = DEFAULT_TARGET_LANG, device: str = None):
        """
        Initialize the translator with the specified model.
        
        Args:
            model_name: Name of the Hugging Face translation model
            target_lang: Target language code (ISO 639-1)
            device: Device to use for inference ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.target_lang = target_lang
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing translator with model: {model_name} on {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        logger.info("Translator initialized successfully")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        
        Args:
            text: The text to detect language for
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'zh', 'ja')
        """
        try:
            # Get language with highest probability
            langs = detect_langs(text)
            if not langs:
                logger.warning(f"No language detected for text: {text[:50]}...")
                return self.target_lang
            
            # Return the most probable language
            return langs[0].lang
        except LangDetectException as e:
            logger.error(f"Language detection error: {str(e)}")
            return self.target_lang
    
    def translate_text(self, text: str, source_lang: str = None) -> str:
        """
        Translate text to the target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (if None, will be auto-detected)
            
        Returns:
            Translated text
        """
        if not text or text.strip() == "":
            return ""
        
        # Detect language if not provided
        if source_lang is None:
            source_lang = self.detect_language(text)
        
        # Skip translation if already in target language
        if source_lang == self.target_lang:
            return text
        
        # Convert ISO language code to model-specific code
        src_lang_code = LANG_CODE_MAP.get(source_lang, f"{source_lang}_Latn")
        tgt_lang_code = LANG_CODE_MAP.get(self.target_lang, f"{self.target_lang}_Latn")
        
        try:
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Set the source language
            inputs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[tgt_lang_code]
            
            # Generate translation
            with torch.no_grad():
                translated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang_code],
                    max_length=512
                )
            
            # Decode the generated tokens
            result = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            return result
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text
    
    def translate_batch(self, texts: List[str], source_langs: List[str] = None) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            source_langs: List of source language codes (if None, will be auto-detected)
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        # Detect languages if not provided
        if source_langs is None:
            source_langs = [self.detect_language(text) for text in texts]
        
        results = []
        for text, src_lang in zip(texts, source_langs):
            # Skip translation if already in target language
            if src_lang == self.target_lang:
                results.append(text)
                continue
                
            # Translate text
            translated = self.translate_text(text, src_lang)
            results.append(translated)
            
        return results

def auto_detect_text_fields(article: Dict[str, Any]) -> List[str]:
    """
    Automatically detect text fields that should be translated.
    
    Args:
        article: Article dictionary
        
    Returns:
        List of field names containing text content
    """
    text_fields = []
    
    # Common text field names
    common_fields = ['title', 'content', 'text', 'body', 'summary', 'description', 'headline']
    
    for field in common_fields:
        if field in article and isinstance(article[field], str) and len(article[field].strip()) > 0:
            text_fields.append(field)
    
    return text_fields

def translate_articles(articles: List[Dict], 
                      batch_size: int = DEFAULT_BATCH_SIZE, 
                      from_lang: Optional[str] = None, 
                      to_lang: str = DEFAULT_TARGET_LANG,
                      translate_fields: Optional[List[str]] = None,
                      auto_detect_fields: bool = True) -> List[Dict]:
    """
    Translate articles that are not in the target language.
    
    Args:
        articles: List of article dictionaries
        batch_size: Number of articles to translate in each batch
        from_lang: Source language code (if None, will be auto-detected)
        to_lang: Target language code
        translate_fields: List of fields to translate (if None, will auto-detect)
        auto_detect_fields: Whether to auto-detect text fields in articles
        
    Returns:
        List of articles with translations added
    """
    if not articles:
        logger.warning("No articles provided for translation")
        return []
    
    logger.info(f"Translating {len(articles)} articles from {from_lang or 'auto'} to {to_lang}")
    
    # Auto-detect fields if not provided
    if translate_fields is None and auto_detect_fields:
        # Get all unique text fields from all articles
        all_fields = set()
        for article in articles[:10]:  # Sample first 10 articles
            all_fields.update(auto_detect_text_fields(article))
        translate_fields = list(all_fields) if all_fields else ['title']
        logger.info(f"Auto-detected text fields: {translate_fields}")
    elif translate_fields is None:
        translate_fields = ['title']
    
    # Initialize translator
    translator = ArticleTranslator(target_lang=to_lang)
    
    translated_articles = []
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        
        # Process each field separately for better performance
        batch_copies = [article.copy() for article in batch]
        
        for field in translate_fields:
            # Extract texts and languages for this field
            texts = [article.get(field, "") for article in batch]
            
            # Skip empty texts
            if not any(texts):
                continue
            
            # Use provided language or get from article
            if from_lang:
                langs = [from_lang] * len(batch)
            else:
                langs = [article.get('language', translator.detect_language(article.get(field, ""))) for article in batch]
            
            # Translate texts for this field
            translated_texts = translator.translate_batch(texts, langs)
            
            # Update articles with translations for this field
            for j, (article_copy, translated_text) in enumerate(zip(batch_copies, translated_texts)):
                # Add translated text field
                translated_field = f"translated_{field}"
                article_copy[translated_field] = translated_text
                
                # Add detected language if not present
                if 'language' not in article_copy:
                    article_copy['language'] = langs[j]
        
        translated_articles.extend(batch_copies)
        logger.info(f"Translated batch {i//batch_size + 1}/{(len(articles) + batch_size - 1)//batch_size}")
    
    logger.info(f"Completed translation of {len(translated_articles)} articles")
    return translated_articles

def save_translated_articles(articles: List[Dict], output_path: Optional[str] = None) -> str:
    """
    Save translated articles to a JSON file.
    
    Args:
        articles: List of article dictionaries with translations
        output_path: Path to save the file (if None, a default path will be generated)
        
    Returns:
        Path to the saved file
    """
    if not output_path:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/translated_articles_{timestamp}.json"
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(articles)} translated articles to {output_file}")
    return str(output_file)

def load_articles(input_path: str) -> List[Dict]:
    """
    Load articles from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        List of article dictionaries
    """
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    logger.info(f"Loaded {len(articles)} articles from {input_file}")
    return articles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate news articles")
    parser.add_argument("--input", "-i", help="Path to input JSON file with articles")
    parser.add_argument("--output", "-o", help="Path to output JSON file for translated articles")
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for translation")
    parser.add_argument("--from-lang", "-f", help="Source language code (auto-detect if not provided)")
    parser.add_argument("--to-lang", "-t", default=DEFAULT_TARGET_LANG, help="Target language code")
    parser.add_argument("--field", default="title", help="Field to translate (default: title)")
    
    args = parser.parse_args()
    
    if args.input:
        # Load articles from file
        articles = load_articles(args.input)
        
        if articles:
            # Translate articles
            translated_articles = translate_articles(
                articles, 
                batch_size=args.batch_size,
                from_lang=args.from_lang,
                to_lang=args.to_lang,
                translate_field=args.field
            )
            
            # Save translated articles
            output_file = save_translated_articles(translated_articles, args.output)
            print(f"Translated articles saved to: {output_file}")
    else:
        print("No input file specified. Use --input to specify a JSON file with articles to translate.")
        print("Example: python translation_pipeline.py --input data/articles.json --output data/translated_articles.json") 