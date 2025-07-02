#!/usr/bin/env python3
"""
Summarization service for article clusters
"""

import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
_summarizer_model = None

def get_summarizer_model():
    """Get or create HuggingFace summarizer model"""
    global _summarizer_model
    if _summarizer_model is None:
        try:
            from transformers import pipeline
            _summarizer_model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU
            )
            logger.info("✅ Loaded HuggingFace summarizer model")
        except Exception as e:
            logger.error(f"❌ Error loading summarizer model: {e}")
            raise
    return _summarizer_model

def generate_summary(text: str, max_length: int = 150, min_length: int = 50) -> str:
    """
    Generate summary for text using HuggingFace summarizer
    
    Args:
        text: Input text to summarize
        max_length: Maximum length of summary
        min_length: Minimum length of summary
        
    Returns:
        str: Generated summary
    """
    try:
        if not text or len(text.strip()) < 100:
            return text[:200] + "..." if len(text) > 200 else text
        
        model = get_summarizer_model()
        
        # Truncate text if too long (model has limits)
        max_input_length = 1024
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        # Generate summary
        summary_result = model(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        
        summary = summary_result[0]['summary_text']
        
        logger.info(f"📄 Generated summary: {len(summary)} characters")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error generating summary: {e}")
        # Fallback: return first 200 characters
        return text[:200] + "..." if len(text) > 200 else text

def generate_batch_summaries(texts: list, max_length: int = 150, min_length: int = 50) -> list:
    """
    Generate summaries for a batch of texts
    
    Args:
        texts: List of texts to summarize
        max_length: Maximum length of each summary
        min_length: Minimum length of each summary
        
    Returns:
        list: List of generated summaries
    """
    try:
        model = get_summarizer_model()
        
        # Filter out short texts
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and len(text.strip()) >= 100:
                valid_texts.append(text[:1024])  # Truncate long texts
                valid_indices.append(i)
        
        if not valid_texts:
            # Return truncated versions for all texts
            return [text[:200] + "..." if len(text) > 200 else text for text in texts]
        
        # Generate summaries in batch
        summary_results = model(
            valid_texts,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        
        # Extract summaries
        summaries = [result['summary_text'] for result in summary_results]
        
        # Reconstruct full list with summaries in correct positions
        full_summaries = []
        summary_idx = 0
        
        for i in range(len(texts)):
            if i in valid_indices:
                full_summaries.append(summaries[summary_idx])
                summary_idx += 1
            else:
                # Use truncated text for short inputs
                text = texts[i]
                full_summaries.append(text[:200] + "..." if len(text) > 200 else text)
        
        logger.info(f"📄 Generated {len(summaries)} summaries from {len(texts)} texts")
        
        return full_summaries
        
    except Exception as e:
        logger.error(f"❌ Error generating batch summaries: {e}")
        # Fallback: return truncated versions
        return [text[:200] + "..." if len(text) > 200 else text for text in texts] 