#!/usr/bin/env python3
"""
Demonstration script that combines the scraper and translation pipeline.
"""
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scrape_and_translate')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape and translate news articles")
    parser.add_argument("--output", "-o", help="Path to output JSON file for translated articles")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size for translation")
    parser.add_argument("--to-lang", "-t", default="en", help="Target language code")
    parser.add_argument("--field", default="title", help="Field to translate (default: title)")
    parser.add_argument("--skip-translation", action="store_true", help="Skip translation and only scrape")
    
    args = parser.parse_args()
    
    # Import here to avoid loading the models unless needed
    logger.info("Starting scraper")
    from pipelines.dynamic_scraper import scrape_all_dynamic, save_articles_to_file
    
    # Run the scraper
    articles = scrape_all_dynamic()
    
    if not articles:
        logger.warning("No articles were scraped")
        exit(0)
    
    # Save raw articles
    raw_output = save_articles_to_file(articles)
    logger.info(f"Raw articles saved to: {raw_output}")
    
    # Skip translation if requested
    if args.skip_translation:
        logger.info("Translation skipped as requested")
        exit(0)
    
    # Translate the articles
    logger.info("Starting translation")
    from pipelines.translation_pipeline import translate_articles, save_translated_articles
    
    translated_articles = translate_articles(
        articles,
        batch_size=args.batch_size,
        to_lang=args.to_lang,
        translate_field=args.field
    )
    
    # Save translated articles
    output_file = save_translated_articles(translated_articles, args.output)
    logger.info(f"Translated articles saved to: {output_file}")
    
    # Print summary
    total_translated = sum(1 for a in translated_articles if f"translated_{args.field}" in a)
    logger.info(f"Summary: {len(articles)} articles scraped, {total_translated} articles translated") 