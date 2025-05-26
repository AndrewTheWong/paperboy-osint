# Translation Pipeline Documentation

The translation pipeline is responsible for detecting languages and translating non-English articles into English. It uses the Facebook NLLB (No Language Left Behind) model for high-quality machine translation.

## Components

### 1. ArticleTranslator Class

The core of the translation pipeline is the `ArticleTranslator` class, which handles:

- Loading the translation model and tokenizer
- Language detection
- Text translation
- Batch processing

### 2. Main Functions

- `translate_articles`: Processes a list of article dictionaries, translating the specified field
- `save_translated_articles`: Saves translated articles to a JSON file
- `load_articles`: Loads articles from a JSON file

## Supported Languages

The translation pipeline supports a wide range of languages, including:

- Chinese (Simplified and Traditional)
- Japanese
- Korean
- Russian
- Arabic
- Thai
- Vietnamese
- French
- German
- Spanish
- Portuguese
- and many more

## Usage Examples

### Basic Usage

```python
from pipelines.translation_pipeline import translate_articles

# List of article dictionaries
articles = [
    {
        'title': 'This is an English article',
        'url': 'https://example.com/en/article1',
        'source': 'Test Source',
        'language': 'en'
    },
    {
        'title': '这是一篇中文文章',
        'url': 'https://example.com/zh/article2',
        'source': 'Test Source',
        'language': 'zh'
    }
]

# Translate the articles
translated_articles = translate_articles(articles)

# Access translated titles
for article in translated_articles:
    print(f"Original: {article['title']}")
    print(f"Translated: {article.get('translated_title', 'No translation')}")
```

### Advanced Usage

```python
from pipelines.translation_pipeline import translate_articles

# Translate with custom settings
translated_articles = translate_articles(
    articles,
    batch_size=16,           # Process 16 articles at a time
    from_lang='ja',          # Force source language to Japanese
    to_lang='fr',            # Target language is French
    translate_field='body'   # Translate the 'body' field instead of 'title'
)
```

### Command Line Usage

```bash
# Translate articles from a file
python -m pipelines.translation_pipeline --input data/articles.json --output data/translated.json

# Specify batch size and languages
python -m pipelines.translation_pipeline --input data/articles.json --batch-size 16 --from-lang zh --to-lang en

# Translate a different field
python -m pipelines.translation_pipeline --input data/articles.json --field body
```

## Performance Considerations

- Translation is computationally intensive, especially for long texts
- Using a GPU significantly improves performance
- Batch processing helps optimize throughput
- The model requires approximately 1.2GB of RAM
- First-time use will download the model (~1.2GB) 