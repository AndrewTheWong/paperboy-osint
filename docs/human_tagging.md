# Human Article Tagging System

This document provides information about the human tagging system for OSINT articles in the Paperboy project.

## Overview

The human tagging system is a Streamlit-based web application that allows human analysts to review articles that have been automatically tagged by the system. Analysts can view article content, edit tags, and submit their reviews. This human-in-the-loop approach ensures that the tagging system improves over time and that important articles are properly categorized.

## Features

- Display articles that need human review
- Show article metadata (title, source, URL, scraped time)
- Show existing auto-generated tags
- Allow analysts to select tags from a comprehensive taxonomy
- Save reviewed articles with analyst attribution
- Track which articles have been reviewed

## Tag Taxonomy

The tagging system uses a comprehensive taxonomy organized by category:

| Category | Tags |
|----------|------|
| Military & Security | military, naval, aerospace, missile, coast guard, intelligence, nuclear |
| Geopolitics & Diplomacy | diplomacy, alliances, treaty, summit, sanctions, arms sale |
| Domestic Affairs | protest, riot, election, civil society, governance, law |
| Economy & Industry | trade, tariff, investment, semiconductors, technology, infrastructure |
| Cyber & Espionage | cyber, surveillance, espionage, disinformation, AI, hack |
| Gray Zone & Maritime | maritime, spratly, fishing fleet, air defense zone, incursion |
| Regional Focus | taiwan, south china sea, okinawa, philippines, japan, guam |
| Tone/Signal | escalatory, conciliatory, neutral, cooperative, threatening |

## Workflow

1. The system presents articles that require human review (based on the `needs_review` flag or having `["unknown"]` tags)
2. The analyst selects an article to review
3. The system displays the article's content and metadata
4. The analyst selects appropriate tags from the taxonomy
5. After submission:
   - The article is marked as reviewed (`needs_review = false`)
   - The selected tags are stored as `human_tags`
   - The analyst's name is recorded
   - The review timestamp is saved

## File Structure

- `human_tag_review.py`: Main Streamlit application
- `articles.json`: Source of articles to review
- `reviewed_articles.json`: Storage for reviewed articles

## Running the Application

To run the application, use the following command:

```bash
streamlit run human_tag_review.py
```

## Testing

The system includes comprehensive tests to ensure functionality:

```bash
python run_tests.py
```

This will:
1. Check that all required files and dependencies are present
2. Run unit tests for all components
3. Verify basic functionality of the application

## Tag Selection Interface

The tag selection interface presents tags in a hierarchical format:

```
Category – tag
```

For example:
- Military & Security – military
- Regional Focus – taiwan

This approach makes it easier for analysts to find relevant tags within categories while maintaining a flat data structure for storage.

## Data Model

### Input Article Format

```json
{
  "title": "PLA conducts live-fire drills",
  "translated_text": "The People's Liberation Army began exercises near the Taiwan Strait.",
  "url": "https://example.com/article",
  "source": "Global Times",
  "scraped_at": "2025-05-26T12:00:00Z",
  "tags": ["military", "naval"],
  "needs_review": true
}
```

### Reviewed Article Format

```json
{
  "title": "PLA conducts live-fire drills",
  "translated_text": "The People's Liberation Army began exercises near the Taiwan Strait.",
  "url": "https://example.com/article",
  "source": "Global Times",
  "scraped_at": "2025-05-26T12:00:00Z",
  "tags": ["military", "naval"],
  "human_tags": ["military", "naval", "missile", "taiwan", "escalatory"],
  "reviewed_by": "Analyst Name",
  "reviewed_at": "2025-05-26T15:30:00Z",
  "needs_review": false
}
``` 