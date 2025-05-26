# Agentic AI Analyst

An intelligent system for ingesting, analyzing, and predicting events based on OSINT (Open Source Intelligence) data. This MVP integrates data collection, auto-tagging, and prediction capabilities with a human-in-the-loop review interface.

## Features

- **OSINT Ingestion**: Collect data from RSS feeds, static websites, and PDF documents
- **Automated Tagging**: NER-based entity recognition and rule-based tagging
- **Event Prediction**: Generate event predictions with likelihood scores and regional focus
- **Human-in-the-Loop Interface**: Streamlit dashboard for reviewing and correcting tags
- **Analytics Dashboard**: Visualize predictions and insights

## System Architecture

The system consists of four main components:

1. **Ingestion Engine**: Collects data from various sources
2. **Storage Layer**: Manages data in Supabase
3. **ML Models**: NER tagging and prediction algorithms
4. **Dashboard**: User interface for review and visualization

## Setup and Installation

### Prerequisites

- Python 3.8+
- Supabase account
- API keys for Supabase

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Create a `.env` file with your Supabase credentials:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

## Usage

Run the full pipeline:
```
python main.py
```

Or run individual components:
```
python main.py --ingest    # Run the ingestion pipeline
python main.py --tag       # Run the auto-tagging pipeline
python main.py --predict   # Run the prediction pipeline
python main.py --dashboard # Launch the dashboard
```

## Testing

Test the Supabase connection:
```
python test_supabase.py
```

## Database Schema

### OSINT Raw Data
- `id`: UUID (primary key)
- `source_url`: TEXT
- `content`: TEXT
- `ingested_at`: TIMESTAMPTZ
- `tags`: TEXT[]
- `confidence_score`: FLOAT
- `manual_review`: BOOLEAN

### Predictions
- `id`: UUID (primary key)
- `osint_id`: UUID (foreign key)
- `event_type`: TEXT
- `region`: TEXT
- `likelihood_score`: FLOAT
- `model_used`: TEXT
- `generated_at`: TIMESTAMPTZ

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 