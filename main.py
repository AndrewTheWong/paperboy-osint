import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import pipeline components
from ingestion.ingest import ingest_sources
from models.tagging import process_untagged_entries
from models.predict import run_prediction_pipeline
from dashboard.ui import launch_dashboard

def main():
    """Main entry point for the Agentic AI Analyst pipeline."""
    parser = argparse.ArgumentParser(description="Agentic AI Analyst Pipeline")
    parser.add_argument("--ingest", action="store_true", help="Run the ingestion pipeline")
    parser.add_argument("--tag", action="store_true", help="Run the auto-tagging pipeline")
    parser.add_argument("--predict", action="store_true", help="Run the prediction pipeline")
    parser.add_argument("--dashboard", action="store_true", help="Launch the dashboard")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline")
    
    args = parser.parse_args()
    
    # If no arguments are provided, run everything
    if not any(vars(args).values()):
        args.all = True
    
    # Run the requested components
    if args.all or args.ingest:
        print("Running ingestion pipeline...")
        ingest_sources()
    
    if args.all or args.tag:
        print("Running auto-tagging pipeline...")
        process_untagged_entries()
    
    if args.all or args.predict:
        print("Running prediction pipeline...")
        run_prediction_pipeline()
    
    if args.all or args.dashboard:
        print("Launching dashboard...")
        dashboard_process = launch_dashboard()
        # Keep the script running while the dashboard is open
        if dashboard_process:
            try:
                dashboard_process.wait()
            except KeyboardInterrupt:
                print("Dashboard closed.")

if __name__ == "__main__":
    main()
