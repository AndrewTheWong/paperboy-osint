#!/usr/bin/env python3
"""
Main entry point for the Paperboy news analysis pipeline.
"""
import os
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Main entry point for the Paperboy news analysis pipeline."""
    parser = argparse.ArgumentParser(description="Paperboy News Analysis Pipeline")
    parser.add_argument("--scrape", action="store_true", help="Run the scraper pipeline")
    parser.add_argument("--tag", action="store_true", help="Run the auto-tagging pipeline")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings for articles")
    parser.add_argument("--cluster", action="store_true", help="Cluster articles by topic similarity")
    parser.add_argument("--store", action="store_true", help="Store articles in Supabase")
    parser.add_argument("--review", action="store_true", help="Launch the human review UI")
    parser.add_argument("--train", action="store_true", help="Train models with tagged data")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline")
    
    args = parser.parse_args()
    
    # If no arguments are provided, run everything
    if not any(vars(args).values()):
        args.all = True
    
    # Run the requested components
    if args.all or args.scrape:
        print("Running scraper pipeline...")
        from pipelines.dynamic_scraper import scrape_all_dynamic, save_articles_to_file
        
        articles = scrape_all_dynamic()
        if articles:
            save_articles_to_file(articles, "data/articles.json")
            print(f"Scraped {len(articles)} articles to data/articles.json")
        else:
            print("No articles were scraped.")
    
    if args.all or args.tag:
        print("Running auto-tagging pipeline...")
        from tagging.tagging_pipeline import load_articles, tag_articles, save_tagged_articles
        
        try:
            articles = load_articles("data/articles.json")
            tagged_articles = tag_articles(articles)
            save_tagged_articles(tagged_articles, "data/articles.json")
            
            # Print tagging statistics
            review_count = sum(1 for a in tagged_articles if a.get("needs_review", False))
            unknown_count = sum(1 for a in tagged_articles if "unknown" in a.get("tags", []))
            print(f"Tagged {len(tagged_articles)} articles")
            print(f"{review_count} articles need human review")
            print(f"{unknown_count} articles couldn't be automatically tagged")
        except Exception as e:
            print(f"Error during tagging: {str(e)}")
            
    if args.all or args.embed:
        print("Generating article embeddings...")
        from pipelines.embedding_pipeline import embed_articles, save_embedded_articles
        
        try:
            # Load articles (reuse tagging module's load function)
            from tagging.tagging_pipeline import load_articles
            articles = load_articles("data/articles.json")
            
            # Generate embeddings
            embedded_articles = embed_articles(articles)
            
            # Save articles with embeddings
            save_embedded_articles(embedded_articles, "data/embedded_articles.json")
            
            # Report statistics
            if embedded_articles and 'embedding' in embedded_articles[0]:
                emb_dim = len(embedded_articles[0]['embedding'])
                print(f"Generated {emb_dim}-dimensional embeddings for {len(embedded_articles)} articles")
                print(f"Saved to data/embedded_articles.json")
            else:
                print("Failed to generate embeddings")
                
        except Exception as e:
            print(f"Error during embedding generation: {str(e)}")
            
    if args.all or args.cluster:
        print("Clustering articles by topic similarity...")
        from pipelines.cluster_articles import cluster_articles
        
        try:
            # Load articles with embeddings
            from tagging.tagging_pipeline import load_articles
            articles = load_articles("data/embedded_articles.json")
            
            # Cluster articles
            clustered_articles = cluster_articles(articles=articles, output_path="data/clustered_articles.json")
            
            # Report statistics
            if clustered_articles:
                cluster_ids = [article.get('cluster_id', -1) for article in clustered_articles]
                n_clusters = len(set(cluster_ids) - {-1})
                n_noise = cluster_ids.count(-1)
                
                print(f"✅ Clustered {len(clustered_articles)} articles into {n_clusters} clusters")
                print(f"📊 Noise points: {n_noise} ({n_noise/len(clustered_articles)*100:.1f}%)")
                print(f"Saved to data/clustered_articles.json")
            else:
                print("Failed to cluster articles")
                
        except Exception as e:
            print(f"Error during article clustering: {str(e)}")
            
    if args.all or args.store:
        print("Storing articles in Supabase...")
        from pipelines.supabase_storage import load_articles_from_file, upload_articles_to_supabase
        
        try:
            # Try to load clustered articles first, fall back to embedded if needed
            clustered_path = "data/clustered_articles.json"
            embedded_path = "data/embedded_articles.json"
            
            if os.path.exists(clustered_path):
                articles = load_articles_from_file(clustered_path)
                file_used = clustered_path
            else:
                articles = load_articles_from_file(embedded_path)
                file_used = embedded_path
            
            if not articles:
                print("No articles to store. Make sure to run embedding/clustering steps first.")
            else:
                # Upload to Supabase
                uploaded_count = upload_articles_to_supabase(articles)
                print(f"Successfully stored {uploaded_count} new articles from {file_used}")
                
        except Exception as e:
            print(f"Error during article storage: {str(e)}")
    
    if args.all or args.review:
        print("Launching human review UI...")
        try:
            subprocess.Popen(["streamlit", "run", "ui/human_tag_review.py"])
            print("Human review UI is now running. Press Ctrl+C to stop it.")
            try:
                while True:
                    pass  # Keep the main process running
            except KeyboardInterrupt:
                print("Human review UI stopped.")
        except Exception as e:
            print(f"Error launching human review UI: {str(e)}")
    
    if args.all or args.train:
        print("Training models not implemented yet.")
        # This would be where you'd add model training code

if __name__ == "__main__":
    main()
