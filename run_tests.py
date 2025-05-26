#!/usr/bin/env python3
"""
Script to run all tests for the human_tag_review.py implementation.
"""
import os
import sys
import unittest
import json
from pathlib import Path

def check_requirements():
    """Check if all required files and modules are available"""
    print("Checking requirements...")
    
    # Check if human_tag_review.py exists
    if not Path("human_tag_review.py").exists():
        print("❌ ERROR: human_tag_review.py file not found")
        return False
    
    # Check if articles.json exists
    if not Path("articles.json").exists():
        print("❌ ERROR: articles.json file not found")
        return False
    
    # Check if test_human_tag_review.py exists
    if not Path("test_human_tag_review.py").exists():
        print("❌ ERROR: test_human_tag_review.py file not found")
        return False
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ ERROR: streamlit module not found. Install with 'pip install streamlit'")
        return False
    
    print("✅ All requirements satisfied")
    return True

def run_tests():
    """Run the unit tests"""
    print("\nRunning unit tests...")
    # Import the test module
    import test_human_tag_review
    
    # Run the tests
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_human_tag_review)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Check if all tests passed
    if result.wasSuccessful():
        print("✅ All tests passed")
        return True
    else:
        print(f"❌ {len(result.failures) + len(result.errors)} tests failed")
        return False

def verify_functionality():
    """Verify the basic functionality by checking file structure and content"""
    print("\nVerifying basic functionality...")
    
    # Import the human_tag_review module
    import human_tag_review
    
    # Check if the tag system is properly defined
    if not hasattr(human_tag_review, 'ALL_TAGS'):
        print("❌ ALL_TAGS not defined in human_tag_review.py")
        return False
    
    if not hasattr(human_tag_review, 'FLAT_TAGS'):
        print("❌ FLAT_TAGS not defined in human_tag_review.py")
        return False
    
    # Check if the required functions are defined
    required_functions = [
        'load_articles', 
        'load_reviewed_articles', 
        'save_reviewed_article',
        'get_articles_needing_review',
        'display_article',
        'tag_to_flat',
        'flat_to_tag',
        'main'
    ]
    
    for func in required_functions:
        if not hasattr(human_tag_review, func):
            print(f"❌ Required function '{func}' not defined in human_tag_review.py")
            return False
    
    # Count the number of tags in ALL_TAGS
    total_tags = sum(len(tags) for tags in human_tag_review.ALL_TAGS.values())
    if total_tags != len(human_tag_review.FLAT_TAGS):
        print(f"❌ FLAT_TAGS count ({len(human_tag_review.FLAT_TAGS)}) does not match ALL_TAGS count ({total_tags})")
        return False
    
    print("✅ All required components are properly defined")
    
    # Test with sample data
    print("\nTesting with sample data...")
    
    # Load articles
    articles = human_tag_review.load_articles()
    if not articles:
        print("❌ Failed to load articles from articles.json")
        return False
    
    print(f"✅ Successfully loaded {len(articles)} articles")
    
    # Try converting tags
    sample_tag = "military"
    flat_tag = human_tag_review.tag_to_flat(sample_tag)
    if not flat_tag:
        print(f"❌ Failed to convert tag '{sample_tag}' to flat representation")
        return False
    
    recovered_tag = human_tag_review.flat_to_tag(flat_tag)
    if recovered_tag != sample_tag:
        print(f"❌ Tag conversion mismatch: {sample_tag} -> {flat_tag} -> {recovered_tag}")
        return False
    
    print(f"✅ Tag conversion works correctly: {sample_tag} -> {flat_tag} -> {recovered_tag}")
    
    return True

def main():
    """Main function to run all checks and tests"""
    print("=" * 60)
    print("HUMAN TAG REVIEW - TEST SUITE")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues and try again.")
        return 1
    
    # Run tests
    if not run_tests():
        print("\n❌ Unit tests failed. Please fix the issues and try again.")
        return 1
    
    # Verify functionality
    if not verify_functionality():
        print("\n❌ Functionality verification failed. Please fix the issues and try again.")
        return 1
    
    # All checks passed
    print("\n" + "=" * 60)
    print("✅ SUCCESS: All tests passed. The implementation meets the requirements.")
    print("=" * 60)
    print("\nTo run the Streamlit app, use:")
    print("  streamlit run human_tag_review.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 