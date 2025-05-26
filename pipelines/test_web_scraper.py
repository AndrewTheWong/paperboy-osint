import unittest
import sys
from unittest.mock import patch, MagicMock
import os
import json
from datetime import datetime
import requests

# Add the parent directory to sys.path to import web_scraper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.web_scraper import (
    scrape_taipei_times, 
    scrape_xinhua, 
    scrape_south_china_morning_post,
    scrape_nyt_world,
    scrape_china_daily,
    scrape_globaltimes,
    scrape_all_sources,
    get_random_headers,
    make_request
)

class TestWebScraper(unittest.TestCase):
    
    def setUp(self):
        # Create mock responses for each site
        self.taipei_times_html = """
        <div class="list">
            <h3><a href="/News/front/archives/2023/05/26/2003801234">Test Taipei Times Article</a></h3>
        </div>
        """
        
        self.xinhua_html = """
        <div class="headline-item">
            <h3><a href="/202305/26/c_123456.htm">Test Xinhua Article</a></h3>
        </div>
        """
        
        self.scmp_html = """
        <div class="article-title">
            <a href="/news/china/politics/article/3123456/test-article">Test SCMP Article</a>
        </div>
        """
        
        self.nyt_html = """
        <article>
            <h2><a href="/2023/05/26/world/asia/test-article.html">Test NYT Article</a></h2>
        </article>
        """
        
        self.china_daily_html = """
        <div class="item">
            <h3><a href="/a/202305/26/WS123456.html">Test China Daily Article</a></h3>
        </div>
        """
        
        self.globaltimes_html = """
        <div class="list_text">
            <a class="list_link" href="/page/test-article-123456.html">Test Global Times Article</a>
        </div>
        """
    
    def test_get_random_headers(self):
        """Test that random headers contain required fields."""
        headers = get_random_headers()
        self.assertIn('User-Agent', headers)
        self.assertIn('Accept', headers)
        
    @patch('pipelines.web_scraper.requests.get')
    def test_make_request(self, mock_get):
        """Test the request functionality with mocked response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Test successful request
        result = make_request('https://example.com')
        self.assertEqual(result, mock_response)
        
        # Test failed request
        mock_get.side_effect = requests.exceptions.RequestException("Test error")
        result = make_request('https://example.com', max_retries=1)
        self.assertIsNone(result)
    
    @patch('pipelines.web_scraper.make_request')
    def test_scrape_taipei_times(self, mock_make_request):
        """Test scraping Taipei Times."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = self.taipei_times_html
        mock_make_request.return_value = mock_response
        
        articles = scrape_taipei_times()
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Test Taipei Times Article')
        self.assertEqual(articles[0]['source'], 'Taipei Times')
        self.assertTrue(articles[0]['url'].startswith('https://www.taipeitimes.com'))
        self.assertIn('scraped_at', articles[0])
    
    @patch('pipelines.web_scraper.make_request')
    def test_scrape_xinhua(self, mock_make_request):
        """Test scraping Xinhua."""
        mock_response = MagicMock()
        mock_response.content = self.xinhua_html
        mock_make_request.return_value = mock_response
        
        articles = scrape_xinhua()
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Test Xinhua Article')
        self.assertEqual(articles[0]['source'], 'Xinhua')
        self.assertIn('scraped_at', articles[0])
    
    @patch('pipelines.web_scraper.make_request')
    def test_scrape_scmp(self, mock_make_request):
        """Test scraping South China Morning Post."""
        mock_response = MagicMock()
        mock_response.content = self.scmp_html
        mock_make_request.return_value = mock_response
        
        articles = scrape_south_china_morning_post()
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Test SCMP Article')
        self.assertEqual(articles[0]['source'], 'South China Morning Post')
        self.assertIn('scraped_at', articles[0])
    
    @patch('pipelines.web_scraper.make_request')
    def test_scrape_nyt(self, mock_make_request):
        """Test scraping New York Times."""
        mock_response = MagicMock()
        mock_response.content = self.nyt_html
        mock_make_request.return_value = mock_response
        
        articles = scrape_nyt_world()
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Test NYT Article')
        self.assertEqual(articles[0]['source'], 'New York Times')
        self.assertIn('scraped_at', articles[0])
        
    @patch('pipelines.web_scraper.make_request')
    def test_scrape_china_daily(self, mock_make_request):
        """Test scraping China Daily."""
        mock_response = MagicMock()
        mock_response.content = self.china_daily_html
        mock_make_request.return_value = mock_response
        
        articles = scrape_china_daily()
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Test China Daily Article')
        self.assertEqual(articles[0]['source'], 'China Daily')
        self.assertIn('scraped_at', articles[0])
        
    @patch('pipelines.web_scraper.make_request')
    def test_scrape_globaltimes(self, mock_make_request):
        """Test scraping Global Times."""
        mock_response = MagicMock()
        mock_response.content = self.globaltimes_html
        mock_make_request.return_value = mock_response
        
        articles = scrape_globaltimes()
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Test Global Times Article')
        self.assertEqual(articles[0]['source'], 'Global Times')
        self.assertIn('scraped_at', articles[0])
    
    @patch('pipelines.web_scraper.scrape_taipei_times')
    @patch('pipelines.web_scraper.scrape_xinhua')
    @patch('pipelines.web_scraper.scrape_south_china_morning_post')
    @patch('pipelines.web_scraper.scrape_nyt_world')
    @patch('pipelines.web_scraper.scrape_china_daily')
    @patch('pipelines.web_scraper.scrape_globaltimes')
    def test_scrape_all_sources(self, mock_globaltimes, mock_china_daily, mock_nyt, mock_scmp, mock_xinhua, mock_taipei):
        """Test the aggregate scraper function."""
        # Setup mock returns for each source
        mock_taipei.return_value = [{'title': 'Test Taipei Article', 'source': 'Taipei Times', 'url': 'https://example.com/1', 'scraped_at': datetime.utcnow().isoformat()}]
        mock_xinhua.return_value = [{'title': 'Test Xinhua Article', 'source': 'Xinhua', 'url': 'https://example.com/2', 'scraped_at': datetime.utcnow().isoformat()}]
        mock_scmp.return_value = [{'title': 'Test SCMP Article', 'source': 'South China Morning Post', 'url': 'https://example.com/3', 'scraped_at': datetime.utcnow().isoformat()}]
        mock_nyt.return_value = [{'title': 'Test NYT Article', 'source': 'New York Times', 'url': 'https://example.com/4', 'scraped_at': datetime.utcnow().isoformat()}]
        mock_china_daily.return_value = [{'title': 'Test China Daily Article', 'source': 'China Daily', 'url': 'https://example.com/5', 'scraped_at': datetime.utcnow().isoformat()}]
        mock_globaltimes.return_value = [{'title': 'Test Global Times Article', 'source': 'Global Times', 'url': 'https://example.com/6', 'scraped_at': datetime.utcnow().isoformat()}]
        
        # Run the test
        all_articles = scrape_all_sources()
        
        # Verify all sources were called
        mock_taipei.assert_called_once()
        mock_xinhua.assert_called_once()
        mock_scmp.assert_called_once()
        mock_nyt.assert_called_once()
        mock_china_daily.assert_called_once()
        mock_globaltimes.assert_called_once()
        
        # Verify we got all articles
        self.assertEqual(len(all_articles), 6)
        sources = [article['source'] for article in all_articles]
        self.assertIn('Taipei Times', sources)
        self.assertIn('Xinhua', sources)
        self.assertIn('South China Morning Post', sources)
        self.assertIn('New York Times', sources)
        self.assertIn('China Daily', sources)
        self.assertIn('Global Times', sources)
    
    @patch('pipelines.web_scraper.make_request')
    def test_error_handling(self, mock_make_request):
        """Test error handling when a request fails."""
        # Setup mock to return None (failed request)
        mock_make_request.return_value = None
        
        # Test that we get an empty list rather than an exception
        articles = scrape_taipei_times()
        self.assertEqual(articles, [])

if __name__ == '__main__':
    unittest.main() 