import asyncio
import aiohttp
import logging
import json
import random
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from readability import Document
from langdetect import detect
import time
import uuid
import os

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A372 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Mobile Safari/537.36",
    # ... more ...
]

class RobustAsyncScraper:
    def __init__(self):
        self.logger = logging.getLogger("RobustAsyncScraper")

    async def scrape_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._scrape_one(session, src) for src in sources]
            for coro in asyncio.as_completed(tasks):
                try:
                    res = await coro
                    if res:
                        results.append(res)
                except Exception as e:
                    self.logger.error(f"Scrape error: {e}")
        return results

    async def _scrape_one(self, session, source):
        url = source["url"]
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        try:
            async with session.get(url, headers=headers, timeout=15) as resp:
                if resp.status != 200:
                    self.logger.warning(f"Failed {url}: {resp.status}")
                    return None
                html = await resp.text()
                meta = self.extract_metadata(html, url)
                meta["source_name"] = source.get("name")
                meta["url"] = url
                meta["type"] = source.get("type")
                meta["language"] = self.detect_language(meta.get("content") or html)
                if meta["language"] != "en":
                    meta["needs_translation"] = True
                else:
                    meta["needs_translation"] = False
                meta["raw_html"] = html
                return meta
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return None

    def extract_metadata(self, html: str, url: str) -> Dict[str, Any]:
        try:
            doc = Document(html)
            title = doc.short_title()
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, "html.parser")
            content = soup.get_text("\n", strip=True)
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.string if soup.title else None
            content = soup.get_text("\n", strip=True)
        author = None
        date = None
        # Heuristics for author/date
        for meta in soup.find_all("meta"):
            if meta.get("name") == "author":
                author = meta.get("content")
            if meta.get("property") == "article:published_time":
                date = meta.get("content")
        return {"title": title, "content": content, "author": author, "date": date}

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "unknown"

    @staticmethod
    def load_sources_from_file(path: str) -> List[Dict[str, Any]]:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                return [{"name": line.strip(), "url": line.strip(), "type": "news"} for line in f if line.strip()]
        else:
            raise ValueError("Unsupported source file format") 