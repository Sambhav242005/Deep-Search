
import random
import sys
import time
from typing import List
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS, exceptions as ddg_exc

import urllib.parse as _url

import requests

from backend.constant import CircuitBreaker, SearchResult


def _scrape_ddg_html(query: str, k: int) -> List[SearchResult]:
    """Fallback HTML scraper if the API keeps throttling."""
    encoded = _url.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={encoded}&kl=us-en"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SearchBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        results: List[SearchResult] = []
        for result in soup.select(".result")[:k]:
            title_elem = result.select_one(".result__title a")
            url_elem = result.select_one(".result__url")
            snippet_elem = result.select_one(".result__snippet")
            
            if title_elem and url_elem:
                title = title_elem.get_text(strip=True)
                href = url_elem.get("href", "")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                results.append(SearchResult(title, href, snippet))
        
        return results
        
    except Exception as e:
        print(f"[warn] HTML scrape failed: {e}", file=sys.stderr)
        return []

# Circuit breaker for DuckDuckGo
_ddg_breaker = CircuitBreaker()

def _search_ddg(query: str, k: int = 5) -> List[SearchResult]:
    """DuckDuckGo search with enhanced rate‑limit handling and circuit breaker."""
    if not _ddg_breaker.can_call():
        print("[warn] DuckDuckGo circuit breaker open, skipping search", file=sys.stderr)
        return []
    
    attempts = 5
    base_backoff = 1.0
    
    for attempt in range(attempts):
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=k))
                results = [
                    SearchResult(r.get("title", ""), r.get("href", ""), r.get("body", ""))
                    for r in raw_results
                ]
                _ddg_breaker.record_success()
                return results
                
        except ddg_exc.RatelimitException:
            # Exponential backoff with jitter
            backoff = base_backoff * (2 ** attempt) + random.uniform(0, 1)
            print(f"[rate-limit] DuckDuckGo throttled (attempt {attempt + 1}/{attempts}); sleeping {backoff:.1f}s", file=sys.stderr)
            time.sleep(backoff)
            
        except Exception as e:
            _ddg_breaker.record_failure()
            if attempt == attempts - 1:  # Last attempt
                print(f"[error] DuckDuckGo search failed after {attempts} attempts: {e}", file=sys.stderr)
                break
            time.sleep(base_backoff * (attempt + 1))
    
    # Fallback to HTML scraping
    print("[info] Falling back to HTML scrape", file=sys.stderr)
    return _scrape_ddg_html(query, k)
