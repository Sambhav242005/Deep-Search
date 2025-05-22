
import asyncio
import io
import mimetypes
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import aiohttp
from bs4 import BeautifulSoup
from markitdown import UnsupportedFormatException

from backend.constant import FETCH_TIMEOUT, MAX_CONTENT_LENGTH
from markitdown import MarkItDown


MKD = MarkItDown()


def _fallback_clean(html: str) -> str:
    """Strip scripts/styles and collapse whitespace (quick & dirty)."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "iframe", "svg"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())

def _calculate_relevance_score(content: str, keywords: List[str]) -> float:
    """Calculate relevance score based on keyword presence."""
    if not keywords:
        return 1.0
    
    content_lower = content.lower()
    score = 0.0
    total_keywords = len(keywords)
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        count = content_lower.count(keyword_lower)
        if count > 0:
            # Logarithmic scoring to avoid over-weighting high-frequency terms
            score += min(1.0, 0.1 + 0.1 * count)
    
    return score / total_keywords if total_keywords > 0 else 1.0

async def _fetch_and_convert(
    session: aiohttp.ClientSession, 
    url: str, 
    timeout: int = FETCH_TIMEOUT,
    keywords: List[str] = None
) -> Tuple[str, str, float]:
    """Download URL → markdown (first N kB) or plain text fallback with relevance scoring."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with session.get(url, timeout=timeout, headers=headers) as resp:
            resp.raise_for_status()
            raw = await resp.read()
            
            # Determine file type for MarkItDown
            content_type = resp.content_type or "text/html"
            suffix = Path(url).suffix or mimetypes.guess_extension(content_type) or ".html"
            filename = f"download{suffix}"
            
            try:
                md = MKD.convert_stream(io.BytesIO(raw), filename=filename, url=url).markdown
            except UnsupportedFormatException:
                md = _fallback_clean(raw.decode(errors="ignore"))
            
            # Truncate content
            content = md[:MAX_CONTENT_LENGTH]
            
            # Calculate relevance score
            relevance = _calculate_relevance_score(content, keywords or [])
            
            return url, content, relevance
            
    except Exception as e:
        print(f"[warn] fetch failed {url}: {e}", file=sys.stderr)
        return url, "", 0.0

async def _gather(urls: List[str], keywords: List[str] = None) -> Dict[str, Tuple[str, float]]:
    """Concurrent fetch/convert helper with relevance scoring."""
    results: Dict[str, Tuple[str, float]] = {}
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT * 2)
    ) as session:
        tasks = [_fetch_and_convert(session, u, keywords=keywords) for u in urls]
        
        for coro in asyncio.as_completed(tasks):
            url, content, relevance = await coro
            if content:
                results[url] = (content, relevance)
    
    return results
