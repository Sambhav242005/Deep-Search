#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import textwrap
from typing import Any, Dict, List, Tuple, Optional



from backend.constant import MAX_CONTENT_LENGTH,REQUEST_TIMEOUT, SEARCH_PLAN_SCHEMA, SearchResult
from backend.duckduckgo import _search_ddg
from backend.ollama_client import _ask_ollama
from backend.schema_utils import _load_schema
from backend.utility import _gather

# ───────────────────────────── Auto‑planner ───────────────────────────── #

def _auto_plan(question: str, model: str, max_steps: int = 5) -> List[Tuple[str, int, List[str]]]:
    """Generate ≤max_steps focused sub-queries via the LLM with keywords."""
    sys_prompt = (
        "You are a research assistant that breaks down complex questions into focused sub-queries. "
        f"Create up to {max_steps} specific, targeted search queries that will help answer the main question. "
        "For each query, also suggest 2-4 relevant keywords that would help identify the most useful sources. "
        "Return valid JSON matching the provided schema exactly."
    )

    user_prompt = f"""
    Main research question: {question}
    
    Please create a search plan with specific sub-queries that will comprehensively address this question.
    Focus on different aspects or components of the topic.
    """

    try:
        raw = _ask_ollama(model, user_prompt, system=sys_prompt, fmt=SEARCH_PLAN_SCHEMA)
        raw = raw.strip()

        # Handle various response formats
        try:
            plan_json = json.loads(raw)
        except json.JSONDecodeError:
            # Look for JSON array in markdown code blocks or mixed content
            patterns = [
                r'```json\s*(\[.*?\])\s*```',
                r'```\s*(\[.*?\])\s*```', 
                r'(\[.*?\])'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, raw, re.DOTALL)
                if match:
                    plan_json = json.loads(match.group(1))
                    break
            else:
                raise RuntimeError("Auto‑planner did not return parseable JSON")

        # Handle double-encoded JSON
        if isinstance(plan_json, str):
            plan_json = json.loads(plan_json)

        if not isinstance(plan_json, list):
            raise RuntimeError("Auto‑plan JSON is not a list")

        plan: List[Tuple[str, int, List[str]]] = []
        for item in plan_json[:max_steps]:
            try:
                q = str(item["question"]).strip()
                k = int(item.get("num_results", 3))
                keywords = item.get("relevance_keywords", [])
                if isinstance(keywords, list):
                    keywords = [str(kw).strip() for kw in keywords if str(kw).strip()]
                else:
                    keywords = []
                
                if q:
                    plan.append((q, max(1, min(k, 10)), keywords))
            except (KeyError, TypeError, ValueError):
                continue

        return plan or [(question, 5, [])]
        
    except Exception as e:
        print(f"[warn] Auto-planning failed: {e}. Using original question.", file=sys.stderr)
        return [(question, 5, [])]

# ───────────────────────────── Deep Search ────────────────────────────── #

def deep_search(
    question: str, 
    model: str, 
    *, 
    k: int = 5, 
    auto: bool = False,
    schema: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[SearchResult], Optional[List[Tuple[str, int, List[str]]]]]:
    """
    Perform deep search and return answer, sources, and plan (if auto=True).
    
    Returns:
        - answer: LLM response text
        - sources: List of SearchResult objects with metadata
        - plan: Search plan if auto=True, None otherwise
    """
    urls: List[str] = []
    plan_used: Optional[List[Tuple[str, int, List[str]]]] = None
    all_keywords: List[str] = []
    
    if auto:
        plan_used = _auto_plan(question, model)
        print(f"[info] Generated {len(plan_used)} search queries", file=sys.stderr)
        
        for query, num_results, keywords in plan_used:
            print(f"  → {query} (expecting {num_results} results)", file=sys.stderr)
            search_results = _search_ddg(query, num_results)
            urls.extend([r.href for r in search_results])
            all_keywords.extend(keywords)
    else:
        search_results = _search_ddg(question, k)
        urls.extend([r.href for r in search_results])

    # Remove duplicates while preserving order
    unique_urls = list(dict.fromkeys(urls))
    print(f"[info] Fetching {len(unique_urls)} unique URLs", file=sys.stderr)
    
    # Fetch and convert documents with relevance scoring
    docs_with_scores = asyncio.run(_gather(unique_urls, all_keywords))
    
    # Sort by relevance score if we have keywords
    if all_keywords:
        sorted_docs = sorted(docs_with_scores.items(), key=lambda x: x[1][1], reverse=True)
        docs = {url: content for url, (content, score) in sorted_docs}
    else:
        docs = {url: content for url, (content, score) in docs_with_scores.items()}

    if not docs:
        return "I don't know - no documents could be retrieved.", [], plan_used

    # Prepare context for LLM
    docs_section = "\n\n".join(
        f"URL: {url}\n\n{textwrap.shorten(content, MAX_CONTENT_LENGTH)}" 
        for url, content in docs.items()
    )

    # Create system prompt
    system_prompt = (
        "You are a helpful research assistant. Answer questions based strictly on the provided documents. "
        "If the information needed to answer the question is not present in the documents, "
        "respond with 'I don't know' and explain what information is missing."
    )

    # Create user prompt
    user_prompt = (
        f"# QUESTION\n{question}\n\n"
        f"# DOCUMENTS\n{docs_section}\n\n"
        "Please provide a comprehensive answer based on the documents above."
    )

    # Get answer from LLM
    try:
        answer = _ask_ollama(model, user_prompt, system=system_prompt, fmt=schema)
    except Exception as e:
        return f"Error generating answer: {e}", [], plan_used

    # Create SearchResult objects for sources
    sources = [SearchResult("Document", url, "") for url in docs.keys()]
    
    return answer, sources, plan_used

# ───────────────────────────── CLI ──────────────────────────────── #

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Deep search via DuckDuckGo + MarkItDown + Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          %(prog)s "What are the latest developments in AI safety?"
          %(prog)s "Climate change impacts" --auto --model llama3.2
          %(prog)s "Market trends" --schema '{"type": "object", "properties": {"summary": {"type": "string"}}}'
          %(prog)s "Tech news" --schema schema.json --num_results 10
        """)
    )
    
    p.add_argument("question", help="Natural language question or search query")
    p.add_argument("--model", default="llama3.2", help="Ollama model to use (default: llama3.2)")
    p.add_argument("--num_results", "-k", type=int, default=5, 
                   help="Number of search results if --auto is off (default: 5)")
    p.add_argument("--auto", action="store_true", 
                   help="Let LLM generate optimized sub-queries automatically")
    p.add_argument("--schema", help="Path to JSON schema file or raw JSON string for structured output")
    p.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT,
                   help=f"Request timeout in seconds (default: {REQUEST_TIMEOUT})")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = p.parse_args()
    
    # Update global timeout
    REQUEST_TIMEOUT = args.timeout
    
    # Load schema if provided
    output_schema = _load_schema(args.schema) if args.schema else None
    
    try:
        answer, sources, plan = deep_search(
            args.question, 
            args.model, 
            k=args.num_results, 
            auto=args.auto,
            schema=output_schema
        )
        
        print("\n=== ANSWER ===\n")
        print(answer)
        
        if args.auto and plan:
            print("\n=== SEARCH PLAN ===")
            for i, (query, num_results, keywords) in enumerate(plan, 1):
                print(f"{i}. {query} ({num_results} results)")
                if keywords:
                    print(f"   Keywords: {', '.join(keywords)}")
        
        if sources:
            print("\n=== SOURCES ===")
            for i, source in enumerate(sources, 1):
                print(f"{i}. {source.href}")
                if args.verbose and source.snippet:
                    print(f"   {textwrap.shorten(source.snippet, 100)}")
        
    except KeyboardInterrupt:
        print("\n[interrupted] Search cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)