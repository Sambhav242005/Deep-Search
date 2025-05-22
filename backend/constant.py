
import os
import time
from typing import Any, Dict, NamedTuple


REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "20"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "8000"))
BASE_OLLAMA = os.getenv("OLLAMA_BASE", "http://localhost:11435")
GEN_ENDPOINT = f"{BASE_OLLAMA}/api/generate"
CHAT_ENDPOINT = f"{BASE_OLLAMA}/api/chat"


SEARCH_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "num_results": {"type": "integer", "minimum": 1, "maximum": 2},
            "relevance_keywords": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "Keywords to prioritize in results"
            }
        },
        "required": ["question", "num_results"],
        "additionalProperties": False,
    },
    "minItems": 1,
    "maxItems": 2,
}

class SearchResult(NamedTuple):
    """Structured search result."""
    title: str
    href: str
    snippet: str = ""

class CircuitBreaker:
    """Simple circuit breaker for external service calls."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def can_call(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        return True  # half-open

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
