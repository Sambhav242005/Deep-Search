import json
import sys
from typing import Any, Dict, Optional


def _load_schema(schema_arg: str) -> Optional[Dict[str, Any]]:
    """Load JSON schema from file path or raw JSON string."""
    if not schema_arg:
        return None
    
    try:
        # Try parsing as JSON first
        return json.loads(schema_arg)
    except json.JSONDecodeError:
        # Try loading as file path
        try:
            with open(schema_arg, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[warn] Could not load schema: {e}", file=sys.stderr)
            return None