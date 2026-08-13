import json
from typing import Dict, Any

def state_to_json(state_dict: Dict[str, Any]) -> str:
    """Helper to convert workflow state dictionary to formatted JSON string."""
    return json.dumps(state_dict, indent=2, default=str)

def format_file_size(size_in_bytes: int) -> str:
    """Format bytes to human readable KB / MB."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    else:
        return f"{size_in_bytes / (1024 * 1024):.1f} MB"
