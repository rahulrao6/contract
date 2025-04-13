"""
Helper functions for Contract Analyzer.
"""

import re
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

def create_unique_id(prefix: str = "", length: int = 8) -> str:
    """
    Create a unique identifier.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of random part
        
    Returns:
        Unique identifier
    """
    random_part = uuid.uuid4().hex[:length]
    if prefix:
        return f"{prefix}-{random_part}"
    return random_part

def clean_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: Text with potential HTML tags
        
    Returns:
        Clean text without HTML tags
    """
    if not text:
        return ""
    return re.sub(r'<[^>]+>', '', text)

def truncate_text(text: str, max_length: int = 100, add_ellipsis: bool = True) -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        add_ellipsis: Whether to add '...' when truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    truncated = text[:max_length].rstrip()
    if add_ellipsis:
        truncated += "..."
    
    return truncated

def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    Create hash of text.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hashed text
    """
    if algorithm.lower() == "md5":
        return hashlib.md5(text.encode()).hexdigest()
    elif algorithm.lower() == "sha1":
        return hashlib.sha1(text.encode()).hexdigest()
    else:
        return hashlib.sha256(text.encode()).hexdigest()

def format_timestamp(timestamp: Optional[Union[str, datetime]] = None, 
                     format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp as string.
    
    Args:
        timestamp: Timestamp to format (defaults to now)
        format_str: Format string
        
    Returns:
        Formatted timestamp
    """
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            return timestamp  # Return as-is if can't parse
    
    return timestamp.strftime(format_str)

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deeply merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
            
    return result