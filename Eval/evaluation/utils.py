import re

def remove_tags(text):
    """
    Remove all text enclosed in angle brackets <>
    and strip surrounding whitespace.
    
    Args:
        text (str): Input string with potential tags
        
    Returns:
        str: Cleaned text without tags and extra whitespace
    """
    # Remove any content within <>
    cleaned = re.sub(r'<[^>]+>', '', text)
    # Remove leading/trailing whitespace and normalize newlines
    return cleaned.strip()