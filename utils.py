"""
Utility functions for the Rabin-Karp string matching project.
"""

def calculate_hash(text, length, base=256, prime=101):
    """
    Calculate hash value for a string of given length.
    
    Args:
        text (str): Input text
        length (int): Length of substring to hash
        base (int): Base for polynomial hash
        prime (int): Prime number for modular arithmetic
    
    Returns:
        int: Hash value
    """
    hash_value = 0
    for i in range(length):
        hash_value = (hash_value * base + ord(text[i])) % prime
    return hash_value

def calculate_power(base, length, prime):
    """
    Calculate base^(length-1) % prime efficiently.
    
    Args:
        base (int): Base value
        length (int): Pattern length
        prime (int): Prime modulus
    
    Returns:
        int: base^(length-1) % prime
    """
    power = 1
    for _ in range(length - 1):
        power = (power * base) % prime
    return power

def rolling_hash(old_hash, old_char, new_char, power, base=256, prime=101):
    """
    Calculate new hash using rolling hash technique.
    
    Args:
        old_hash (int): Previous hash value
        old_char (str): Character being removed
        new_char (str): Character being added
        power (int): base^(pattern_length-1) % prime
        base (int): Base for polynomial hash
        prime (int): Prime number for modular arithmetic
    
    Returns:
        int: New hash value
    """
    # Remove old character and add new character
    new_hash = (old_hash - ord(old_char) * power) % prime
    new_hash = (new_hash * base + ord(new_char)) % prime
    return new_hash

def format_matches(matches, text, pattern):
    """
    Format match results for display.
    
    Args:
        matches (list): List of match positions
        text (str): Original text
        pattern (str): Search pattern
    
    Returns:
        list: Formatted match information
    """
    formatted_matches = []
    for pos in matches:
        context_start = max(0, pos - 10)
        context_end = min(len(text), pos + len(pattern) + 10)
        context = text[context_start:context_end]
        
        formatted_matches.append({
            'position': pos,
            'match': text[pos:pos + len(pattern)],
            'context': context,
            'context_start': context_start
        })
    
    return formatted_matches