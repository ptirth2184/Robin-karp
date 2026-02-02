"""
Rabin-Karp String Matching Algorithm Implementation
"""

from utils import calculate_hash, calculate_power, rolling_hash

class RabinKarp:
    def __init__(self, base=256, prime=101):
        """
        Initialize Rabin-Karp algorithm with hash parameters.
        
        Args:
            base (int): Base for polynomial hash function
            prime (int): Prime number for modular arithmetic
        """
        self.base = base
        self.prime = prime
        self.comparisons = 0  # Track number of character comparisons
        self.hash_calculations = 0  # Track hash calculations
        self.spurious_hits = 0  # Track false positives
    
    def search(self, text, pattern, case_sensitive=True):
        """
        Search for pattern in text using Rabin-Karp algorithm.
        
        Args:
            text (str): Text to search in
            pattern (str): Pattern to search for
            case_sensitive (bool): Whether search is case sensitive
        
        Returns:
            dict: Search results with matches and statistics
        """
        # Reset statistics
        self.comparisons = 0
        self.hash_calculations = 0
        self.spurious_hits = 0
        
        # Handle case sensitivity
        if not case_sensitive:
            text = text.lower()
            pattern = pattern.lower()
        
        # Edge cases
        if len(pattern) > len(text) or len(pattern) == 0:
            return {
                'matches': [],
                'statistics': self._get_statistics(),
                'hash_values': []
            }
        
        pattern_length = len(pattern)
        text_length = len(text)
        matches = []
        hash_values = []  # Store hash values for visualization
        
        # Calculate hash of pattern and first window of text
        pattern_hash = calculate_hash(pattern, pattern_length, self.base, self.prime)
        text_hash = calculate_hash(text, pattern_length, self.base, self.prime)
        self.hash_calculations += 2
        
        # Calculate power for rolling hash
        power = calculate_power(self.base, pattern_length, self.prime)
        
        # Store initial hash value
        hash_values.append({
            'position': 0,
            'hash': text_hash,
            'pattern_hash': pattern_hash,
            'match': text_hash == pattern_hash
        })
        
        # Check first window
        if text_hash == pattern_hash:
            if self._verify_match(text, pattern, 0):
                matches.append(0)
            else:
                self.spurious_hits += 1
        
        # Slide the pattern over text one by one
        for i in range(1, text_length - pattern_length + 1):
            # Calculate hash for current window using rolling hash
            text_hash = rolling_hash(
                text_hash, 
                text[i - 1], 
                text[i + pattern_length - 1], 
                power, 
                self.base, 
                self.prime
            )
            self.hash_calculations += 1
            
            # Store hash value for visualization
            hash_values.append({
                'position': i,
                'hash': text_hash,
                'pattern_hash': pattern_hash,
                'match': text_hash == pattern_hash
            })
            
            # Check if hash values match
            if text_hash == pattern_hash:
                if self._verify_match(text, pattern, i):
                    matches.append(i)
                else:
                    self.spurious_hits += 1
        
        return {
            'matches': matches,
            'statistics': self._get_statistics(),
            'hash_values': hash_values
        }
    
    def _verify_match(self, text, pattern, position):
        """
        Verify if there's an actual match at given position.
        
        Args:
            text (str): Text to search in
            pattern (str): Pattern to match
            position (int): Position to check
        
        Returns:
            bool: True if match is verified
        """
        for i in range(len(pattern)):
            self.comparisons += 1
            if text[position + i] != pattern[i]:
                return False
        return True
    
    def _get_statistics(self):
        """
        Get algorithm performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        return {
            'comparisons': self.comparisons,
            'hash_calculations': self.hash_calculations,
            'spurious_hits': self.spurious_hits
        }

def rabin_karp_search(text, pattern, case_sensitive=True, base=256, prime=101):
    """
    Convenience function for Rabin-Karp search.
    
    Args:
        text (str): Text to search in
        pattern (str): Pattern to search for
        case_sensitive (bool): Whether search is case sensitive
        base (int): Base for hash function
        prime (int): Prime for modular arithmetic
    
    Returns:
        dict: Search results
    """
    rk = RabinKarp(base, prime)
    return rk.search(text, pattern, case_sensitive)