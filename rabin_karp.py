"""
Rabin-Karp String Matching Algorithm Implementation
"""

from utils import calculate_hash, calculate_power, rolling_hash
from hash_functions import get_hash_function

class RabinKarp:
    def __init__(self, base=256, prime=101, hash_type="polynomial"):
        """
        Initialize Rabin-Karp algorithm with hash parameters.
        
        Args:
            base (int): Base for polynomial hash function
            prime (int): Prime number for modular arithmetic
            hash_type (str): Type of hash function to use
        """
        self.base = base
        self.prime = prime
        self.hash_type = hash_type
        self.hash_function = get_hash_function(hash_type, base, prime)
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
        pattern_hash = self.hash_function.calculate_hash(pattern, pattern_length)
        text_hash = self.hash_function.calculate_hash(text, pattern_length)
        self.hash_calculations += 2
        
        # Calculate power for rolling hash
        power = self.hash_function.calculate_power(pattern_length)
        
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
            text_hash = self.hash_function.rolling_hash(
                text_hash, 
                text[i - 1], 
                text[i + pattern_length - 1], 
                power
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

def rabin_karp_search(text, pattern, case_sensitive=True, base=256, prime=101, hash_type="polynomial"):
    """
    Convenience function for Rabin-Karp search with hash function selection.
    
    Args:
        text (str): Text to search in
        pattern (str): Pattern to search for
        case_sensitive (bool): Whether search is case sensitive
        base (int): Base for hash function
        prime (int): Prime for modular arithmetic
        hash_type (str): Type of hash function to use
    
    Returns:
        dict: Search results
    """
    rk = RabinKarp(base, prime, hash_type)
    return rk.search(text, pattern, case_sensitive)
    def search_multiple_patterns(self, text, patterns, case_sensitive=True):
        """
        Search for multiple patterns in text using Rabin-Karp algorithm.
        
        Args:
            text (str): Text to search in
            patterns (list): List of patterns to search for
            case_sensitive (bool): Whether search is case sensitive
        
        Returns:
            dict: Search results for all patterns
        """
        # Reset statistics
        self.comparisons = 0
        self.hash_calculations = 0
        self.spurious_hits = 0
        
        # Handle case sensitivity
        if not case_sensitive:
            text = text.lower()
            patterns = [p.lower() for p in patterns]
        
        # Remove empty patterns and duplicates
        patterns = list(set([p for p in patterns if p]))
        
        if not patterns:
            return {'matches': {}, 'statistics': self._get_statistics()}
        
        # Group patterns by length for efficiency
        patterns_by_length = {}
        for pattern in patterns:
            length = len(pattern)
            if length <= len(text):
                if length not in patterns_by_length:
                    patterns_by_length[length] = []
                patterns_by_length[length].append(pattern)
        
        all_matches = {}
        all_hash_values = {}
        
        # Search for each group of patterns with same length
        for pattern_length, pattern_group in patterns_by_length.items():
            # Calculate hashes for all patterns in this group
            pattern_hashes = {}
            for pattern in pattern_group:
                pattern_hash = self.hash_function.calculate_hash(pattern, pattern_length)
                if pattern_hash not in pattern_hashes:
                    pattern_hashes[pattern_hash] = []
                pattern_hashes[pattern_hash].append(pattern)
                self.hash_calculations += 1
            
            # Search through text for this pattern length
            matches_for_length, hash_values_for_length = self._search_by_length(
                text, pattern_length, pattern_hashes
            )
            
            # Merge results
            for pattern in pattern_group:
                all_matches[pattern] = matches_for_length.get(pattern, [])
                all_hash_values[pattern] = hash_values_for_length
        
        return {
            'matches': all_matches,
            'statistics': self._get_statistics(),
            'hash_values': all_hash_values
        }
    
    def _search_by_length(self, text, pattern_length, pattern_hashes):
        """
        Search for patterns of specific length.
        
        Args:
            text (str): Text to search in
            pattern_length (int): Length of patterns
            pattern_hashes (dict): Hash -> patterns mapping
        
        Returns:
            tuple: (matches dict, hash values list)
        """
        text_length = len(text)
        matches = {pattern: [] for patterns in pattern_hashes.values() for pattern in patterns}
        hash_values = []
        
        if pattern_length > text_length:
            return matches, hash_values
        
        # Calculate hash of first window
        text_hash = self.hash_function.calculate_hash(text, pattern_length)
        self.hash_calculations += 1
        
        # Calculate power for rolling hash
        power = self.hash_function.calculate_power(pattern_length)
        
        # Store initial hash value
        hash_values.append({
            'position': 0,
            'hash': text_hash,
            'matches': []
        })
        
        # Check first window
        if text_hash in pattern_hashes:
            for pattern in pattern_hashes[text_hash]:
                if self._verify_match(text, pattern, 0):
                    matches[pattern].append(0)
                    hash_values[-1]['matches'].append(pattern)
                else:
                    self.spurious_hits += 1
        
        # Slide through the rest of the text
        for i in range(1, text_length - pattern_length + 1):
            # Calculate rolling hash
            text_hash = self.hash_function.rolling_hash(
                text_hash,
                text[i - 1],
                text[i + pattern_length - 1],
                power
            )
            self.hash_calculations += 1
            
            # Store hash value
            hash_info = {
                'position': i,
                'hash': text_hash,
                'matches': []
            }
            
            # Check for matches
            if text_hash in pattern_hashes:
                for pattern in pattern_hashes[text_hash]:
                    if self._verify_match(text, pattern, i):
                        matches[pattern].append(i)
                        hash_info['matches'].append(pattern)
                    else:
                        self.spurious_hits += 1
            
            hash_values.append(hash_info)
        
        return matches, hash_values

class MultiPatternRabinKarp:
    """
    Specialized class for efficient multiple pattern matching.
    """
    
    def __init__(self, base=256, prime=101, hash_type="polynomial"):
        """Initialize multi-pattern Rabin-Karp."""
        self.base = base
        self.prime = prime
        self.hash_type = hash_type
        self.hash_function = get_hash_function(hash_type, base, prime)
        self.statistics = {
            'comparisons': 0,
            'hash_calculations': 0,
            'spurious_hits': 0
        }
    
    def search(self, text, patterns, case_sensitive=True):
        """
        Optimized search for multiple patterns.
        
        Args:
            text (str): Text to search in
            patterns (list): List of patterns to search for
            case_sensitive (bool): Whether search is case sensitive
        
        Returns:
            dict: Comprehensive search results
        """
        # Reset statistics
        self.statistics = {
            'comparisons': 0,
            'hash_calculations': 0,
            'spurious_hits': 0
        }
        
        # Handle case sensitivity
        if not case_sensitive:
            text = text.lower()
            patterns = [p.lower() for p in patterns]
        
        # Remove empty and duplicate patterns
        unique_patterns = list(set([p for p in patterns if p and len(p) <= len(text)]))
        
        if not unique_patterns:
            return {
                'matches': {},
                'pattern_stats': {},
                'overall_stats': self.statistics,
                'efficiency_metrics': {}
            }
        
        # Find all matches
        all_matches = {}
        pattern_stats = {}
        
        for pattern in unique_patterns:
            rk = RabinKarp(self.base, self.prime, self.hash_type)
            result = rk.search(text, pattern, case_sensitive)
            
            all_matches[pattern] = result['matches']
            pattern_stats[pattern] = {
                'matches_found': len(result['matches']),
                'comparisons': result['statistics']['comparisons'],
                'hash_calculations': result['statistics']['hash_calculations'],
                'spurious_hits': result['statistics']['spurious_hits']
            }
            
            # Accumulate statistics
            self.statistics['comparisons'] += result['statistics']['comparisons']
            self.statistics['hash_calculations'] += result['statistics']['hash_calculations']
            self.statistics['spurious_hits'] += result['statistics']['spurious_hits']
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            text, unique_patterns, all_matches, pattern_stats
        )
        
        return {
            'matches': all_matches,
            'pattern_stats': pattern_stats,
            'overall_stats': self.statistics,
            'efficiency_metrics': efficiency_metrics
        }
    
    def _calculate_efficiency_metrics(self, text, patterns, matches, pattern_stats):
        """Calculate efficiency metrics for multiple pattern search."""
        total_patterns = len(patterns)
        total_matches = sum(len(match_list) for match_list in matches.values())
        patterns_with_matches = sum(1 for match_list in matches.values() if match_list)
        
        avg_pattern_length = sum(len(p) for p in patterns) / total_patterns if patterns else 0
        
        # Theoretical vs actual comparisons
        naive_comparisons = sum(
            len(text) * len(pattern) for pattern in patterns
        )
        actual_comparisons = self.statistics['comparisons']
        
        efficiency_improvement = (
            (naive_comparisons - actual_comparisons) / naive_comparisons * 100
            if naive_comparisons > 0 else 0
        )
        
        return {
            'total_patterns': total_patterns,
            'patterns_with_matches': patterns_with_matches,
            'total_matches_found': total_matches,
            'average_pattern_length': round(avg_pattern_length, 2),
            'match_success_rate': round(patterns_with_matches / total_patterns * 100, 2) if total_patterns > 0 else 0,
            'efficiency_improvement': round(efficiency_improvement, 2),
            'comparisons_per_pattern': round(actual_comparisons / total_patterns, 2) if total_patterns > 0 else 0
        }
