"""
Implementation of various string matching algorithms for comparison with Rabin-Karp
"""

import time
from typing import List, Dict, Tuple

class StringMatchingAlgorithms:
    """Collection of string matching algorithms for performance comparison."""
    
    def __init__(self):
        self.statistics = {}
    
    def naive_search(self, text: str, pattern: str) -> Dict:
        """
        Naive (Brute Force) string matching algorithm.
        
        Time Complexity: O(n*m)
        Space Complexity: O(1)
        """
        start_time = time.perf_counter()
        matches = []
        comparisons = 0
        
        n, m = len(text), len(pattern)
        
        for i in range(n - m + 1):
            j = 0
            while j < m:
                comparisons += 1
                if text[i + j] != pattern[j]:
                    break
                j += 1
            
            if j == m:  # Pattern found
                matches.append(i)
        
        end_time = time.perf_counter()
        
        return {
            'algorithm': 'Naive (Brute Force)',
            'matches': matches,
            'time': end_time - start_time,
            'comparisons': comparisons,
            'space_complexity': 'O(1)',
            'time_complexity_avg': 'O(n*m)',
            'time_complexity_worst': 'O(n*m)'
        }
    
    def kmp_search(self, text: str, pattern: str) -> Dict:
        """
        Knuth-Morris-Pratt (KMP) string matching algorithm.
        
        Time Complexity: O(n+m)
        Space Complexity: O(m)
        """
        start_time = time.perf_counter()
        
        def compute_lps(pattern):
            """Compute Longest Proper Prefix which is also Suffix array."""
            m = len(pattern)
            lps = [0] * m
            length = 0
            i = 1
            
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps
        
        matches = []
        comparisons = 0
        n, m = len(text), len(pattern)
        
        if m == 0:
            end_time = time.perf_counter()
            return {
                'algorithm': 'KMP (Knuth-Morris-Pratt)',
                'matches': matches,
                'time': end_time - start_time,
                'comparisons': comparisons,
                'space_complexity': 'O(m)',
                'time_complexity_avg': 'O(n+m)',
                'time_complexity_worst': 'O(n+m)'
            }
        
        # Compute LPS array
        lps = compute_lps(pattern)
        
        i = j = 0  # i for text, j for pattern
        
        while i < n:
            comparisons += 1
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        end_time = time.perf_counter()
        
        return {
            'algorithm': 'KMP (Knuth-Morris-Pratt)',
            'matches': matches,
            'time': end_time - start_time,
            'comparisons': comparisons,
            'space_complexity': 'O(m)',
            'time_complexity_avg': 'O(n+m)',
            'time_complexity_worst': 'O(n+m)'
        }
    
    def boyer_moore_search(self, text: str, pattern: str) -> Dict:
        """
        Boyer-Moore string matching algorithm (simplified version).
        
        Time Complexity: O(n*m) worst case, O(n/m) best case
        Space Complexity: O(σ) where σ is alphabet size
        """
        start_time = time.perf_counter()
        
        def bad_character_heuristic(pattern):
            """Create bad character table."""
            bad_char = {}
            m = len(pattern)
            
            for i in range(m):
                bad_char[pattern[i]] = i
            
            return bad_char
        
        matches = []
        comparisons = 0
        n, m = len(text), len(pattern)
        
        if m == 0:
            end_time = time.perf_counter()
            return {
                'algorithm': 'Boyer-Moore (Simplified)',
                'matches': matches,
                'time': end_time - start_time,
                'comparisons': comparisons,
                'space_complexity': 'O(σ)',
                'time_complexity_avg': 'O(n/m)',
                'time_complexity_worst': 'O(n*m)'
            }
        
        bad_char = bad_character_heuristic(pattern)
        
        shift = 0
        while shift <= n - m:
            j = m - 1
            
            # Match pattern from right to left
            while j >= 0:
                comparisons += 1
                if pattern[j] != text[shift + j]:
                    break
                j -= 1
            
            if j < 0:  # Pattern found
                matches.append(shift)
                # Shift pattern to align with next character
                shift += (m - bad_char.get(text[shift + m], -1) - 1) if shift + m < n else 1
            else:
                # Shift pattern based on bad character heuristic
                shift += max(1, j - bad_char.get(text[shift + j], -1))
        
        end_time = time.perf_counter()
        
        return {
            'algorithm': 'Boyer-Moore (Simplified)',
            'matches': matches,
            'time': end_time - start_time,
            'comparisons': comparisons,
            'space_complexity': 'O(σ)',
            'time_complexity_avg': 'O(n/m)',
            'time_complexity_worst': 'O(n*m)'
        }
    
    def z_algorithm_search(self, text: str, pattern: str) -> Dict:
        """
        Z Algorithm for string matching.
        
        Time Complexity: O(n+m)
        Space Complexity: O(n+m)
        """
        start_time = time.perf_counter()
        
        def z_function(s):
            """Compute Z array for string s."""
            n = len(s)
            z = [0] * n
            l, r = 0, 0
            
            for i in range(1, n):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
            
            return z
        
        matches = []
        comparisons = 0
        n, m = len(text), len(pattern)
        
        if m == 0:
            end_time = time.perf_counter()
            return {
                'algorithm': 'Z Algorithm',
                'matches': matches,
                'time': end_time - start_time,
                'comparisons': comparisons,
                'space_complexity': 'O(n+m)',
                'time_complexity_avg': 'O(n+m)',
                'time_complexity_worst': 'O(n+m)'
            }
        
        # Create combined string: pattern + '$' + text
        combined = pattern + '$' + text
        z_array = z_function(combined)
        
        # Count comparisons (approximation for Z algorithm)
        comparisons = len(combined)
        
        # Find matches
        for i in range(m + 1, len(combined)):
            if z_array[i] == m:
                matches.append(i - m - 1)
        
        end_time = time.perf_counter()
        
        return {
            'algorithm': 'Z Algorithm',
            'matches': matches,
            'time': end_time - start_time,
            'comparisons': comparisons,
            'space_complexity': 'O(n+m)',
            'time_complexity_avg': 'O(n+m)',
            'time_complexity_worst': 'O(n+m)'
        }
    
    def compare_all_algorithms(self, text: str, pattern: str) -> Dict:
        """
        Compare all string matching algorithms on the same input.
        
        Args:
            text (str): Text to search in
            pattern (str): Pattern to search for
        
        Returns:
            Dict: Comparison results for all algorithms
        """
        algorithms = [
            self.naive_search,
            self.kmp_search,
            self.boyer_moore_search,
            self.z_algorithm_search
        ]
        
        results = {}
        
        for algorithm in algorithms:
            try:
                result = algorithm(text, pattern)
                results[result['algorithm']] = result
            except Exception as e:
                # Handle any algorithm-specific errors
                results[algorithm.__name__] = {
                    'error': str(e),
                    'matches': [],
                    'time': float('inf'),
                    'comparisons': 0
                }
        
        return results

class BenchmarkSuite:
    """Comprehensive benchmarking suite for string matching algorithms."""
    
    def __init__(self):
        self.algorithms = StringMatchingAlgorithms()
    
    def generate_test_cases(self) -> List[Tuple[str, str, str]]:
        """Generate various test cases for benchmarking."""
        test_cases = []
        
        # Best case scenarios
        test_cases.append((
            "Best Case - No Match",
            "a" * 1000,
            "b"
        ))
        
        test_cases.append((
            "Best Case - Single Match at End",
            "a" * 999 + "b",
            "b"
        ))
        
        # Average case scenarios
        test_cases.append((
            "Average Case - Random Text",
            "the quick brown fox jumps over the lazy dog " * 20,
            "fox"
        ))
        
        test_cases.append((
            "Average Case - DNA Sequence",
            "ATCGATCGATCG" * 50 + "AAAA" + "ATCGATCGATCG" * 50,
            "AAAA"
        ))
        
        # Worst case scenarios
        test_cases.append((
            "Worst Case - Many Partial Matches",
            "aaa" * 100 + "aab",
            "aaab"
        ))
        
        test_cases.append((
            "Worst Case - Repeated Pattern",
            "abcabc" * 100,
            "abcabcabc"
        ))
        
        # Edge cases
        test_cases.append((
            "Edge Case - Empty Pattern",
            "hello world",
            ""
        ))
        
        test_cases.append((
            "Edge Case - Pattern Longer than Text",
            "short",
            "this is a very long pattern"
        ))
        
        test_cases.append((
            "Edge Case - Single Character",
            "a" * 1000,
            "a"
        ))
        
        return test_cases
    
    def run_benchmark(self, include_rabin_karp=True) -> Dict:
        """
        Run comprehensive benchmark on all algorithms.
        
        Args:
            include_rabin_karp (bool): Whether to include Rabin-Karp in comparison
        
        Returns:
            Dict: Benchmark results
        """
        test_cases = self.generate_test_cases()
        results = {}
        
        for case_name, text, pattern in test_cases:
            print(f"Running benchmark: {case_name}")
            
            case_results = self.algorithms.compare_all_algorithms(text, pattern)
            
            # Add Rabin-Karp if requested
            if include_rabin_karp and pattern:  # Skip empty pattern for RK
                try:
                    from rabin_karp import RabinKarp
                    
                    start_time = time.perf_counter()
                    rk = RabinKarp()
                    rk_result = rk.search(text, pattern)
                    end_time = time.perf_counter()
                    
                    case_results['Rabin-Karp'] = {
                        'algorithm': 'Rabin-Karp',
                        'matches': rk_result['matches'],
                        'time': end_time - start_time,
                        'comparisons': rk_result['statistics']['comparisons'],
                        'space_complexity': 'O(1)',
                        'time_complexity_avg': 'O(n+m)',
                        'time_complexity_worst': 'O(n*m)',
                        'hash_calculations': rk_result['statistics']['hash_calculations'],
                        'spurious_hits': rk_result['statistics']['spurious_hits']
                    }
                except ImportError:
                    pass
            
            results[case_name] = {
                'text_length': len(text),
                'pattern_length': len(pattern),
                'algorithms': case_results
            }
        
        return results
    
    def analyze_performance_trends(self, results: Dict) -> Dict:
        """
        Analyze performance trends across different test cases.
        
        Args:
            results (Dict): Benchmark results
        
        Returns:
            Dict: Performance analysis
        """
        analysis = {
            'algorithm_rankings': {},
            'best_case_performance': {},
            'worst_case_performance': {},
            'scalability_analysis': {},
            'recommendations': []
        }
        
        # Collect performance data
        algorithm_times = {}
        algorithm_comparisons = {}
        
        for case_name, case_data in results.items():
            for algo_name, algo_result in case_data['algorithms'].items():
                if 'error' in algo_result:
                    continue
                
                if algo_name not in algorithm_times:
                    algorithm_times[algo_name] = []
                    algorithm_comparisons[algo_name] = []
                
                algorithm_times[algo_name].append(algo_result['time'])
                algorithm_comparisons[algo_name].append(algo_result['comparisons'])
        
        # Calculate average performance
        for algo_name in algorithm_times:
            avg_time = sum(algorithm_times[algo_name]) / len(algorithm_times[algo_name])
            avg_comparisons = sum(algorithm_comparisons[algo_name]) / len(algorithm_comparisons[algo_name])
            
            analysis['algorithm_rankings'][algo_name] = {
                'average_time': avg_time,
                'average_comparisons': avg_comparisons,
                'consistency': max(algorithm_times[algo_name]) / min(algorithm_times[algo_name]) if min(algorithm_times[algo_name]) > 0 else float('inf')
            }
        
        # Generate recommendations
        fastest_algo = min(analysis['algorithm_rankings'].items(), 
                          key=lambda x: x[1]['average_time'])
        
        most_efficient_algo = min(analysis['algorithm_rankings'].items(), 
                                 key=lambda x: x[1]['average_comparisons'])
        
        analysis['recommendations'] = [
            f"Fastest Algorithm: {fastest_algo[0]} (avg: {fastest_algo[1]['average_time']:.6f}s)",
            f"Most Efficient: {most_efficient_algo[0]} (avg: {most_efficient_algo[1]['average_comparisons']:.0f} comparisons)",
            "Rabin-Karp is best for: Multiple pattern search, large alphabets",
            "KMP is best for: Guaranteed O(n+m) performance, small patterns",
            "Boyer-Moore is best for: Large patterns, natural language text",
            "Z Algorithm is best for: Preprocessing requirements, pattern analysis"
        ]
        
        return analysis