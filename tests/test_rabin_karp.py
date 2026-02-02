"""
Unit tests for Rabin-Karp algorithm implementation
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rabin_karp import RabinKarp, MultiPatternRabinKarp, rabin_karp_search

class TestRabinKarp(unittest.TestCase):
    """Test cases for Rabin-Karp algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rk = RabinKarp()
        self.test_cases = [
            ("hello world", "world", [6]),
            ("abcabcabc", "abc", [0, 3, 6]),
            ("aaaaaaa", "aa", [0, 1, 2, 3, 4, 5]),
            ("", "test", []),
            ("test", "", []),
            ("short", "verylongpattern", []),
            ("case sensitive", "CASE", []),
            ("The Quick Brown Fox", "Quick", [4]),
        ]
    
    def test_basic_search(self):
        """Test basic pattern matching functionality."""
        for text, pattern, expected in self.test_cases:
            with self.subTest(text=text, pattern=pattern):
                result = self.rk.search(text, pattern)
                self.assertEqual(result['matches'], expected, 
                               f"Failed for text='{text}', pattern='{pattern}'")
    
    def test_case_sensitivity(self):
        """Test case sensitive and insensitive search."""
        text = "Hello World Hello"
        pattern = "hello"
        
        # Case sensitive (should find no matches)
        result_sensitive = self.rk.search(text, pattern, case_sensitive=True)
        self.assertEqual(result_sensitive['matches'], [])
        
        # Case insensitive (should find matches)
        result_insensitive = self.rk.search(text, pattern, case_sensitive=False)
        self.assertEqual(result_insensitive['matches'], [0, 12])
    
    def test_different_hash_functions(self):
        """Test algorithm with different hash function types."""
        text = "abcdefghijklmnop"
        pattern = "def"
        expected = [3]
        
        hash_types = ["polynomial", "simple", "djb2", "fnv"]
        
        for hash_type in hash_types:
            with self.subTest(hash_type=hash_type):
                rk = RabinKarp(hash_type=hash_type)
                result = rk.search(text, pattern)
                self.assertEqual(result['matches'], expected,
                               f"Hash function {hash_type} failed")
    
    def test_different_parameters(self):
        """Test algorithm with different base and prime parameters."""
        text = "testing parameters"
        pattern = "test"
        expected = [0]
        
        parameters = [
            (256, 101),
            (128, 103),
            (512, 107),
            (33, 109)
        ]
        
        for base, prime in parameters:
            with self.subTest(base=base, prime=prime):
                rk = RabinKarp(base=base, prime=prime)
                result = rk.search(text, pattern)
                self.assertEqual(result['matches'], expected,
                               f"Parameters base={base}, prime={prime} failed")
    
    def test_statistics_tracking(self):
        """Test that algorithm statistics are properly tracked."""
        text = "abcdefghijk"
        pattern = "def"
        
        result = self.rk.search(text, pattern)
        stats = result['statistics']
        
        # Check that statistics are present and reasonable
        self.assertIn('comparisons', stats)
        self.assertIn('hash_calculations', stats)
        self.assertIn('spurious_hits', stats)
        
        self.assertGreaterEqual(stats['comparisons'], 0)
        self.assertGreaterEqual(stats['hash_calculations'], 0)
        self.assertGreaterEqual(stats['spurious_hits'], 0)
    
    def test_hash_values_tracking(self):
        """Test that hash values are properly tracked for visualization."""
        text = "abcdef"
        pattern = "cd"
        
        result = self.rk.search(text, pattern)
        hash_values = result['hash_values']
        
        # Should have hash values for each position
        expected_positions = len(text) - len(pattern) + 1
        self.assertEqual(len(hash_values), expected_positions)
        
        # Each hash value entry should have required fields
        for hash_info in hash_values:
            self.assertIn('position', hash_info)
            self.assertIn('hash', hash_info)
            self.assertIn('pattern_hash', hash_info)
            self.assertIn('match', hash_info)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty text
        result = self.rk.search("", "pattern")
        self.assertEqual(result['matches'], [])
        
        # Empty pattern
        result = self.rk.search("text", "")
        self.assertEqual(result['matches'], [])
        
        # Pattern longer than text
        result = self.rk.search("short", "verylongpattern")
        self.assertEqual(result['matches'], [])
        
        # Single character text and pattern
        result = self.rk.search("a", "a")
        self.assertEqual(result['matches'], [0])
        
        # Pattern at the very end
        result = self.rk.search("abcdef", "ef")
        self.assertEqual(result['matches'], [4])
        
        # Pattern at the very beginning
        result = self.rk.search("abcdef", "ab")
        self.assertEqual(result['matches'], [0])
    
    def test_spurious_hits_detection(self):
        """Test detection and handling of spurious hits."""
        # Create a scenario likely to produce spurious hits
        # Using small prime to increase collision probability
        rk = RabinKarp(base=2, prime=3)
        
        text = "abcdefghijklmnop"
        pattern = "xyz"  # Pattern not in text
        
        result = rk.search(text, pattern)
        
        # Should find no matches despite potential hash collisions
        self.assertEqual(result['matches'], [])
        
        # Spurious hits might be detected
        self.assertGreaterEqual(result['statistics']['spurious_hits'], 0)
    
    def test_convenience_function(self):
        """Test the convenience function rabin_karp_search."""
        text = "test convenience function"
        pattern = "convenience"
        expected = [5]
        
        result = rabin_karp_search(text, pattern)
        self.assertEqual(result['matches'], expected)
        
        # Test with parameters
        result = rabin_karp_search(text, pattern, base=128, prime=103)
        self.assertEqual(result['matches'], expected)

class TestMultiPatternRabinKarp(unittest.TestCase):
    """Test cases for multi-pattern Rabin-Karp algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.multi_rk = MultiPatternRabinKarp()
    
    def test_multiple_patterns_search(self):
        """Test searching for multiple patterns simultaneously."""
        text = "the quick brown fox jumps over the lazy dog"
        patterns = ["the", "fox", "dog", "cat"]
        
        result = self.multi_rk.search(text, patterns)
        
        # Check that all patterns are in results
        for pattern in patterns:
            self.assertIn(pattern, result['matches'])
        
        # Check specific expected matches
        self.assertEqual(result['matches']['the'], [0, 31])
        self.assertEqual(result['matches']['fox'], [16])
        self.assertEqual(result['matches']['dog'], [40])
        self.assertEqual(result['matches']['cat'], [])  # Not in text
    
    def test_empty_patterns_list(self):
        """Test behavior with empty patterns list."""
        text = "some text"
        patterns = []
        
        result = self.multi_rk.search(text, patterns)
        self.assertEqual(result['matches'], {})
    
    def test_duplicate_patterns(self):
        """Test handling of duplicate patterns."""
        text = "test duplicate patterns"
        patterns = ["test", "test", "duplicate"]
        
        result = self.multi_rk.search(text, patterns)
        
        # Should handle duplicates gracefully
        self.assertIn("test", result['matches'])
        self.assertIn("duplicate", result['matches'])
        self.assertEqual(result['matches']['test'], [0])
        self.assertEqual(result['matches']['duplicate'], [5])
    
    def test_overlapping_matches(self):
        """Test patterns that overlap in the text."""
        text = "ababab"
        patterns = ["ab", "ba", "abab"]
        
        result = self.multi_rk.search(text, patterns)
        
        self.assertEqual(result['matches']['ab'], [0, 2, 4])
        self.assertEqual(result['matches']['ba'], [1, 3])
        self.assertEqual(result['matches']['abab'], [0, 2])
    
    def test_efficiency_metrics(self):
        """Test that efficiency metrics are calculated."""
        text = "efficiency test text"
        patterns = ["test", "text", "missing"]
        
        result = self.multi_rk.search(text, patterns)
        
        # Check that efficiency metrics are present
        self.assertIn('efficiency_metrics', result)
        metrics = result['efficiency_metrics']
        
        expected_keys = [
            'total_patterns', 'patterns_with_matches', 'total_matches_found',
            'average_pattern_length', 'match_success_rate', 'efficiency_improvement',
            'comparisons_per_pattern'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)

class TestRabinKarpPerformance(unittest.TestCase):
    """Performance-related tests for Rabin-Karp algorithm."""
    
    def test_large_text_performance(self):
        """Test algorithm performance with large text."""
        # Generate large text
        large_text = "a" * 10000 + "pattern" + "b" * 10000
        pattern = "pattern"
        
        rk = RabinKarp()
        result = rk.search(large_text, pattern)
        
        # Should find the pattern
        self.assertEqual(result['matches'], [10000])
        
        # Should be reasonably efficient
        stats = result['statistics']
        self.assertLess(stats['comparisons'], len(large_text) * len(pattern))
    
    def test_many_matches_performance(self):
        """Test performance when there are many matches."""
        # Text with many occurrences of pattern
        text = "abc" * 1000  # "abcabcabc..."
        pattern = "abc"
        
        rk = RabinKarp()
        result = rk.search(text, pattern)
        
        # Should find all matches
        expected_matches = list(range(0, len(text) - len(pattern) + 1, 3))
        self.assertEqual(result['matches'], expected_matches)
    
    def test_worst_case_scenario(self):
        """Test worst-case scenario with many spurious hits."""
        # Create scenario with potential for spurious hits
        rk = RabinKarp(base=2, prime=3)  # Small prime increases collisions
        
        text = "a" * 100 + "b"
        pattern = "ab"
        
        result = rk.search(text, pattern)
        
        # Should still find correct match
        self.assertEqual(result['matches'], [99])

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestRabinKarp))
    test_suite.addTest(unittest.makeSuite(TestMultiPatternRabinKarp))
    test_suite.addTest(unittest.makeSuite(TestRabinKarpPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")