"""
Unit tests for all string matching algorithms
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import StringMatchingAlgorithms, BenchmarkSuite

class TestStringMatchingAlgorithms(unittest.TestCase):
    """Test cases for all string matching algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.algorithms = StringMatchingAlgorithms()
        self.test_cases = [
            ("hello world", "world", [6]),
            ("abcabcabc", "abc", [0, 3, 6]),
            ("aaaaaaa", "aa", [0, 1, 2, 3, 4, 5]),
            ("", "test", []),
            ("test", "", []),
            ("short", "verylongpattern", []),
            ("The Quick Brown Fox", "Fox", [16]),
            ("mississippi", "issi", [1, 4]),
            ("ABABDABACDABABCABCABCABC", "ABABCAB", [15]),
        ]
    
    def test_naive_algorithm(self):
        """Test naive (brute force) algorithm."""
        for text, pattern, expected in self.test_cases:
            with self.subTest(text=text, pattern=pattern):
                result = self.algorithms.naive_search(text, pattern)
                
                self.assertEqual(result['matches'], expected)
                self.assertEqual(result['algorithm'], 'Naive (Brute Force)')
                self.assertIn('time', result)
                self.assertIn('comparisons', result)
                self.assertGreaterEqual(result['comparisons'], 0)
    
    def test_kmp_algorithm(self):
        """Test KMP (Knuth-Morris-Pratt) algorithm."""
        for text, pattern, expected in self.test_cases:
            with self.subTest(text=text, pattern=pattern):
                result = self.algorithms.kmp_search(text, pattern)
                
                self.assertEqual(result['matches'], expected)
                self.assertEqual(result['algorithm'], 'KMP (Knuth-Morris-Pratt)')
                self.assertIn('time', result)
                self.assertIn('comparisons', result)
                self.assertGreaterEqual(result['comparisons'], 0)
    
    def test_boyer_moore_algorithm(self):
        """Test Boyer-Moore algorithm."""
        for text, pattern, expected in self.test_cases:
            with self.subTest(text=text, pattern=pattern):
                result = self.algorithms.boyer_moore_search(text, pattern)
                
                self.assertEqual(result['matches'], expected)
                self.assertEqual(result['algorithm'], 'Boyer-Moore (Simplified)')
                self.assertIn('time', result)
                self.assertIn('comparisons', result)
                self.assertGreaterEqual(result['comparisons'], 0)
    
    def test_z_algorithm(self):
        """Test Z Algorithm."""
        for text, pattern, expected in self.test_cases:
            with self.subTest(text=text, pattern=pattern):
                result = self.algorithms.z_algorithm_search(text, pattern)
                
                self.assertEqual(result['matches'], expected)
                self.assertEqual(result['algorithm'], 'Z Algorithm')
                self.assertIn('time', result)
                self.assertIn('comparisons', result)
                self.assertGreaterEqual(result['comparisons'], 0)
    
    def test_algorithm_consistency(self):
        """Test that all algorithms return the same matches."""
        test_cases = [
            ("programming algorithms", "algorithm"),
            ("abcdefghijklmnop", "def"),
            ("aaabaaabaaab", "aab"),
            ("hello world hello", "hello"),
        ]
        
        for text, pattern in test_cases:
            with self.subTest(text=text, pattern=pattern):
                naive_result = self.algorithms.naive_search(text, pattern)
                kmp_result = self.algorithms.kmp_search(text, pattern)
                bm_result = self.algorithms.boyer_moore_search(text, pattern)
                z_result = self.algorithms.z_algorithm_search(text, pattern)
                
                # All algorithms should find the same matches
                expected_matches = naive_result['matches']
                self.assertEqual(kmp_result['matches'], expected_matches, "KMP mismatch")
                self.assertEqual(bm_result['matches'], expected_matches, "Boyer-Moore mismatch")
                self.assertEqual(z_result['matches'], expected_matches, "Z Algorithm mismatch")
    
    def test_compare_all_algorithms(self):
        """Test the compare_all_algorithms function."""
        text = "test string for algorithm comparison"
        pattern = "algorithm"
        
        results = self.algorithms.compare_all_algorithms(text, pattern)
        
        # Should return results for all algorithms
        expected_algorithms = [
            'Naive (Brute Force)',
            'KMP (Knuth-Morris-Pratt)',
            'Boyer-Moore (Simplified)',
            'Z Algorithm'
        ]
        
        for algo_name in expected_algorithms:
            self.assertIn(algo_name, results)
            result = results[algo_name]
            
            # Each result should have required fields
            self.assertIn('matches', result)
            self.assertIn('time', result)
            self.assertIn('comparisons', result)
            self.assertIn('algorithm', result)
            
            # All should find the same matches
            self.assertEqual(result['matches'], [11])
    
    def test_empty_pattern_handling(self):
        """Test how algorithms handle empty patterns."""
        text = "test text"
        pattern = ""
        
        results = self.algorithms.compare_all_algorithms(text, pattern)
        
        # All algorithms should handle empty pattern gracefully
        for algo_name, result in results.items():
            if 'error' not in result:
                self.assertEqual(result['matches'], [])
    
    def test_pattern_longer_than_text(self):
        """Test algorithms with pattern longer than text."""
        text = "short"
        pattern = "this is a very long pattern"
        
        results = self.algorithms.compare_all_algorithms(text, pattern)
        
        # All algorithms should return no matches
        for algo_name, result in results.items():
            if 'error' not in result:
                self.assertEqual(result['matches'], [])
    
    def test_performance_characteristics(self):
        """Test performance characteristics of different algorithms."""
        # Create test case that might show performance differences
        text = "a" * 1000 + "pattern" + "b" * 1000
        pattern = "pattern"
        
        results = self.algorithms.compare_all_algorithms(text, pattern)
        
        # All should find the pattern at position 1000
        expected_matches = [1000]
        
        for algo_name, result in results.items():
            if 'error' not in result:
                self.assertEqual(result['matches'], expected_matches)
                
                # Performance should be reasonable
                self.assertLess(result['time'], 1.0)  # Should complete in under 1 second
                self.assertGreater(result['comparisons'], 0)  # Should do some work

class TestBenchmarkSuite(unittest.TestCase):
    """Test cases for the benchmark suite."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = BenchmarkSuite()
    
    def test_generate_test_cases(self):
        """Test test case generation."""
        test_cases = self.benchmark.generate_test_cases()
        
        # Should generate multiple test cases
        self.assertGreater(len(test_cases), 5)
        
        # Each test case should have proper structure
        for case_name, text, pattern in test_cases:
            self.assertIsInstance(case_name, str)
            self.assertIsInstance(text, str)
            self.assertIsInstance(pattern, str)
            self.assertGreater(len(case_name), 0)
    
    def test_run_benchmark_without_rabin_karp(self):
        """Test running benchmark without Rabin-Karp."""
        # Run a limited benchmark for testing
        original_generate = self.benchmark.generate_test_cases
        
        # Override to generate fewer test cases for faster testing
        def limited_test_cases():
            return [
                ("Test Case 1", "hello world", "world"),
                ("Test Case 2", "abcabc", "abc")
            ]
        
        self.benchmark.generate_test_cases = limited_test_cases
        
        try:
            results = self.benchmark.run_benchmark(include_rabin_karp=False)
            
            # Should have results for test cases
            self.assertGreater(len(results), 0)
            
            # Each result should have proper structure
            for case_name, case_data in results.items():
                self.assertIn('text_length', case_data)
                self.assertIn('pattern_length', case_data)
                self.assertIn('algorithms', case_data)
                
                # Should have results for multiple algorithms
                self.assertGreater(len(case_data['algorithms']), 1)
        
        finally:
            # Restore original method
            self.benchmark.generate_test_cases = original_generate
    
    def test_analyze_performance_trends(self):
        """Test performance trend analysis."""
        # Create mock results for testing
        mock_results = {
            "Test Case 1": {
                "text_length": 100,
                "pattern_length": 3,
                "algorithms": {
                    "Algorithm A": {
                        "time": 0.001,
                        "comparisons": 50
                    },
                    "Algorithm B": {
                        "time": 0.002,
                        "comparisons": 75
                    }
                }
            },
            "Test Case 2": {
                "text_length": 200,
                "pattern_length": 3,
                "algorithms": {
                    "Algorithm A": {
                        "time": 0.002,
                        "comparisons": 100
                    },
                    "Algorithm B": {
                        "time": 0.004,
                        "comparisons": 150
                    }
                }
            }
        }
        
        analysis = self.benchmark.analyze_performance_trends(mock_results)
        
        # Should have proper analysis structure
        self.assertIn('algorithm_rankings', analysis)
        self.assertIn('recommendations', analysis)
        
        # Should have rankings for both algorithms
        rankings = analysis['algorithm_rankings']
        self.assertIn('Algorithm A', rankings)
        self.assertIn('Algorithm B', rankings)
        
        # Each ranking should have required metrics
        for algo_name, metrics in rankings.items():
            self.assertIn('average_time', metrics)
            self.assertIn('average_comparisons', metrics)
            self.assertIn('consistency', metrics)
        
        # Should have recommendations
        self.assertIsInstance(analysis['recommendations'], list)
        self.assertGreater(len(analysis['recommendations']), 0)

class TestAlgorithmSpecificCases(unittest.TestCase):
    """Test algorithm-specific edge cases and optimizations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.algorithms = StringMatchingAlgorithms()
    
    def test_kmp_failure_function(self):
        """Test KMP with patterns that exercise the failure function."""
        # Pattern with repeating prefix-suffix
        text = "ababcababa"
        pattern = "ababa"
        expected = [5]
        
        result = self.algorithms.kmp_search(text, pattern)
        self.assertEqual(result['matches'], expected)
        
        # Should be efficient due to failure function
        # KMP should do fewer comparisons than naive for this case
        naive_result = self.algorithms.naive_search(text, pattern)
        self.assertLessEqual(result['comparisons'], naive_result['comparisons'])
    
    def test_boyer_moore_bad_character(self):
        """Test Boyer-Moore with cases that exercise bad character heuristic."""
        # Text where bad character heuristic should help
        text = "abcdefghijklmnop"
        pattern = "nop"
        expected = [13]
        
        result = self.algorithms.boyer_moore_search(text, pattern)
        self.assertEqual(result['matches'], expected)
    
    def test_z_algorithm_preprocessing(self):
        """Test Z Algorithm with patterns that benefit from preprocessing."""
        # Pattern that appears multiple times
        text = "abcabcabcabc"
        pattern = "abc"
        expected = [0, 3, 6, 9]
        
        result = self.algorithms.z_algorithm_search(text, pattern)
        self.assertEqual(result['matches'], expected)
    
    def test_worst_case_scenarios(self):
        """Test algorithms on worst-case inputs."""
        # Worst case for naive algorithm
        text = "a" * 100 + "b"
        pattern = "a" * 10 + "b"
        expected = [90]
        
        results = self.algorithms.compare_all_algorithms(text, pattern)
        
        # All algorithms should find the match
        for algo_name, result in results.items():
            if 'error' not in result:
                self.assertEqual(result['matches'], expected)
        
        # KMP and Z Algorithm should be more efficient than naive
        naive_comparisons = results['Naive (Brute Force)']['comparisons']
        kmp_comparisons = results['KMP (Knuth-Morris-Pratt)']['comparisons']
        z_comparisons = results['Z Algorithm']['comparisons']
        
        self.assertLessEqual(kmp_comparisons, naive_comparisons)
        self.assertLessEqual(z_comparisons, naive_comparisons)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestStringMatchingAlgorithms))
    test_suite.addTest(unittest.makeSuite(TestBenchmarkSuite))
    test_suite.addTest(unittest.makeSuite(TestAlgorithmSpecificCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Algorithm Tests Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")