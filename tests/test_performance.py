"""
Performance and stress tests for the Rabin-Karp project
"""

import unittest
import time
import sys
import os
import psutil
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rabin_karp import RabinKarp, MultiPatternRabinKarp
from algorithms import StringMatchingAlgorithms
from hash_functions import get_hash_function

class TestPerformance(unittest.TestCase):
    """Performance tests for string matching algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rk = RabinKarp()
        self.algorithms = StringMatchingAlgorithms()
        self.performance_threshold = 1.0  # 1 second max for most tests
    
    def test_large_text_performance(self):
        """Test performance with large text inputs."""
        # Generate large text (1MB)
        large_text = "a" * 500000 + "pattern" + "b" * 500000
        pattern = "pattern"
        
        start_time = time.perf_counter()
        result = self.rk.search(large_text, pattern)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Should find the pattern
        self.assertEqual(result['matches'], [500000])
        
        # Should complete in reasonable time
        self.assertLess(execution_time, self.performance_threshold)
        
        # Should be efficient in comparisons
        text_length = len(large_text)
        pattern_length = len(pattern)
        max_comparisons = text_length * pattern_length
        actual_comparisons = result['statistics']['comparisons']
        
        # Rabin-Karp should do much fewer comparisons than naive
        self.assertLess(actual_comparisons, max_comparisons / 10)
    
    def test_many_patterns_performance(self):
        """Test performance with many patterns."""
        text = "the quick brown fox jumps over the lazy dog " * 100
        patterns = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"] * 10
        
        multi_rk = MultiPatternRabinKarp()
        
        start_time = time.perf_counter()
        result = multi_rk.search(text, patterns)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Should complete in reasonable time
        self.assertLess(execution_time, self.performance_threshold)
        
        # Should find matches for all patterns
        for pattern in set(patterns):  # Remove duplicates
            self.assertIn(pattern, result['matches'])
            self.assertGreater(len(result['matches'][pattern]), 0)
    
    def test_worst_case_performance(self):
        """Test performance in worst-case scenarios."""
        # Worst case: many spurious hits
        rk = RabinKarp(base=2, prime=3)  # Small prime increases collisions
        
        text = "a" * 1000 + "b"
        pattern = "ab"
        
        start_time = time.perf_counter()
        result = rk.search(text, pattern)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Should still find correct match
        self.assertEqual(result['matches'], [999])
        
        # Should complete even in worst case
        self.assertLess(execution_time, self.performance_threshold * 2)  # Allow more time for worst case
    
    def test_memory_usage(self):
        """Test memory usage during algorithm execution."""
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run algorithm with large input
        large_text = "x" * 100000
        pattern = "pattern"
        
        result = self.rk.search(large_text, pattern)
        
        # Get memory usage after execution
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 10MB for this test)
        max_memory_increase = 10 * 1024 * 1024  # 10MB
        self.assertLess(memory_increase, max_memory_increase)
    
    def test_algorithm_comparison_performance(self):
        """Test performance comparison between algorithms."""
        text = "performance test text " * 1000
        pattern = "test"
        
        # Time each algorithm
        algorithms_to_test = [
            ('Rabin-Karp', lambda: self.rk.search(text, pattern)),
            ('Naive', lambda: self.algorithms.naive_search(text, pattern)),
            ('KMP', lambda: self.algorithms.kmp_search(text, pattern)),
            ('Boyer-Moore', lambda: self.algorithms.boyer_moore_search(text, pattern)),
            ('Z Algorithm', lambda: self.algorithms.z_algorithm_search(text, pattern))
        ]
        
        results = {}
        
        for algo_name, algo_func in algorithms_to_test:
            start_time = time.perf_counter()
            result = algo_func()
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            results[algo_name] = {
                'time': execution_time,
                'matches': result['matches'] if 'matches' in result else result.get('matches', [])
            }
            
            # Each algorithm should complete in reasonable time
            self.assertLess(execution_time, self.performance_threshold)
        
        # All algorithms should find the same matches
        expected_matches = results['Naive']['matches']
        for algo_name, result in results.items():
            self.assertEqual(result['matches'], expected_matches, 
                           f"{algo_name} found different matches")
    
    def test_hash_function_performance(self):
        """Test performance of different hash functions."""
        text = "hash function performance test " * 1000
        pattern = "performance"
        
        hash_types = ["polynomial", "simple", "djb2", "fnv"]
        
        for hash_type in hash_types:
            with self.subTest(hash_type=hash_type):
                rk = RabinKarp(hash_type=hash_type)
                
                start_time = time.perf_counter()
                result = rk.search(text, pattern)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                
                # Should complete in reasonable time
                self.assertLess(execution_time, self.performance_threshold)
                
                # Should find correct matches
                self.assertGreater(len(result['matches']), 0)
    
    def test_scalability(self):
        """Test algorithm scalability with increasing input sizes."""
        pattern = "scale"
        base_text = "scalability test text "
        
        sizes = [1000, 5000, 10000, 20000]
        times = []
        
        for size in sizes:
            # Generate text of specified size
            multiplier = size // len(base_text) + 1
            text = (base_text * multiplier)[:size]
            
            start_time = time.perf_counter()
            result = self.rk.search(text, pattern)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            
            # Should complete in reasonable time
            self.assertLess(execution_time, self.performance_threshold)
        
        # Time should scale reasonably (not exponentially)
        # Check that doubling input size doesn't increase time by more than 4x
        for i in range(1, len(times)):
            if times[i-1] > 0:  # Avoid division by zero
                time_ratio = times[i] / times[i-1]
                size_ratio = sizes[i] / sizes[i-1]
                
                # Time growth should be reasonable relative to size growth
                self.assertLess(time_ratio, size_ratio * 2, 
                              f"Poor scalability: time ratio {time_ratio} for size ratio {size_ratio}")

class TestStressTests(unittest.TestCase):
    """Stress tests for robustness and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rk = RabinKarp()
    
    def test_extreme_input_sizes(self):
        """Test with extremely large inputs."""
        # Very large text
        huge_text = "x" * 1000000  # 1 million characters
        pattern = "y"
        
        start_time = time.perf_counter()
        result = self.rk.search(huge_text, pattern)
        end_time = time.perf_counter()
        
        # Should handle gracefully
        self.assertEqual(result['matches'], [])
        self.assertLess(end_time - start_time, 5.0)  # Should complete within 5 seconds
    
    def test_many_matches_stress(self):
        """Test with input that has many matches."""
        # Text with pattern repeated many times
        pattern = "ab"
        text = pattern * 10000  # 10,000 occurrences
        
        start_time = time.perf_counter()
        result = self.rk.search(text, pattern)
        end_time = time.perf_counter()
        
        # Should find all matches
        expected_matches = list(range(0, len(text) - len(pattern) + 1, len(pattern)))
        self.assertEqual(result['matches'], expected_matches)
        
        # Should complete in reasonable time
        self.assertLess(end_time - start_time, 2.0)
    
    def test_unicode_text_stress(self):
        """Test with Unicode text."""
        # Unicode text with various characters
        unicode_text = "Hello 世界 🌍 Ñoël café résumé naïve Москва العالم"
        pattern = "世界"
        
        result = self.rk.search(unicode_text, pattern)
        
        # Should handle Unicode correctly
        self.assertEqual(result['matches'], [6])
    
    def test_concurrent_searches(self):
        """Test concurrent algorithm execution."""
        text = "concurrent search test " * 1000
        patterns = ["concurrent", "search", "test", "missing"]
        
        results = {}
        threads = []
        
        def search_pattern(pattern):
            result = self.rk.search(text, pattern)
            results[pattern] = result
        
        # Start concurrent searches
        for pattern in patterns:
            thread = threading.Thread(target=search_pattern, args=(pattern,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)  # 5 second timeout
            self.assertFalse(thread.is_alive(), "Thread did not complete in time")
        
        # Check results
        self.assertEqual(len(results), len(patterns))
        for pattern in patterns:
            self.assertIn(pattern, results)
            if pattern != "missing":
                self.assertGreater(len(results[pattern]['matches']), 0)
    
    def test_edge_case_parameters(self):
        """Test with edge case parameters."""
        text = "edge case test"
        pattern = "test"
        
        # Test with minimal parameters
        rk_min = RabinKarp(base=2, prime=3)
        result_min = rk_min.search(text, pattern)
        self.assertEqual(result_min['matches'], [10])
        
        # Test with large parameters
        rk_large = RabinKarp(base=1000, prime=997)
        result_large = rk_large.search(text, pattern)
        self.assertEqual(result_large['matches'], [10])
    
    def test_memory_leak_detection(self):
        """Test for potential memory leaks."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Perform many searches
        for i in range(100):
            text = f"memory test iteration {i} " * 100
            pattern = "test"
            result = self.rk.search(text, pattern)
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 5MB)
        max_increase = 5 * 1024 * 1024  # 5MB
        self.assertLess(memory_increase, max_increase, 
                       f"Potential memory leak: {memory_increase} bytes increase")

class TestRobustnessTests(unittest.TestCase):
    """Robustness tests for error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rk = RabinKarp()
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Test with None inputs (should not crash)
        try:
            result = self.rk.search(None, "pattern")
            # If it doesn't crash, result should indicate no matches
            self.assertEqual(result['matches'], [])
        except (TypeError, AttributeError):
            # It's acceptable to raise an exception for None input
            pass
        
        try:
            result = self.rk.search("text", None)
            self.assertEqual(result['matches'], [])
        except (TypeError, AttributeError):
            pass
    
    def test_special_characters(self):
        """Test with special characters and symbols."""
        special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        pattern = "@#$"
        
        result = self.rk.search(special_text, pattern)
        self.assertEqual(result['matches'], [1])
    
    def test_very_long_pattern(self):
        """Test with very long patterns."""
        text = "short text"
        long_pattern = "a" * 1000
        
        result = self.rk.search(text, long_pattern)
        self.assertEqual(result['matches'], [])
    
    def test_repeated_character_patterns(self):
        """Test with patterns of repeated characters."""
        text = "aaaaaaaaaa"
        pattern = "aaa"
        
        result = self.rk.search(text, pattern)
        expected = [0, 1, 2, 3, 4, 5, 6, 7]  # All possible positions
        self.assertEqual(result['matches'], expected)

if __name__ == '__main__':
    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("Warning: psutil not available. Memory tests will be skipped.")
        # Remove memory-related test methods
        delattr(TestPerformance, 'test_memory_usage')
        delattr(TestStressTests, 'test_memory_leak_detection')
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestPerformance))
    test_suite.addTest(unittest.makeSuite(TestStressTests))
    test_suite.addTest(unittest.makeSuite(TestRobustnessTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Performance & Stress Tests Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")