"""
Unit tests for hash function implementations
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hash_functions import (
    PolynomialHash, SimpleHash, DJB2Hash, FNVHash,
    get_hash_function, compare_hash_functions, calculate_uniformity
)

class TestHashFunctions(unittest.TestCase):
    """Test cases for hash function implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_strings = [
            "hello",
            "world",
            "test",
            "algorithm",
            "hash",
            "function",
            "rabin",
            "karp"
        ]
        self.base = 256
        self.prime = 101
    
    def test_polynomial_hash(self):
        """Test polynomial hash function."""
        poly_hash = PolynomialHash(self.base, self.prime)
        
        # Test basic hash calculation
        test_string = "hello"
        hash_val = poly_hash.calculate_hash(test_string, len(test_string))
        
        # Hash should be within expected range
        self.assertGreaterEqual(hash_val, 0)
        self.assertLess(hash_val, self.prime)
        
        # Same string should produce same hash
        hash_val2 = poly_hash.calculate_hash(test_string, len(test_string))
        self.assertEqual(hash_val, hash_val2)
        
        # Different strings should (usually) produce different hashes
        hash_val3 = poly_hash.calculate_hash("world", 5)
        self.assertNotEqual(hash_val, hash_val3)
    
    def test_simple_hash(self):
        """Test simple additive hash function."""
        simple_hash = SimpleHash(self.base, self.prime)
        
        test_string = "test"
        hash_val = simple_hash.calculate_hash(test_string, len(test_string))
        
        # Hash should be within expected range
        self.assertGreaterEqual(hash_val, 0)
        self.assertLess(hash_val, self.prime)
        
        # Verify deterministic behavior
        hash_val2 = simple_hash.calculate_hash(test_string, len(test_string))
        self.assertEqual(hash_val, hash_val2)
    
    def test_djb2_hash(self):
        """Test DJB2 hash function."""
        djb2_hash = DJB2Hash()
        
        test_string = "djb2"
        hash_val = djb2_hash.calculate_hash(test_string, len(test_string))
        
        # Hash should be within expected range
        self.assertGreaterEqual(hash_val, 0)
        self.assertLess(hash_val, djb2_hash.prime)
        
        # Test consistency
        hash_val2 = djb2_hash.calculate_hash(test_string, len(test_string))
        self.assertEqual(hash_val, hash_val2)
    
    def test_fnv_hash(self):
        """Test FNV hash function."""
        fnv_hash = FNVHash()
        
        test_string = "fnv"
        hash_val = fnv_hash.calculate_hash(test_string, len(test_string))
        
        # Hash should be within expected range
        self.assertGreaterEqual(hash_val, 0)
        self.assertLess(hash_val, fnv_hash.prime)
        
        # Test consistency
        hash_val2 = fnv_hash.calculate_hash(test_string, len(test_string))
        self.assertEqual(hash_val, hash_val2)
    
    def test_rolling_hash_polynomial(self):
        """Test rolling hash functionality for polynomial hash."""
        poly_hash = PolynomialHash(self.base, self.prime)
        
        text = "abcdef"
        pattern_length = 3
        
        # Calculate initial hash
        initial_hash = poly_hash.calculate_hash(text[:pattern_length], pattern_length)
        power = poly_hash.calculate_power(pattern_length)
        
        # Calculate rolling hash
        rolling_hash_val = poly_hash.rolling_hash(
            initial_hash, text[0], text[pattern_length], power
        )
        
        # Calculate direct hash for comparison
        direct_hash = poly_hash.calculate_hash(text[1:pattern_length+1], pattern_length)
        
        # Rolling hash should match direct calculation
        self.assertEqual(rolling_hash_val, direct_hash)
    
    def test_rolling_hash_simple(self):
        """Test rolling hash functionality for simple hash."""
        simple_hash = SimpleHash(self.base, self.prime)
        
        text = "testing"
        pattern_length = 4
        
        # Calculate initial hash
        initial_hash = simple_hash.calculate_hash(text[:pattern_length], pattern_length)
        power = simple_hash.calculate_power(pattern_length)
        
        # Calculate rolling hash
        rolling_hash_val = simple_hash.rolling_hash(
            initial_hash, text[0], text[pattern_length], power
        )
        
        # Calculate direct hash for comparison
        direct_hash = simple_hash.calculate_hash(text[1:pattern_length+1], pattern_length)
        
        # Rolling hash should match direct calculation
        self.assertEqual(rolling_hash_val, direct_hash)
    
    def test_hash_function_factory(self):
        """Test the hash function factory function."""
        # Test all supported hash types
        hash_types = ["polynomial", "simple", "djb2", "fnv"]
        
        for hash_type in hash_types:
            with self.subTest(hash_type=hash_type):
                hash_func = get_hash_function(hash_type, self.base, self.prime)
                
                # Should return appropriate hash function instance
                if hash_type == "polynomial":
                    self.assertIsInstance(hash_func, PolynomialHash)
                elif hash_type == "simple":
                    self.assertIsInstance(hash_func, SimpleHash)
                elif hash_type == "djb2":
                    self.assertIsInstance(hash_func, DJB2Hash)
                elif hash_type == "fnv":
                    self.assertIsInstance(hash_func, FNVHash)
                
                # Should be able to calculate hash
                hash_val = hash_func.calculate_hash("test", 4)
                self.assertIsInstance(hash_val, int)
                self.assertGreaterEqual(hash_val, 0)
    
    def test_invalid_hash_type(self):
        """Test factory function with invalid hash type."""
        # Should return default (polynomial) for invalid type
        hash_func = get_hash_function("invalid_type", self.base, self.prime)
        self.assertIsInstance(hash_func, PolynomialHash)
    
    def test_hash_distribution_quality(self):
        """Test the quality of hash distribution for different functions."""
        hash_types = ["polynomial", "simple", "djb2", "fnv"]
        text = "abcdefghijklmnopqrstuvwxyz" * 10  # Long text
        pattern_length = 5
        
        for hash_type in hash_types:
            with self.subTest(hash_type=hash_type):
                hash_func = get_hash_function(hash_type, self.base, self.prime)
                
                # Calculate hashes for all substrings
                hash_values = []
                for i in range(len(text) - pattern_length + 1):
                    substring = text[i:i + pattern_length]
                    hash_val = hash_func.calculate_hash(substring, pattern_length)
                    hash_values.append(hash_val)
                
                # Check that we get reasonable distribution
                unique_hashes = len(set(hash_values))
                total_hashes = len(hash_values)
                
                # Should have reasonable uniqueness (at least 50% for this test)
                uniqueness_ratio = unique_hashes / total_hashes
                self.assertGreater(uniqueness_ratio, 0.5, 
                                 f"Poor hash distribution for {hash_type}")
    
    def test_compare_hash_functions(self):
        """Test the hash function comparison functionality."""
        text = "this is a test text for comparing hash functions"
        pattern_length = 4
        
        comparison = compare_hash_functions(text, pattern_length, self.base, self.prime)
        
        # Should return results for all hash types
        expected_types = ["polynomial", "simple", "djb2", "fnv"]
        for hash_type in expected_types:
            self.assertIn(hash_type, comparison)
            
            result = comparison[hash_type]
            self.assertIn('name', result)
            self.assertIn('total_windows', result)
            self.assertIn('unique_hashes', result)
            self.assertIn('collision_rate', result)
            self.assertIn('distribution_uniformity', result)
            
            # Validate data ranges
            self.assertGreaterEqual(result['collision_rate'], 0)
            self.assertLessEqual(result['collision_rate'], 100)
            self.assertGreaterEqual(result['distribution_uniformity'], 0)
            self.assertLessEqual(result['distribution_uniformity'], 100)
    
    def test_calculate_uniformity(self):
        """Test the uniformity calculation function."""
        # Test with perfectly uniform distribution
        uniform_values = list(range(100))  # 0, 1, 2, ..., 99
        uniformity = calculate_uniformity(uniform_values, 100)
        self.assertGreater(uniformity, 90)  # Should be very high
        
        # Test with poor distribution (all same values)
        poor_values = [50] * 100
        uniformity = calculate_uniformity(poor_values, 100)
        self.assertLess(uniformity, 50)  # Should be low
        
        # Test with empty list
        uniformity = calculate_uniformity([], 100)
        self.assertEqual(uniformity, 0)
    
    def test_hash_function_properties(self):
        """Test general properties that all hash functions should satisfy."""
        hash_types = ["polynomial", "simple", "djb2", "fnv"]
        
        for hash_type in hash_types:
            with self.subTest(hash_type=hash_type):
                hash_func = get_hash_function(hash_type, self.base, self.prime)
                
                # Test deterministic property
                test_string = "deterministic"
                hash1 = hash_func.calculate_hash(test_string, len(test_string))
                hash2 = hash_func.calculate_hash(test_string, len(test_string))
                self.assertEqual(hash1, hash2, f"{hash_type} is not deterministic")
                
                # Test range property
                for test_str in self.test_strings:
                    hash_val = hash_func.calculate_hash(test_str, len(test_str))
                    self.assertGreaterEqual(hash_val, 0, 
                                          f"{hash_type} produced negative hash")
                    self.assertLess(hash_val, hash_func.prime, 
                                   f"{hash_type} hash exceeds prime modulus")
                
                # Test that different strings usually produce different hashes
                hashes = []
                for test_str in self.test_strings:
                    hash_val = hash_func.calculate_hash(test_str, len(test_str))
                    hashes.append(hash_val)
                
                unique_hashes = len(set(hashes))
                # Should have good uniqueness for different strings
                self.assertGreater(unique_hashes / len(hashes), 0.7, 
                                 f"{hash_type} has poor hash uniqueness")

class TestHashFunctionEdgeCases(unittest.TestCase):
    """Test edge cases for hash functions."""
    
    def test_empty_string_hash(self):
        """Test hash calculation for empty string."""
        hash_types = ["polynomial", "simple", "djb2", "fnv"]
        
        for hash_type in hash_types:
            with self.subTest(hash_type=hash_type):
                hash_func = get_hash_function(hash_type)
                
                # Empty string should produce valid hash (likely 0 or initial value)
                hash_val = hash_func.calculate_hash("", 0)
                self.assertIsInstance(hash_val, int)
                self.assertGreaterEqual(hash_val, 0)
    
    def test_single_character_hash(self):
        """Test hash calculation for single character."""
        hash_types = ["polynomial", "simple", "djb2", "fnv"]
        
        for hash_type in hash_types:
            with self.subTest(hash_type=hash_type):
                hash_func = get_hash_function(hash_type)
                
                hash_val = hash_func.calculate_hash("a", 1)
                self.assertIsInstance(hash_val, int)
                self.assertGreaterEqual(hash_val, 0)
                self.assertLess(hash_val, hash_func.prime)
    
    def test_large_prime_values(self):
        """Test hash functions with large prime values."""
        large_primes = [1009, 10007, 100003]
        
        for prime in large_primes:
            with self.subTest(prime=prime):
                hash_func = PolynomialHash(256, prime)
                
                hash_val = hash_func.calculate_hash("test", 4)
                self.assertGreaterEqual(hash_val, 0)
                self.assertLess(hash_val, prime)
    
    def test_different_base_values(self):
        """Test hash functions with different base values."""
        bases = [2, 16, 128, 256, 512]
        
        for base in bases:
            with self.subTest(base=base):
                hash_func = PolynomialHash(base, 101)
                
                hash_val = hash_func.calculate_hash("test", 4)
                self.assertGreaterEqual(hash_val, 0)
                self.assertLess(hash_val, 101)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestHashFunctions))
    test_suite.addTest(unittest.makeSuite(TestHashFunctionEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Hash Function Tests Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")