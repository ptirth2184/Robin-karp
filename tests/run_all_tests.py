"""
Test runner for all Rabin-Karp project tests
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from test_rabin_karp import TestRabinKarp, TestMultiPatternRabinKarp, TestRabinKarpPerformance
from test_hash_functions import TestHashFunctions, TestHashFunctionEdgeCases
from test_algorithms import TestStringMatchingAlgorithms, TestBenchmarkSuite, TestAlgorithmSpecificCases
from test_performance import TestPerformance, TestStressTests, TestRobustnessTests

class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.verbosity > 1:
            self.stream.write("✅ ")
            self.stream.flush()
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write("❌ ")
            self.stream.flush()
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write("❌ ")
            self.stream.flush()
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write("⏭️ ")
            self.stream.flush()

class TestSuiteRunner:
    """Main test suite runner with comprehensive reporting."""
    
    def __init__(self):
        self.test_modules = [
            ('Core Algorithm Tests', [
                TestRabinKarp,
                TestMultiPatternRabinKarp,
                TestRabinKarpPerformance
            ]),
            ('Hash Function Tests', [
                TestHashFunctions,
                TestHashFunctionEdgeCases
            ]),
            ('Algorithm Comparison Tests', [
                TestStringMatchingAlgorithms,
                TestBenchmarkSuite,
                TestAlgorithmSpecificCases
            ]),
            ('Performance & Stress Tests', [
                TestPerformance,
                TestStressTests,
                TestRobustnessTests
            ])
        ]
    
    def run_test_category(self, category_name, test_classes, verbosity=2):
        """Run tests for a specific category."""
        print(f"\n{'='*60}")
        print(f"🧪 {category_name}")
        print(f"{'='*60}")
        
        # Create test suite for this category
        suite = unittest.TestSuite()
        for test_class in test_classes:
            suite.addTest(unittest.makeSuite(test_class))
        
        # Run tests with custom result class
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=verbosity,
            resultclass=ColoredTextTestResult
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Print results
        output = stream.getvalue()
        print(output)
        
        # Print summary for this category
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        success = total_tests - failures - errors - skipped
        
        print(f"\n📊 {category_name} Summary:")
        print(f"   ✅ Passed: {success}")
        print(f"   ❌ Failed: {failures}")
        print(f"   🚫 Errors: {errors}")
        print(f"   ⏭️ Skipped: {skipped}")
        print(f"   ⏱️ Time: {end_time - start_time:.2f}s")
        print(f"   📈 Success Rate: {(success / total_tests * 100):.1f}%")
        
        return result
    
    def run_all_tests(self, verbosity=2, stop_on_failure=False):
        """Run all test suites."""
        print("🚀 Starting Rabin-Karp Project Test Suite")
        print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall_start_time = time.time()
        all_results = []
        
        # Run each test category
        for category_name, test_classes in self.test_modules:
            try:
                result = self.run_test_category(category_name, test_classes, verbosity)
                all_results.append((category_name, result))
                
                # Stop on failure if requested
                if stop_on_failure and (result.failures or result.errors):
                    print(f"\n⚠️ Stopping due to failures in {category_name}")
                    break
                    
            except Exception as e:
                print(f"\n❌ Error running {category_name}: {e}")
                if stop_on_failure:
                    break
        
        overall_end_time = time.time()
        
        # Print overall summary
        self.print_overall_summary(all_results, overall_end_time - overall_start_time)
        
        return all_results
    
    def print_overall_summary(self, results, total_time):
        """Print comprehensive test summary."""
        print(f"\n{'='*60}")
        print("📋 OVERALL TEST SUMMARY")
        print(f"{'='*60}")
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        
        # Category breakdown
        print("\n📊 Results by Category:")
        for category_name, result in results:
            tests = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)
            skipped = len(result.skipped)
            success = tests - failures - errors - skipped
            
            total_tests += tests
            total_failures += failures
            total_errors += errors
            total_skipped += skipped
            
            status = "✅" if failures == 0 and errors == 0 else "❌"
            print(f"   {status} {category_name}: {success}/{tests} passed")
        
        # Overall statistics
        total_success = total_tests - total_failures - total_errors - total_skipped
        success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🎯 Overall Statistics:")
        print(f"   📝 Total Tests: {total_tests}")
        print(f"   ✅ Passed: {total_success}")
        print(f"   ❌ Failed: {total_failures}")
        print(f"   🚫 Errors: {total_errors}")
        print(f"   ⏭️ Skipped: {total_skipped}")
        print(f"   ⏱️ Total Time: {total_time:.2f}s")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        
        # Final verdict
        if total_failures == 0 and total_errors == 0:
            print(f"\n🎉 ALL TESTS PASSED! Your Rabin-Karp project is ready! 🎉")
        else:
            print(f"\n⚠️ Some tests failed. Please review the failures above.")
        
        # Performance insights
        if total_time > 0:
            tests_per_second = total_tests / total_time
            print(f"\n⚡ Performance: {tests_per_second:.1f} tests/second")
        
        print(f"\n{'='*60}")
    
    def run_quick_tests(self):
        """Run a subset of tests for quick validation."""
        print("🏃‍♂️ Running Quick Test Suite...")
        
        # Run only core algorithm tests
        quick_suite = unittest.TestSuite()
        quick_suite.addTest(unittest.makeSuite(TestRabinKarp))
        quick_suite.addTest(unittest.makeSuite(TestHashFunctions))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(quick_suite)
        
        if result.failures == 0 and result.errors == 0:
            print("✅ Quick tests passed! Core functionality is working.")
        else:
            print("❌ Quick tests failed. Check core implementation.")
        
        return result
    
    def run_performance_only(self):
        """Run only performance tests."""
        print("⚡ Running Performance Tests Only...")
        
        perf_suite = unittest.TestSuite()
        perf_suite.addTest(unittest.makeSuite(TestPerformance))
        perf_suite.addTest(unittest.makeSuite(TestStressTests))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(perf_suite)
        
        return result

def main():
    """Main function with command line argument handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Rabin-Karp project tests')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick test suite only')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests only')
    parser.add_argument('--verbose', '-v', action='count', default=2,
                       help='Increase verbosity')
    parser.add_argument('--stop-on-failure', action='store_true',
                       help='Stop on first failure')
    
    args = parser.parse_args()
    
    runner = TestSuiteRunner()
    
    if args.quick:
        runner.run_quick_tests()
    elif args.performance:
        runner.run_performance_only()
    else:
        runner.run_all_tests(
            verbosity=args.verbose,
            stop_on_failure=args.stop_on_failure
        )

if __name__ == '__main__':
    main()