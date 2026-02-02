"""
Performance analysis and comparison for string matching algorithms
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from rabin_karp import RabinKarp

class PerformanceAnalyzer:
    def __init__(self):
        """Initialize performance analyzer."""
        self.results = []
    
    def naive_search(self, text, pattern):
        """
        Naive string matching algorithm for comparison.
        
        Args:
            text (str): Text to search in
            pattern (str): Pattern to search for
        
        Returns:
            dict: Search results with timing and statistics
        """
        start_time = time.time()
        matches = []
        comparisons = 0
        
        for i in range(len(text) - len(pattern) + 1):
            match = True
            for j in range(len(pattern)):
                comparisons += 1
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                matches.append(i)
        
        end_time = time.time()
        
        return {
            'matches': matches,
            'time': end_time - start_time,
            'comparisons': comparisons,
            'algorithm': 'Naive'
        }
    
    def rabin_karp_search(self, text, pattern, base=256, prime=101):
        """
        Rabin-Karp search with timing.
        
        Args:
            text (str): Text to search in
            pattern (str): Pattern to search for
            base (int): Base for hash function
            prime (int): Prime modulus
        
        Returns:
            dict: Search results with timing and statistics
        """
        start_time = time.time()
        
        rk = RabinKarp(base, prime)
        results = rk.search(text, pattern)
        
        end_time = time.time()
        
        return {
            'matches': results['matches'],
            'time': end_time - start_time,
            'comparisons': results['statistics']['comparisons'],
            'hash_calculations': results['statistics']['hash_calculations'],
            'spurious_hits': results['statistics']['spurious_hits'],
            'algorithm': 'Rabin-Karp'
        }
    
    def compare_algorithms(self, text, pattern, base=256, prime=101):
        """
        Compare Rabin-Karp with naive algorithm.
        
        Args:
            text (str): Text to search in
            pattern (str): Pattern to search for
            base (int): Base for hash function
            prime (int): Prime modulus
        
        Returns:
            dict: Comparison results
        """
        # Run naive algorithm
        naive_results = self.naive_search(text, pattern)
        
        # Run Rabin-Karp algorithm
        rk_results = self.rabin_karp_search(text, pattern, base, prime)
        
        # Calculate efficiency metrics
        comparison = {
            'text_length': len(text),
            'pattern_length': len(pattern),
            'matches_found': len(naive_results['matches']),
            'naive_time': naive_results['time'],
            'naive_comparisons': naive_results['comparisons'],
            'rk_time': rk_results['time'],
            'rk_comparisons': rk_results['comparisons'],
            'rk_hash_calculations': rk_results['hash_calculations'],
            'rk_spurious_hits': rk_results['spurious_hits'],
            'time_improvement': self._calculate_improvement(naive_results['time'], rk_results['time']),
            'comparison_improvement': self._calculate_improvement(naive_results['comparisons'], rk_results['comparisons'])
        }
        
        return comparison
    
    def _calculate_improvement(self, baseline, improved):
        """Calculate percentage improvement."""
        if baseline == 0:
            return 0
        return ((baseline - improved) / baseline) * 100
    
    def benchmark_different_sizes(self, pattern, sizes=[100, 500, 1000, 2000, 5000]):
        """
        Benchmark algorithms with different text sizes.
        
        Args:
            pattern (str): Pattern to search for
            sizes (list): List of text sizes to test
        
        Returns:
            pd.DataFrame: Benchmark results
        """
        results = []
        
        for size in sizes:
            # Generate test text
            test_text = self._generate_test_text(size, pattern)
            
            # Run comparison
            comparison = self.compare_algorithms(test_text, pattern)
            comparison['text_size'] = size
            results.append(comparison)
        
        return pd.DataFrame(results)
    
    def _generate_test_text(self, size, pattern):
        """Generate test text of specified size."""
        base_text = "abcdefghijklmnopqrstuvwxyz " * (size // 27 + 1)
        
        # Insert pattern at random positions
        import random
        text_list = list(base_text[:size])
        
        # Insert pattern a few times
        num_insertions = max(1, size // 1000)
        for _ in range(num_insertions):
            if len(text_list) >= len(pattern):
                pos = random.randint(0, len(text_list) - len(pattern))
                for i, char in enumerate(pattern):
                    if pos + i < len(text_list):
                        text_list[pos + i] = char
        
        return ''.join(text_list)
    
    def analyze_hash_distribution(self, text, pattern_length, base=256, prime=101):
        """
        Analyze hash value distribution for collision analysis.
        
        Args:
            text (str): Text to analyze
            pattern_length (int): Length of patterns to hash
            base (int): Base for hash function
            prime (int): Prime modulus
        
        Returns:
            dict: Hash distribution analysis
        """
        from utils import calculate_hash
        
        if len(text) < pattern_length:
            return {}
        
        hash_values = []
        hash_counts = {}
        
        # Calculate hash for all possible windows
        for i in range(len(text) - pattern_length + 1):
            window = text[i:i + pattern_length]
            hash_val = calculate_hash(window, pattern_length, base, prime)
            hash_values.append(hash_val)
            
            if hash_val in hash_counts:
                hash_counts[hash_val] += 1
            else:
                hash_counts[hash_val] = 1
        
        # Calculate collision statistics
        total_windows = len(hash_values)
        unique_hashes = len(hash_counts)
        collisions = sum(1 for count in hash_counts.values() if count > 1)
        max_collisions = max(hash_counts.values()) if hash_counts else 0
        
        return {
            'total_windows': total_windows,
            'unique_hashes': unique_hashes,
            'collision_groups': collisions,
            'max_collisions_per_hash': max_collisions,
            'collision_rate': (total_windows - unique_hashes) / total_windows * 100 if total_windows > 0 else 0,
            'hash_values': hash_values,
            'hash_distribution': hash_counts
        }

def display_performance_analysis(text, pattern, base=256, prime=101):
    """Display comprehensive performance analysis."""
    
    st.subheader("📈 Performance Analysis")
    
    analyzer = PerformanceAnalyzer()
    
    # Algorithm comparison
    with st.spinner("Running performance comparison..."):
        comparison = analyzer.compare_algorithms(text, pattern, base, prime)
    
    # Display comparison results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🐌 Naive Algorithm")
        st.metric("Execution Time", f"{comparison['naive_time']:.6f}s")
        st.metric("Character Comparisons", comparison['naive_comparisons'])
    
    with col2:
        st.markdown("#### ⚡ Rabin-Karp Algorithm")
        st.metric("Execution Time", f"{comparison['rk_time']:.6f}s")
        st.metric("Character Comparisons", comparison['rk_comparisons'])
        st.metric("Hash Calculations", comparison['rk_hash_calculations'])
        st.metric("Spurious Hits", comparison['rk_spurious_hits'])
    
    # Performance improvements
    st.markdown("#### 🚀 Performance Improvements")
    
    improvement_col1, improvement_col2 = st.columns(2)
    
    with improvement_col1:
        time_improvement = comparison['time_improvement']
        if time_improvement > 0:
            st.success(f"⏱️ Time: {time_improvement:.1f}% faster")
        elif time_improvement < 0:
            st.warning(f"⏱️ Time: {abs(time_improvement):.1f}% slower")
        else:
            st.info("⏱️ Time: Similar performance")
    
    with improvement_col2:
        comp_improvement = comparison['comparison_improvement']
        if comp_improvement > 0:
            st.success(f"🔍 Comparisons: {comp_improvement:.1f}% fewer")
        elif comp_improvement < 0:
            st.warning(f"🔍 Comparisons: {abs(comp_improvement):.1f}% more")
        else:
            st.info("🔍 Comparisons: Similar count")
    
    # Complexity analysis
    st.markdown("#### 📊 Time Complexity Analysis")
    
    complexity_info = f"""
    **Current Input:**
    - Text Length (n): {comparison['text_length']}
    - Pattern Length (m): {comparison['pattern_length']}
    - Matches Found: {comparison['matches_found']}
    
    **Theoretical Complexity:**
    - Naive Algorithm: O(n×m) = O({comparison['text_length']} × {comparison['pattern_length']}) = O({comparison['text_length'] * comparison['pattern_length']})
    - Rabin-Karp Average: O(n+m) = O({comparison['text_length']} + {comparison['pattern_length']}) = O({comparison['text_length'] + comparison['pattern_length']})
    - Rabin-Karp Worst: O(n×m) (with many spurious hits)
    
    **Actual Performance:**
    - Naive Comparisons: {comparison['naive_comparisons']}
    - Rabin-Karp Comparisons: {comparison['rk_comparisons']}
    - Hash Calculations: {comparison['rk_hash_calculations']}
    """
    
    st.code(complexity_info)
    
    # Hash collision analysis
    if st.checkbox("🔍 Show Hash Collision Analysis"):
        with st.spinner("Analyzing hash distribution..."):
            hash_analysis = analyzer.analyze_hash_distribution(text, len(pattern), base, prime)
        
        if hash_analysis:
            st.markdown("#### 🎯 Hash Collision Analysis")
            
            hash_col1, hash_col2, hash_col3 = st.columns(3)
            
            with hash_col1:
                st.metric("Total Windows", hash_analysis['total_windows'])
                st.metric("Unique Hashes", hash_analysis['unique_hashes'])
            
            with hash_col2:
                st.metric("Collision Groups", hash_analysis['collision_groups'])
                st.metric("Max Collisions", hash_analysis['max_collisions_per_hash'])
            
            with hash_col3:
                st.metric("Collision Rate", f"{hash_analysis['collision_rate']:.2f}%")
            
            # Hash distribution visualization
            if len(hash_analysis['hash_distribution']) <= 50:  # Only show for reasonable number of hashes
                st.markdown("##### Hash Value Distribution")
                
                hash_df = pd.DataFrame([
                    {'Hash Value': k, 'Frequency': v} 
                    for k, v in hash_analysis['hash_distribution'].items()
                ])
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(hash_df)), hash_df['Frequency'])
                ax.set_xlabel('Hash Values (ordered)')
                ax.set_ylabel('Frequency')
                ax.set_title('Hash Value Distribution')
                st.pyplot(fig)

def create_complexity_comparison_chart():
    """Create a visual comparison of algorithm complexities."""
    
    st.subheader("📊 Algorithm Complexity Comparison")
    
    # Generate data for different input sizes
    sizes = [10, 50, 100, 500, 1000, 2000]
    pattern_length = 5
    
    naive_complexity = [n * pattern_length for n in sizes]
    rk_avg_complexity = [n + pattern_length for n in sizes]
    rk_worst_complexity = [n * pattern_length for n in sizes]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Input Size (n)': sizes,
        'Naive O(n×m)': naive_complexity,
        'Rabin-Karp Average O(n+m)': rk_avg_complexity,
        'Rabin-Karp Worst O(n×m)': rk_worst_complexity
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['Input Size (n)'], df['Naive O(n×m)'], 'r-', label='Naive O(n×m)', linewidth=2)
    ax.plot(df['Input Size (n)'], df['Rabin-Karp Average O(n+m)'], 'g-', label='Rabin-Karp Average O(n+m)', linewidth=2)
    ax.plot(df['Input Size (n)'], df['Rabin-Karp Worst O(n×m)'], 'b--', label='Rabin-Karp Worst O(n×m)', linewidth=2)
    
    ax.set_xlabel('Input Size (n)')
    ax.set_ylabel('Time Complexity')
    ax.set_title(f'Algorithm Complexity Comparison (Pattern Length = {pattern_length})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Display the data table
    st.markdown("##### Complexity Values")
    st.dataframe(df, hide_index=True)