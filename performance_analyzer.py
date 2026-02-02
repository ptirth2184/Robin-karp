"""
Performance analysis and comparison for string matching algorithms
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from rabin_karp import RabinKarp
from algorithms import StringMatchingAlgorithms, BenchmarkSuite

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

def display_comprehensive_algorithm_comparison(text, pattern, base=256, prime=101, hash_type="polynomial"):
    """Display comprehensive comparison of all string matching algorithms."""
    
    st.subheader("🏆 Comprehensive Algorithm Comparison")
    
    if not text or not pattern:
        st.warning("Please provide both text and pattern for comparison.")
        return
    
    # Initialize algorithms
    algorithms = StringMatchingAlgorithms()
    
    with st.spinner("Running comprehensive algorithm comparison..."):
        # Get results from all algorithms
        all_results = algorithms.compare_all_algorithms(text, pattern)
        
        # Add Rabin-Karp results
        start_time = time.perf_counter()
        rk = RabinKarp(base=base, prime=prime, hash_type=hash_type)
        rk_result = rk.search(text, pattern)
        end_time = time.perf_counter()
        
        all_results['Rabin-Karp'] = {
            'algorithm': 'Rabin-Karp',
            'matches': rk_result['matches'],
            'time': end_time - start_time,
            'comparisons': rk_result['statistics']['comparisons'],
            'space_complexity': 'O(1)',
            'time_complexity_avg': 'O(n+m)',
            'time_complexity_worst': 'O(n*m)',
            'hash_calculations': rk_result['statistics'].get('hash_calculations', 0),
            'spurious_hits': rk_result['statistics'].get('spurious_hits', 0)
        }
    
    # Display results table
    st.markdown("#### 📊 Performance Comparison Table")
    
    comparison_data = []
    for algo_name, result in all_results.items():
        if 'error' in result:
            continue
        
        comparison_data.append({
            'Algorithm': algo_name,
            'Matches Found': len(result['matches']),
            'Execution Time (s)': f"{result['time']:.6f}",
            'Comparisons': result['comparisons'],
            'Space Complexity': result.get('space_complexity', 'N/A'),
            'Avg Time Complexity': result.get('time_complexity_avg', 'N/A'),
            'Worst Time Complexity': result.get('time_complexity_worst', 'N/A')
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True)
    
    # Performance metrics
    st.markdown("#### ⚡ Performance Metrics")
    
    # Find fastest and most efficient algorithms
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    if valid_results:
        fastest = min(valid_results.items(), key=lambda x: x[1]['time'])
        most_efficient = min(valid_results.items(), key=lambda x: x[1]['comparisons'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🏃 Fastest Algorithm", 
                fastest[0], 
                f"{fastest[1]['time']:.6f}s"
            )
        
        with col2:
            st.metric(
                "🎯 Most Efficient", 
                most_efficient[0], 
                f"{most_efficient[1]['comparisons']} comparisons"
            )
        
        with col3:
            total_matches = len(fastest[1]['matches'])
            st.metric("🎯 Total Matches", total_matches)
    
    # Visualization
    create_algorithm_comparison_charts(all_results)
    
    # Algorithm recommendations
    display_algorithm_recommendations(all_results, len(text), len(pattern))

def create_algorithm_comparison_charts(results):
    """Create visualization charts for algorithm comparison."""
    
    st.markdown("#### 📈 Performance Visualization")
    
    # Filter out error results
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        st.warning("No valid results to visualize.")
        return
    
    # Prepare data for charts
    algorithms = list(valid_results.keys())
    times = [result['time'] for result in valid_results.values()]
    comparisons = [result['comparisons'] for result in valid_results.values()]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Execution time chart
    bars1 = ax1.bar(algorithms, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax1.set_title('Execution Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.6f}s', ha='center', va='bottom', fontsize=8)
    
    # Comparisons chart
    bars2 = ax2.bar(algorithms, comparisons, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax2.set_title('Character Comparisons')
    ax2.set_ylabel('Number of Comparisons')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, comp_val in zip(bars2, comparisons):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{comp_val}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Efficiency ratio chart
    st.markdown("#### 🎯 Efficiency Analysis")
    
    # Calculate efficiency metrics
    min_time = min(times)
    min_comparisons = min(comparisons)
    
    efficiency_data = []
    for algo, result in valid_results.items():
        time_ratio = result['time'] / min_time if min_time > 0 else 1
        comp_ratio = result['comparisons'] / min_comparisons if min_comparisons > 0 else 1
        
        efficiency_data.append({
            'Algorithm': algo,
            'Time Ratio': round(time_ratio, 2),
            'Comparison Ratio': round(comp_ratio, 2),
            'Overall Score': round(2 / (time_ratio + comp_ratio), 2)
        })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    st.dataframe(efficiency_df, hide_index=True)

def display_algorithm_recommendations(results, text_length, pattern_length):
    """Display algorithm recommendations based on results and input characteristics."""
    
    st.markdown("#### 💡 Algorithm Recommendations")
    
    # Filter valid results
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        return
    
    # Generate recommendations based on input characteristics
    recommendations = []
    
    # Text length based recommendations
    if text_length < 100:
        recommendations.append("📝 **Small Text**: Naive algorithm is sufficient for small inputs")
    elif text_length < 10000:
        recommendations.append("📄 **Medium Text**: KMP or Rabin-Karp recommended for balanced performance")
    else:
        recommendations.append("📚 **Large Text**: Boyer-Moore or Rabin-Karp for optimal performance")
    
    # Pattern length based recommendations
    if pattern_length == 1:
        recommendations.append("🔤 **Single Character**: Simple algorithms work well")
    elif pattern_length < 10:
        recommendations.append("📝 **Short Pattern**: KMP provides consistent O(n+m) performance")
    else:
        recommendations.append("📄 **Long Pattern**: Boyer-Moore excels with longer patterns")
    
    # Performance based recommendations
    fastest = min(valid_results.items(), key=lambda x: x[1]['time'])
    most_efficient = min(valid_results.items(), key=lambda x: x[1]['comparisons'])
    
    recommendations.extend([
        f"🏃 **Fastest for this input**: {fastest[0]} ({fastest[1]['time']:.6f}s)",
        f"🎯 **Most efficient**: {most_efficient[0]} ({most_efficient[1]['comparisons']} comparisons)"
    ])
    
    # Use case recommendations
    use_case_recommendations = [
        "🔍 **Multiple Patterns**: Use Rabin-Karp for searching multiple patterns",
        "📊 **Guaranteed Performance**: Use KMP for predictable O(n+m) behavior",
        "📖 **Natural Language**: Use Boyer-Moore for text processing applications",
        "🧬 **Bioinformatics**: Use Z Algorithm for pattern analysis and preprocessing",
        "🔢 **Hash-based Applications**: Use Rabin-Karp for rolling hash benefits"
    ]
    
    # Display recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Input-Specific Recommendations:**")
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    with col2:
        st.markdown("**Use Case Recommendations:**")
        for rec in use_case_recommendations:
            st.markdown(f"- {rec}")

def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite and display results."""
    
    st.subheader("🏁 Comprehensive Benchmark Suite")
    st.write("Run standardized tests across multiple scenarios to compare algorithm performance.")
    
    if st.button("🚀 Run Full Benchmark Suite", type="primary"):
        benchmark = BenchmarkSuite()
        
        with st.spinner("Running comprehensive benchmark... This may take a moment."):
            results = benchmark.run_benchmark(include_rabin_karp=True)
            analysis = benchmark.analyze_performance_trends(results)
        
        # Display benchmark results
        st.success("✅ Benchmark completed!")
        
        # Summary statistics
        st.markdown("#### 📊 Benchmark Summary")
        
        # Algorithm rankings
        st.markdown("##### 🏆 Algorithm Rankings")
        
        rankings_data = []
        for algo_name, metrics in analysis['algorithm_rankings'].items():
            rankings_data.append({
                'Algorithm': algo_name,
                'Avg Time (s)': f"{metrics['average_time']:.6f}",
                'Avg Comparisons': f"{metrics['average_comparisons']:.0f}",
                'Consistency Score': f"{1/metrics['consistency']:.3f}" if metrics['consistency'] != float('inf') else "Perfect"
            })
        
        rankings_df = pd.DataFrame(rankings_data)
        rankings_df = rankings_df.sort_values('Avg Time (s)')
        st.dataframe(rankings_df, hide_index=True)
        
        # Detailed results by test case
        st.markdown("##### 📋 Detailed Results by Test Case")
        
        for case_name, case_data in results.items():
            with st.expander(f"📁 {case_name}"):
                st.write(f"**Text Length**: {case_data['text_length']}")
                st.write(f"**Pattern Length**: {case_data['pattern_length']}")
                
                case_results = []
                for algo_name, algo_result in case_data['algorithms'].items():
                    if 'error' not in algo_result:
                        case_results.append({
                            'Algorithm': algo_name,
                            'Time (s)': f"{algo_result['time']:.6f}",
                            'Comparisons': algo_result['comparisons'],
                            'Matches': len(algo_result['matches'])
                        })
                
                if case_results:
                    case_df = pd.DataFrame(case_results)
                    st.dataframe(case_df, hide_index=True)
        
        # Recommendations
        st.markdown("##### 💡 Benchmark Recommendations")
        for recommendation in analysis['recommendations']:
            st.markdown(f"- {recommendation}")

def create_scalability_analysis():
    """Create scalability analysis for different input sizes."""
    
    st.subheader("📈 Scalability Analysis")
    st.write("Analyze how algorithms perform with increasing input sizes.")
    
    # Input size selection
    col1, col2 = st.columns(2)
    
    with col1:
        max_text_size = st.selectbox(
            "Maximum Text Size",
            [100, 500, 1000, 5000, 10000],
            index=2
        )
    
    with col2:
        pattern_size = st.selectbox(
            "Pattern Size",
            [1, 3, 5, 10, 20],
            index=2
        )
    
    if st.button("📊 Run Scalability Analysis"):
        algorithms = StringMatchingAlgorithms()
        
        # Generate test sizes
        sizes = [max_text_size // 10, max_text_size // 5, max_text_size // 2, max_text_size]
        
        scalability_results = {}
        
        with st.spinner("Running scalability analysis..."):
            for size in sizes:
                # Generate test text
                test_text = "abcdefghij" * (size // 10) + "xyz" + "abcdefghij" * (size // 10)
                test_pattern = "x" * pattern_size
                
                # Run algorithms
                size_results = algorithms.compare_all_algorithms(test_text[:size], test_pattern)
                
                # Add Rabin-Karp
                rk = RabinKarp()
                start_time = time.perf_counter()
                rk_result = rk.search(test_text[:size], test_pattern)
                end_time = time.perf_counter()
                
                size_results['Rabin-Karp'] = {
                    'algorithm': 'Rabin-Karp',
                    'time': end_time - start_time,
                    'comparisons': rk_result['statistics']['comparisons']
                }
                
                scalability_results[size] = size_results
        
        # Create scalability charts
        display_scalability_charts(scalability_results)

def display_scalability_charts(results):
    """Display scalability analysis charts."""
    
    st.markdown("#### 📈 Scalability Results")
    
    # Prepare data
    sizes = sorted(results.keys())
    algorithms = set()
    
    for size_results in results.values():
        algorithms.update(size_results.keys())
    
    algorithms = [algo for algo in algorithms if 'error' not in results[sizes[0]].get(algo, {})]
    
    # Create time complexity chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, algo in enumerate(algorithms):
        times = []
        comparisons = []
        
        for size in sizes:
            algo_result = results[size].get(algo, {})
            if 'error' not in algo_result:
                times.append(algo_result['time'])
                comparisons.append(algo_result['comparisons'])
            else:
                times.append(0)
                comparisons.append(0)
        
        color = colors[i % len(colors)]
        ax1.plot(sizes, times, marker='o', label=algo, color=color, linewidth=2)
        ax2.plot(sizes, comparisons, marker='s', label=algo, color=color, linewidth=2)
    
    ax1.set_title('Execution Time vs Input Size')
    ax1.set_xlabel('Text Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Comparisons vs Input Size')
    ax2.set_xlabel('Text Size')
    ax2.set_ylabel('Number of Comparisons')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display scalability table
    st.markdown("#### 📊 Scalability Data")
    
    scalability_data = []
    for size in sizes:
        for algo in algorithms:
            algo_result = results[size].get(algo, {})
            if 'error' not in algo_result:
                scalability_data.append({
                    'Input Size': size,
                    'Algorithm': algo,
                    'Time (s)': f"{algo_result['time']:.6f}",
                    'Comparisons': algo_result['comparisons']
                })
    
    scalability_df = pd.DataFrame(scalability_data)
    st.dataframe(scalability_df, hide_index=True)