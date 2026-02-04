"""
Streamlit Web Application for Rabin-Karp String Matching Algorithm
"""

import streamlit as st
import pandas as pd
from rabin_karp import RabinKarp, MultiPatternRabinKarp
from utils import format_matches
from visualizer import display_step_by_step_visualization, create_algorithm_flow_chart
from performance_analyzer import display_performance_analysis, create_complexity_comparison_chart, display_comprehensive_algorithm_comparison, run_comprehensive_benchmark, create_scalability_analysis
from hash_functions import get_hash_function, compare_hash_functions

def main():
    st.set_page_config(
        page_title="Rabin-Karp String Matching",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Rabin-Karp String Matching Algorithm")
    st.markdown("### Design and Analysis of Algorithms Project")
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Basic Search", 
        "🎯 Multi-Pattern Search",
        "🎬 Step-by-Step Visualization", 
        "📈 Performance Analysis",
        "📚 Algorithm Theory",
        "🧪 Testing Suite"
    ])
    
    # Sidebar for algorithm parameters
    st.sidebar.header("⚙️ Algorithm Settings")
    
    # Hash function selection
    hash_type = st.sidebar.selectbox(
        "Hash Function Type",
        ["polynomial", "simple", "djb2", "fnv"],
        index=0,
        help="Choose the hash function algorithm"
    )
    
    base = st.sidebar.number_input(
        "Base (for hash function)", 
        min_value=2, 
        max_value=1000, 
        value=256 if hash_type == "polynomial" else 33 if hash_type == "djb2" else 256,
        help="Base value for hash function"
    )
    
    prime = st.sidebar.number_input(
        "Prime Modulus", 
        min_value=2, 
        max_value=10000, 
        value=101,
        help="Prime number for modular arithmetic"
    )
    
    case_sensitive = st.sidebar.checkbox(
        "Case Sensitive Search", 
        value=True,
        help="Whether the search should be case sensitive"
    )
    
    # Display hash function info
    hash_func = get_hash_function(hash_type, base, prime)
    st.sidebar.info(f"**{hash_func.name}**\n\n{hash_func.description}")
    
    # Sample text options
    st.sidebar.header("📝 Sample Texts")
    sample_option = st.sidebar.selectbox(
        "Choose a sample text:",
        [
            "Custom",
            "Lorem Ipsum",
            "Programming Text",
            "DNA Sequence",
            "Repeated Pattern"
        ]
    )
    
    # Get sample text and pattern
    text, pattern = get_sample_text_and_pattern(sample_option)
    
    # Tab 1: Basic Search
    with tab1:
        basic_search_tab(text, pattern, base, prime, case_sensitive, hash_type)
    
    # Tab 2: Multi-Pattern Search
    with tab2:
        multi_pattern_search_tab(text, base, prime, case_sensitive, hash_type)
    
    # Tab 3: Step-by-Step Visualization
    with tab3:
        visualization_tab(text, pattern, base, prime, hash_type)
    
    # Tab 4: Performance Analysis
    with tab4:
        performance_tab(text, pattern, base, prime, hash_type)
    
    # Tab 5: Algorithm Theory
    with tab5:
        theory_tab()
    
    # Tab 6: Testing Suite
    with tab6:
        testing_tab()

def get_sample_text_and_pattern(option):
    """Get sample text and pattern based on user selection."""
    
    samples = {
        "Custom": (
            "The quick brown fox jumps over the lazy dog. The fox is quick and brown.",
            "fox"
        ),
        "Lorem Ipsum": (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
            "dolor"
        ),
        "Programming Text": (
            "def rabin_karp_search(text, pattern): for i in range(len(text)): if text[i:i+len(pattern)] == pattern: return i return -1",
            "pattern"
        ),
        "DNA Sequence": (
            "ATCGATCGATCGATCGTAGCTAGCTAGCTAGCTACGATCGATCGATCGTAGCTAGCTAGCT",
            "ATCG"
        ),
        "Repeated Pattern": (
            "abcabcabcabcdefabcabcabcghiabcabcjklabcabcabcmnoabcabc",
            "abc"
        )
    }
    
    return samples.get(option, samples["Custom"])

def basic_search_tab(default_text, default_pattern, base, prime, case_sensitive, hash_type):
    """Basic search functionality tab."""
    
    # Main input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input Text")
        text = st.text_area(
            "Enter the text to search in:",
            height=200,
            placeholder="Enter your text here...",
            value=default_text
        )
    
    with col2:
        st.header("🎯 Search Pattern")
        pattern = st.text_input(
            "Enter pattern to search:",
            placeholder="Enter pattern...",
            value=default_pattern
        )
        
        st.header("📊 Quick Stats")
        if text and pattern:
            st.metric("Text Length", len(text))
            st.metric("Pattern Length", len(pattern))
            st.metric("Max Possible Matches", max(0, len(text) - len(pattern) + 1))
    
    # Search button
    if st.button("🔍 Search Pattern", type="primary"):
        if not text.strip():
            st.error("Please enter some text to search in.")
            return
        
        if not pattern.strip():
            st.error("Please enter a pattern to search for.")
            return
        
        # Perform search
        with st.spinner("Searching..."):
            rk = RabinKarp(base=base, prime=prime, hash_type=hash_type)
            results = rk.search(text, pattern, case_sensitive)
        
        # Display results
        display_results(text, pattern, results)

def multi_pattern_search_tab(default_text, base, prime, case_sensitive, hash_type):
    """Multi-pattern search functionality tab."""
    
    st.header("� Multiple Pattern Search")
    st.write("Search for multiple patterns simultaneously in the same text.")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        text = st.text_area(
            "Enter the text to search in:",
            height=150,
            placeholder="Enter your text here...",
            value=default_text,
            key="multi_text"
        )
    
    with col2:
        st.subheader("🎯 Search Patterns")
        patterns_input = st.text_area(
            "Enter patterns (one per line):",
            height=150,
            placeholder="pattern1\npattern2\npattern3",
            value="fox\ndog\nthe\nbrown",
            key="multi_patterns"
        )
        
        # Parse patterns
        patterns = [p.strip() for p in patterns_input.split('\n') if p.strip()]
        
        st.subheader("📊 Pattern Stats")
        if patterns:
            st.metric("Number of Patterns", len(patterns))
            st.metric("Avg Pattern Length", round(sum(len(p) for p in patterns) / len(patterns), 1))
    
    # Search options
    col1, col2 = st.columns(2)
    
    with col1:
        search_mode = st.radio(
            "Search Mode:",
            ["Individual Search", "Optimized Multi-Search"],
            help="Individual: Search each pattern separately\nOptimized: Use multi-pattern algorithm"
        )
    
    with col2:
        show_details = st.checkbox("Show Detailed Statistics", value=True)
    
    # Search button
    if st.button("� Search All Patterns", type="primary", key="multi_search"):
        if not text.strip():
            st.error("Please enter some text to search in.")
            return
        
        if not patterns:
            st.error("Please enter at least one pattern to search for.")
            return
        
        # Perform multi-pattern search
        with st.spinner("Searching for multiple patterns..."):
            if search_mode == "Optimized Multi-Search":
                multi_rk = MultiPatternRabinKarp(base=base, prime=prime, hash_type=hash_type)
                results = multi_rk.search(text, patterns, case_sensitive)
            else:
                # Individual searches
                results = perform_individual_searches(text, patterns, base, prime, case_sensitive, hash_type)
        
        # Display multi-pattern results
        display_multi_pattern_results(text, patterns, results, show_details)

def perform_individual_searches(text, patterns, base, prime, case_sensitive, hash_type):
    """Perform individual searches for each pattern."""
    matches = {}
    pattern_stats = {}
    overall_stats = {'comparisons': 0, 'hash_calculations': 0, 'spurious_hits': 0}
    
    for pattern in patterns:
        rk = RabinKarp(base=base, prime=prime, hash_type=hash_type)
        result = rk.search(text, pattern, case_sensitive)
        
        matches[pattern] = result['matches']
        pattern_stats[pattern] = {
            'matches_found': len(result['matches']),
            'comparisons': result['statistics']['comparisons'],
            'hash_calculations': result['statistics']['hash_calculations'],
            'spurious_hits': result['statistics']['spurious_hits']
        }
        
        # Accumulate overall stats
        overall_stats['comparisons'] += result['statistics']['comparisons']
        overall_stats['hash_calculations'] += result['statistics']['hash_calculations']
        overall_stats['spurious_hits'] += result['statistics']['spurious_hits']
    
    return {
        'matches': matches,
        'pattern_stats': pattern_stats,
        'overall_stats': overall_stats,
        'efficiency_metrics': {}  # Will be calculated if needed
    }

def display_multi_pattern_results(text, patterns, results, show_details):
    """Display results for multi-pattern search."""
    
    matches = results['matches']
    pattern_stats = results['pattern_stats']
    overall_stats = results['overall_stats']
    
    # Summary metrics
    st.header("📊 Search Summary")
    
    total_matches = sum(len(match_list) for match_list in matches.values())
    patterns_with_matches = sum(1 for match_list in matches.values() if match_list)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", total_matches)
    with col2:
        st.metric("Patterns Found", f"{patterns_with_matches}/{len(patterns)}")
    with col3:
        st.metric("Total Comparisons", overall_stats['comparisons'])
    with col4:
        st.metric("Spurious Hits", overall_stats['spurious_hits'])
    
    # Pattern-wise results
    st.header("🎯 Pattern-wise Results")
    
    # Create results table
    results_data = []
    for pattern in patterns:
        pattern_matches = matches.get(pattern, [])
        stats = pattern_stats.get(pattern, {})
        
        results_data.append({
            'Pattern': pattern,
            'Matches Found': len(pattern_matches),
            'Positions': ', '.join(map(str, pattern_matches[:5])) + ('...' if len(pattern_matches) > 5 else ''),
            'Comparisons': stats.get('comparisons', 0),
            'Hash Calculations': stats.get('hash_calculations', 0),
            'Spurious Hits': stats.get('spurious_hits', 0)
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, hide_index=True)
    
    # Detailed view for each pattern
    if show_details:
        st.header("🔍 Detailed Match Information")
        
        for pattern in patterns:
            pattern_matches = matches.get(pattern, [])
            if pattern_matches:
                with st.expander(f"Pattern '{pattern}' - {len(pattern_matches)} matches"):
                    formatted_matches = format_matches(pattern_matches, text, pattern)
                    
                    for i, match_info in enumerate(formatted_matches, 1):
                        st.write(f"**Match {i}:** Position {match_info['position']}")
                        st.write(f"Context: {match_info['context']}")
                        st.write("---")
    
    # Efficiency metrics (if available)
    if 'efficiency_metrics' in results and results['efficiency_metrics']:
        st.header("⚡ Efficiency Metrics")
        metrics = results['efficiency_metrics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Success Rate", f"{metrics.get('match_success_rate', 0):.1f}%")
        with col2:
            st.metric("Efficiency Improvement", f"{metrics.get('efficiency_improvement', 0):.1f}%")
        with col3:
            st.metric("Avg Comparisons/Pattern", f"{metrics.get('comparisons_per_pattern', 0):.1f}")

def visualization_tab(default_text, default_pattern, base, prime, hash_type):
    """Step-by-step visualization tab."""
    
    st.header("🎬 Algorithm Visualization")
    st.write("Watch the Rabin-Karp algorithm execute step by step!")
    
    # Input section for visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        viz_text = st.text_input(
            "Text for visualization (keep it short for better display):",
            value=default_text[:50] if len(default_text) > 50 else default_text,
            help="Shorter text works better for step-by-step visualization"
        )
    
    with col2:
        viz_pattern = st.text_input(
            "Pattern:",
            value=default_pattern
        )
    
    if viz_text and viz_pattern:
        if len(viz_text) > 100:
            st.warning("⚠️ Text is quite long. Consider using shorter text for better visualization experience.")
        
        if len(viz_pattern) > len(viz_text):
            st.error("Pattern cannot be longer than text!")
        else:
            display_step_by_step_visualization(viz_text, viz_pattern, base, prime)

def performance_tab(default_text, default_pattern, base, prime, hash_type):
    """Performance analysis tab."""
    
    st.header("📈 Performance Analysis")
    st.write("Compare Rabin-Karp with other string matching algorithms.")
    
    # Create sub-tabs for different types of analysis
    perf_tab1, perf_tab2, perf_tab3 = st.tabs([
        "🔍 Basic Comparison",
        "🏆 Algorithm Comparison", 
        "📊 Benchmark Suite"
    ])
    
    # Basic Performance Analysis
    with perf_tab1:
        st.subheader("🔍 Basic Performance Analysis")
        st.write("Compare Rabin-Karp with naive string matching algorithm.")
        
        # Input section for performance testing
        perf_text = st.text_area(
            "Text for performance testing:",
            value=default_text,
            height=150,
            help="Larger text will show more significant performance differences"
        )
        
        perf_pattern = st.text_input(
            "Pattern for performance testing:",
            value=default_pattern
        )
        
        if st.button("🚀 Run Basic Performance Analysis", type="primary"):
            if perf_text and perf_pattern:
                display_performance_analysis(perf_text, perf_pattern, base, prime)
            else:
                st.error("Please provide both text and pattern for analysis.")
        
        # Hash function comparison
        st.markdown("---")
        st.subheader("🔢 Hash Function Comparison")
        
        if st.button("Compare Hash Functions"):
            if perf_text and perf_pattern:
                with st.spinner("Comparing hash functions..."):
                    comparison = compare_hash_functions(perf_text, len(perf_pattern), base, prime)
                
                if comparison:
                    display_hash_function_comparison(comparison)
            else:
                st.error("Please provide text and pattern for hash function comparison.")
    
    # Comprehensive Algorithm Comparison
    with perf_tab2:
        st.subheader("🏆 Comprehensive Algorithm Comparison")
        st.write("Compare Rabin-Karp with Naive, KMP, Boyer-Moore, and Z Algorithm.")
        
        # Input section
        comp_text = st.text_area(
            "Text for algorithm comparison:",
            value=default_text,
            height=120,
            key="comp_text"
        )
        
        comp_pattern = st.text_input(
            "Pattern for algorithm comparison:",
            value=default_pattern,
            key="comp_pattern"
        )
        
        if st.button("🏁 Run Algorithm Comparison", type="primary"):
            if comp_text and comp_pattern:
                display_comprehensive_algorithm_comparison(comp_text, comp_pattern, base, prime, hash_type)
            else:
                st.error("Please provide both text and pattern for comparison.")
        
        # Scalability Analysis
        st.markdown("---")
        create_scalability_analysis()
    
    # Benchmark Suite
    with perf_tab3:
        run_comprehensive_benchmark()
    
    # Theoretical complexity comparison (always visible)
    st.markdown("---")
    create_complexity_comparison_chart()

def display_hash_function_comparison(comparison):
    """Display hash function comparison results."""
    
    st.subheader("📊 Hash Function Performance Comparison")
    
    # Create comparison table
    comparison_data = []
    for hash_type, data in comparison.items():
        comparison_data.append({
            'Hash Function': data['name'],
            'Unique Hashes': data['unique_hashes'],
            'Collision Groups': data['collision_groups'],
            'Collision Rate (%)': f"{data['collision_rate']:.2f}%",
            'Uniformity Score': data['distribution_uniformity']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True)
    
    # Detailed analysis
    st.subheader("🔍 Detailed Analysis")
    
    for hash_type, data in comparison.items():
        with st.expander(f"{data['name']} - Details"):
            st.write(f"**Description:** {data['description']}")
            st.write(f"**Total Windows Analyzed:** {data['total_windows']}")
            st.write(f"**Unique Hash Values:** {data['unique_hashes']}")
            st.write(f"**Collision Groups:** {data['collision_groups']}")
            st.write(f"**Collision Rate:** {data['collision_rate']:.2f}%")
            st.write(f"**Distribution Uniformity:** {data['distribution_uniformity']}/100")
            
            if data['hash_values']:
                st.write("**Sample Hash Values:**")
                st.code(', '.join(map(str, data['hash_values'][:10])))

def theory_tab():
    """Algorithm theory and explanation tab."""
    
    st.header("📚 Rabin-Karp Algorithm Theory")
    
    # Algorithm explanation
    st.markdown("""
    ### 🎯 What is Rabin-Karp Algorithm?
    
    The Rabin-Karp algorithm is a string-searching algorithm that uses **hashing** to find patterns in text. 
    It was developed by Richard Karp and Michael Rabin in 1987.
    
    ### 🔑 Key Concepts
    
    1. **Rolling Hash**: Instead of recalculating hash from scratch for each position, 
       we use a rolling hash technique that updates the hash in constant time.
    
    2. **Hash Function**: We use various hash functions for different performance characteristics.
    
    3. **Spurious Hits**: When hash values match but strings don't, we need character-by-character verification.
    """)
    
    # Algorithm steps
    create_algorithm_flow_chart()
    
    # Hash function details
    st.markdown("""
    ### 🔢 Hash Function Types
    
    **Polynomial Hash**: `hash = (c₀×base^(n-1) + c₁×base^(n-2) + ... + cₙ₋₁) mod prime`
    - Best overall performance
    - Good distribution properties
    - Efficient rolling hash
    
    **Simple Additive Hash**: `hash = (sum of ASCII values × base) mod prime`
    - Simple to understand
    - Fast computation
    - Higher collision rate
    
    **DJB2 Hash**: `hash = ((hash × 33) + c) mod prime`
    - Popular in practice
    - Good distribution
    - Fast computation
    
    **FNV Hash**: `hash = (hash × prime ⊕ c) mod prime`
    - Good avalanche effect
    - Used in many applications
    - Slightly more complex
    """)
    
    # Advantages and disadvantages
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Advantages
        - **Average O(n+m) time complexity**
        - **Good for multiple pattern search**
        - **Works well with large alphabets**
        - **Multiple hash function options**
        - **Effective for long patterns**
        - **Parallelizable for multiple patterns**
        """)
    
    with col2:
        st.markdown("""
        ### ❌ Disadvantages
        - **Worst case O(nm) complexity**
        - **Spurious hits can degrade performance**
        - **Hash function choice affects performance**
        - **Parameter tuning required**
        - **Memory overhead for hash calculations**
        """)
    
    # Applications
    st.markdown("""
    ### 🚀 Real-World Applications
    
    - **Plagiarism Detection**: Finding similar text passages
    - **DNA Sequence Analysis**: Searching for genetic patterns
    - **Text Editors**: Find and replace functionality
    - **Web Search**: Pattern matching in large documents
    - **Data Deduplication**: Finding duplicate content
    - **Network Security**: Pattern matching in network packets
    - **Bioinformatics**: Protein sequence matching
    - **Document Similarity**: Comparing document content
    """)
    
    # Interactive hash calculator
    st.markdown("### 🧮 Interactive Hash Calculator")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        calc_text = st.text_input("Text to hash:", value="hello")
        calc_hash_type = st.selectbox("Hash Function:", ["polynomial", "simple", "djb2", "fnv"])
    
    with calc_col2:
        calc_base = st.number_input("Base:", value=256, min_value=2)
        calc_prime = st.number_input("Prime:", value=101, min_value=2)
        
        if calc_text:
            hash_func = get_hash_function(calc_hash_type, calc_base, calc_prime)
            hash_val = hash_func.calculate_hash(calc_text, len(calc_text))
            st.metric("Hash Value", hash_val)
            st.write(f"**Function:** {hash_func.name}")
            st.write(f"**Formula:** {hash_func.description}")

def display_results(text, pattern, results):
    """Display search results in the Streamlit interface."""
    
    matches = results['matches']
    statistics = results['statistics']
    hash_values = results['hash_values']
    
    # Results summary
    st.header("🎯 Search Results")
    
    if matches:
        st.success(f"✅ Found {len(matches)} match(es) for pattern '{pattern}'")
        
        # Display matches in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Matches", len(matches))
        with col2:
            st.metric("Character Comparisons", statistics['comparisons'])
        with col3:
            st.metric("Spurious Hits", statistics['spurious_hits'])
        
        # Show match details
        st.subheader("📍 Match Details")
        
        formatted_matches = format_matches(matches, text, pattern)
        
        for i, match_info in enumerate(formatted_matches, 1):
            with st.expander(f"Match {i} at position {match_info['position']}"):
                st.write(f"**Position:** {match_info['position']}")
                st.write(f"**Matched Text:** `{match_info['match']}`")
                st.write(f"**Context:** {match_info['context']}")
        
        # Highlight matches in text
        st.subheader("📝 Text with Highlighted Matches")
        highlighted_text = highlight_matches(text, matches, len(pattern))
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
    else:
        st.warning(f"❌ No matches found for pattern '{pattern}'")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Character Comparisons", statistics['comparisons'])
        with col2:
            st.metric("Hash Calculations", statistics['hash_calculations'])
    
    # Hash values visualization
    if hash_values:
        st.subheader("🔢 Hash Values")
        
        # Create DataFrame for hash values
        hash_df = pd.DataFrame(hash_values)
        
        # Display hash values table
        st.dataframe(
            hash_df,
            column_config={
                "position": "Position",
                "hash": "Text Hash",
                "pattern_hash": "Pattern Hash",
                "match": "Hash Match"
            },
            hide_index=True
        )
        
        # Show algorithm statistics
        st.subheader("📈 Algorithm Statistics")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric(
                "Hash Calculations", 
                statistics['hash_calculations'],
                help="Total number of hash calculations performed"
            )
        
        with stats_col2:
            st.metric(
                "Character Comparisons", 
                statistics['comparisons'],
                help="Number of character-by-character comparisons"
            )
        
        with stats_col3:
            efficiency = round((statistics['hash_calculations'] / max(1, len(text))) * 100, 2)
            st.metric(
                "Efficiency %", 
                f"{efficiency}%",
                help="Hash calculations as percentage of text length"
            )

def highlight_matches(text, matches, pattern_length):
    """Highlight matches in the text using HTML."""
    if not matches:
        return text
    
    # Sort matches in reverse order to avoid index shifting
    sorted_matches = sorted(matches, reverse=True)
    
    highlighted = text
    for match_pos in sorted_matches:
        before = highlighted[:match_pos]
        match = highlighted[match_pos:match_pos + pattern_length]
        after = highlighted[match_pos + pattern_length:]
        
        highlighted = f"{before}<mark style='background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;'>{match}</mark>{after}"
    
    return highlighted

def testing_tab():
    """Testing and validation tab."""
    
    st.header("🧪 Testing & Validation Suite")
    st.write("Comprehensive testing suite to validate algorithm correctness and performance.")
    
    # Test categories
    test_categories = {
        "Core Algorithm Tests": {
            "description": "Test basic Rabin-Karp functionality, edge cases, and correctness",
            "tests": ["Pattern matching accuracy", "Case sensitivity", "Hash function variants", "Statistics tracking"]
        },
        "Hash Function Tests": {
            "description": "Validate all hash function implementations and their properties", 
            "tests": ["Polynomial hash", "Simple hash", "DJB2 hash", "FNV hash", "Rolling hash accuracy"]
        },
        "Algorithm Comparison Tests": {
            "description": "Test all string matching algorithms for consistency and correctness",
            "tests": ["Naive algorithm", "KMP algorithm", "Boyer-Moore algorithm", "Z algorithm", "Result consistency"]
        },
        "Performance Tests": {
            "description": "Validate performance characteristics and stress testing",
            "tests": ["Large input handling", "Memory usage", "Scalability", "Concurrent execution"]
        }
    }
    
    # Display test categories
    st.subheader("📋 Available Test Categories")
    
    for category, info in test_categories.items():
        with st.expander(f"🔍 {category}"):
            st.write(f"**Description:** {info['description']}")
            st.write("**Test Coverage:**")
            for test in info['tests']:
                st.write(f"- {test}")
    
    # Test execution section
    st.subheader("🚀 Run Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Quick Tests")
        st.write("Run essential tests to verify core functionality")
        
        if st.button("🏃‍♂️ Run Quick Tests", type="primary"):
            run_quick_tests_ui()
    
    with col2:
        st.markdown("#### Full Test Suite")
        st.write("Run comprehensive test suite (may take longer)")
        
        if st.button("🧪 Run All Tests"):
            run_full_tests_ui()
    
    # Manual testing section
    st.subheader("🔧 Manual Testing Tools")
    
    st.markdown("#### Custom Test Case")
    st.write("Create your own test case to verify specific functionality:")
    
    test_col1, test_col2 = st.columns(2)
    
    with test_col1:
        custom_text = st.text_area(
            "Test Text:",
            value="Create your own test case here",
            height=100
        )
    
    with test_col2:
        custom_pattern = st.text_input("Test Pattern:", value="test")
        expected_matches = st.text_input(
            "Expected Match Positions (comma-separated):", 
            value="17",
            help="Enter expected positions where pattern should be found"
        )
    
    if st.button("🧪 Run Custom Test"):
        run_custom_test_ui(custom_text, custom_pattern, expected_matches)

def run_quick_tests_ui():
    """Run quick tests and display results in UI."""
    
    st.info("🏃‍♂️ Running quick tests...")
    
    try:
        # Simple validation tests without importing test modules
        rk = RabinKarp()
        
        # Test 1: Basic functionality
        result1 = rk.search("hello world", "world")
        test1_pass = result1['matches'] == [6]
        
        # Test 2: Multiple matches
        result2 = rk.search("abcabcabc", "abc")
        test2_pass = result2['matches'] == [0, 3, 6]
        
        # Test 3: No matches
        result3 = rk.search("hello", "xyz")
        test3_pass = result3['matches'] == []
        
        # Test 4: Case sensitivity
        result4 = rk.search("Hello", "hello", case_sensitive=False)
        test4_pass = result4['matches'] == [0]
        
        tests_passed = sum([test1_pass, test2_pass, test3_pass, test4_pass])
        
        if tests_passed == 4:
            st.success("✅ All 4 quick tests passed! Core functionality is working correctly.")
        else:
            st.error(f"❌ {4 - tests_passed} out of 4 tests failed.")
        
        # Show test details
        with st.expander("📋 Test Details"):
            st.write(f"✅ Basic search: {'PASS' if test1_pass else 'FAIL'}")
            st.write(f"✅ Multiple matches: {'PASS' if test2_pass else 'FAIL'}")
            st.write(f"✅ No matches: {'PASS' if test3_pass else 'FAIL'}")
            st.write(f"✅ Case insensitive: {'PASS' if test4_pass else 'FAIL'}")
    
    except Exception as e:
        st.error(f"❌ Error running quick tests: {e}")

def run_full_tests_ui():
    """Run full test suite and display results in UI."""
    
    st.info("🧪 Running comprehensive validation...")
    
    try:
        # Extended validation tests
        rk = RabinKarp()
        algorithms = StringMatchingAlgorithms()
        
        test_results = []
        
        # Test different algorithms for consistency
        text = "algorithm testing for consistency"
        pattern = "test"
        
        rk_result = rk.search(text, pattern)
        naive_result = algorithms.naive_search(text, pattern)
        kmp_result = algorithms.kmp_search(text, pattern)
        
        consistency_test = (rk_result['matches'] == naive_result['matches'] == kmp_result['matches'])
        test_results.append(("Algorithm Consistency", consistency_test))
        
        # Test hash functions
        hash_types = ["polynomial", "simple", "djb2", "fnv"]
        hash_test_pass = True
        
        for hash_type in hash_types:
            try:
                rk_hash = RabinKarp(hash_type=hash_type)
                result = rk_hash.search("test hash functions", "hash")
                if result['matches'] != [5]:
                    hash_test_pass = False
                    break
            except:
                hash_test_pass = False
                break
        
        test_results.append(("Hash Functions", hash_test_pass))
        
        # Test multi-pattern search
        try:
            multi_rk = MultiPatternRabinKarp()
            multi_result = multi_rk.search("test multiple patterns", ["test", "multiple"])
            multi_test_pass = (len(multi_result['matches']['test']) > 0 and 
                             len(multi_result['matches']['multiple']) > 0)
        except:
            multi_test_pass = False
        
        test_results.append(("Multi-Pattern Search", multi_test_pass))
        
        # Test edge cases
        edge_cases = [
            ("", "pattern", []),
            ("text", "", []),
            ("short", "verylongpattern", []),
            ("a", "a", [0])
        ]
        
        edge_test_pass = True
        for test_text, test_pattern, expected in edge_cases:
            try:
                result = rk.search(test_text, test_pattern)
                if result['matches'] != expected:
                    edge_test_pass = False
                    break
            except:
                edge_test_pass = False
                break
        
        test_results.append(("Edge Cases", edge_test_pass))
        
        # Display results
        passed_tests = sum(1 for _, passed in test_results if passed)
        total_tests = len(test_results)
        
        if passed_tests == total_tests:
            st.success(f"🎉 All {total_tests} test categories passed! Your implementation is robust!")
        else:
            st.error(f"❌ {total_tests - passed_tests} out of {total_tests} test categories failed.")
        
        # Show detailed results
        st.subheader("📊 Test Results by Category")
        
        for test_name, passed in test_results:
            status = "✅" if passed else "❌"
            st.write(f"{status} {test_name}: {'PASS' if passed else 'FAIL'}")
    
    except Exception as e:
        st.error(f"❌ Error running comprehensive tests: {e}")

def run_custom_test_ui(text, pattern, expected_positions_str):
    """Run a custom test case and display results."""
    
    try:
        # Parse expected positions
        if expected_positions_str.strip():
            expected_positions = [int(x.strip()) for x in expected_positions_str.split(',')]
        else:
            expected_positions = []
        
        # Run Rabin-Karp search
        rk = RabinKarp()
        result = rk.search(text, pattern)
        actual_positions = result['matches']
        
        # Compare results
        if actual_positions == expected_positions:
            st.success(f"✅ Test passed! Found matches at positions: {actual_positions}")
        else:
            st.error(f"❌ Test failed!")
            st.write(f"**Expected:** {expected_positions}")
            st.write(f"**Actual:** {actual_positions}")
        
        # Show additional details
        with st.expander("📋 Test Details"):
            st.write(f"**Text length:** {len(text)}")
            st.write(f"**Pattern length:** {len(pattern)}")
            st.write(f"**Comparisons:** {result['statistics']['comparisons']}")
            st.write(f"**Hash calculations:** {result['statistics']['hash_calculations']}")
            st.write(f"**Spurious hits:** {result['statistics']['spurious_hits']}")
    
    except ValueError:
        st.error("❌ Invalid expected positions format. Use comma-separated numbers.")
    except Exception as e:
        st.error(f"❌ Error running custom test: {e}")



if __name__ == "__main__":
    main()