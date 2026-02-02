"""
Streamlit Web Application for Rabin-Karp String Matching Algorithm
"""

import streamlit as st
import pandas as pd
from rabin_karp import RabinKarp
from utils import format_matches
from visualizer import display_step_by_step_visualization, create_algorithm_flow_chart
from performance_analyzer import display_performance_analysis, create_complexity_comparison_chart

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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Basic Search", 
        "🎬 Step-by-Step Visualization", 
        "📈 Performance Analysis",
        "📚 Algorithm Theory"
    ])
    
    # Sidebar for algorithm parameters
    st.sidebar.header("⚙️ Algorithm Settings")
    
    base = st.sidebar.number_input(
        "Base (for hash function)", 
        min_value=2, 
        max_value=1000, 
        value=256,
        help="Base value for polynomial hash function"
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
        basic_search_tab(text, pattern, base, prime, case_sensitive)
    
    # Tab 2: Step-by-Step Visualization
    with tab2:
        visualization_tab(text, pattern, base, prime)
    
    # Tab 3: Performance Analysis
    with tab3:
        performance_tab(text, pattern, base, prime)
    
    # Tab 4: Algorithm Theory
    with tab4:
        theory_tab()

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

def basic_search_tab(default_text, default_pattern, base, prime, case_sensitive):
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
            rk = RabinKarp(base=base, prime=prime)
            results = rk.search(text, pattern, case_sensitive)
        
        # Display results
        display_results(text, pattern, results)

def visualization_tab(default_text, default_pattern, base, prime):
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

def performance_tab(default_text, default_pattern, base, prime):
    """Performance analysis tab."""
    
    st.header("📈 Performance Analysis")
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
    
    if st.button("🚀 Run Performance Analysis", type="primary"):
        if perf_text and perf_pattern:
            display_performance_analysis(perf_text, perf_pattern, base, prime)
        else:
            st.error("Please provide both text and pattern for analysis.")
    
    # Theoretical complexity comparison
    st.markdown("---")
    create_complexity_comparison_chart()

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
    
    2. **Hash Function**: We use a polynomial hash function:
       ```
       hash(s) = (s[0] × base^(m-1) + s[1] × base^(m-2) + ... + s[m-1]) mod prime
       ```
    
    3. **Spurious Hits**: When hash values match but strings don't, we need character-by-character verification.
    """)
    
    # Algorithm steps
    create_algorithm_flow_chart()
    
    # Advantages and disadvantages
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Advantages
        - **Average O(n+m) time complexity**
        - **Good for multiple pattern search**
        - **Works well with large alphabets**
        - **Simple to implement**
        - **Effective for long patterns**
        """)
    
    with col2:
        st.markdown("""
        ### ❌ Disadvantages
        - **Worst case O(nm) complexity**
        - **Spurious hits can degrade performance**
        - **Hash function choice affects performance**
        - **Not cache-friendly for very short patterns**
        - **Requires good hash parameters**
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
    """)
    
    # Hash function analysis
    st.markdown("""
    ### 🔢 Hash Function Parameters
    
    **Base Value**: Typically 256 (number of ASCII characters)
    - Larger base → better distribution but higher values
    - Should be larger than alphabet size
    
    **Prime Modulus**: A large prime number (e.g., 101, 1009, 10007)
    - Reduces hash collisions
    - Should be large enough to minimize spurious hits
    - Common choices: 101, 1009, 1000000007
    """)
    
    # Interactive hash calculator
    st.markdown("### 🧮 Hash Calculator")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        calc_text = st.text_input("Text to hash:", value="hello")
        calc_base = st.number_input("Base:", value=256, min_value=2)
    
    with calc_col2:
        calc_prime = st.number_input("Prime:", value=101, min_value=2)
        
        if calc_text:
            from utils import calculate_hash
            hash_val = calculate_hash(calc_text, len(calc_text), calc_base, calc_prime)
            st.metric("Hash Value", hash_val)

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

if __name__ == "__main__":
    main()