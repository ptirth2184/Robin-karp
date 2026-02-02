"""
Step-by-step visualization for Rabin-Karp algorithm
"""

import streamlit as st
import pandas as pd
import time

class RabinKarpVisualizer:
    def __init__(self, text, pattern, base=256, prime=101):
        """
        Initialize visualizer for Rabin-Karp algorithm.
        
        Args:
            text (str): Text to search in
            pattern (str): Pattern to search for
            base (int): Base for hash function
            prime (int): Prime modulus
        """
        self.text = text
        self.pattern = pattern
        self.base = base
        self.prime = prime
        self.steps = []
        self.current_step = 0
    
    def generate_steps(self):
        """Generate all visualization steps for the algorithm."""
        from utils import calculate_hash, calculate_power, rolling_hash
        
        if len(self.pattern) > len(self.text) or len(self.pattern) == 0:
            return []
        
        pattern_length = len(self.pattern)
        text_length = len(self.text)
        
        # Step 1: Calculate pattern hash
        pattern_hash = calculate_hash(self.pattern, pattern_length, self.base, self.prime)
        self.steps.append({
            'step_number': 1,
            'action': 'Calculate Pattern Hash',
            'description': f"Calculate hash for pattern '{self.pattern}'",
            'pattern_hash': pattern_hash,
            'text_hash': None,
            'position': -1,
            'window': '',
            'match_found': False,
            'hash_match': False,
            'calculation': self._show_hash_calculation(self.pattern)
        })
        
        # Step 2: Calculate first window hash
        first_window = self.text[:pattern_length]
        text_hash = calculate_hash(first_window, pattern_length, self.base, self.prime)
        hash_match = text_hash == pattern_hash
        match_found = hash_match and first_window == self.pattern
        
        self.steps.append({
            'step_number': 2,
            'action': 'Calculate First Window Hash',
            'description': f"Calculate hash for first window '{first_window}'",
            'pattern_hash': pattern_hash,
            'text_hash': text_hash,
            'position': 0,
            'window': first_window,
            'match_found': match_found,
            'hash_match': hash_match,
            'calculation': self._show_hash_calculation(first_window)
        })
        
        # Calculate power for rolling hash
        power = calculate_power(self.base, pattern_length, self.prime)
        
        # Steps 3+: Slide window and calculate rolling hash
        step_num = 3
        for i in range(1, text_length - pattern_length + 1):
            window = self.text[i:i + pattern_length]
            
            # Calculate rolling hash
            old_char = self.text[i - 1]
            new_char = self.text[i + pattern_length - 1]
            text_hash = rolling_hash(text_hash, old_char, new_char, power, self.base, self.prime)
            
            hash_match = text_hash == pattern_hash
            match_found = hash_match and window == self.pattern
            
            self.steps.append({
                'step_number': step_num,
                'action': 'Rolling Hash',
                'description': f"Slide window to position {i}, calculate rolling hash",
                'pattern_hash': pattern_hash,
                'text_hash': text_hash,
                'position': i,
                'window': window,
                'match_found': match_found,
                'hash_match': hash_match,
                'old_char': old_char,
                'new_char': new_char,
                'rolling_calculation': self._show_rolling_calculation(text_hash, old_char, new_char, power)
            })
            step_num += 1
        
        return self.steps
    
    def _show_hash_calculation(self, text):
        """Show detailed hash calculation for a string."""
        calculation = f"Hash calculation for '{text}':\n"
        hash_val = 0
        
        for i, char in enumerate(text):
            old_hash = hash_val
            hash_val = (hash_val * self.base + ord(char)) % self.prime
            calculation += f"Step {i+1}: ({old_hash} × {self.base} + {ord(char)}) mod {self.prime} = {hash_val}\n"
        
        return calculation
    
    def _show_rolling_calculation(self, new_hash, old_char, new_char, power):
        """Show rolling hash calculation details."""
        return f"Rolling hash calculation:\n" \
               f"Remove '{old_char}' (ASCII: {ord(old_char)})\n" \
               f"Add '{new_char}' (ASCII: {ord(new_char)})\n" \
               f"Power factor: {power}\n" \
               f"New hash: {new_hash}"

def display_step_by_step_visualization(text, pattern, base=256, prime=101):
    """Display step-by-step visualization of Rabin-Karp algorithm."""
    
    st.subheader("🎬 Step-by-Step Algorithm Visualization")
    
    # Initialize visualizer
    visualizer = RabinKarpVisualizer(text, pattern, base, prime)
    steps = visualizer.generate_steps()
    
    if not steps:
        st.warning("No steps to visualize. Check your input.")
        return
    
    # Controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⏮️ First Step"):
            st.session_state.current_step = 0
    
    with col2:
        current_step = st.slider(
            "Step", 
            0, 
            len(steps) - 1, 
            value=st.session_state.get('current_step', 0),
            key='step_slider'
        )
        st.session_state.current_step = current_step
    
    with col3:
        if st.button("⏭️ Last Step"):
            st.session_state.current_step = len(steps) - 1
    
    # Navigation buttons
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("⏪ Previous") and current_step > 0:
            st.session_state.current_step = current_step - 1
            st.rerun()
    
    with nav_col2:
        if st.button("▶️ Next") and current_step < len(steps) - 1:
            st.session_state.current_step = current_step + 1
            st.rerun()
    
    with nav_col3:
        auto_play = st.button("🎮 Auto Play")
    
    with nav_col4:
        if st.button("🔄 Reset"):
            st.session_state.current_step = 0
            st.rerun()
    
    # Auto play functionality
    if auto_play:
        placeholder = st.empty()
        for i in range(current_step, len(steps)):
            st.session_state.current_step = i
            with placeholder.container():
                display_current_step(steps[i], text, pattern)
            time.sleep(1.5)
        st.rerun()
    
    # Display current step
    if 0 <= current_step < len(steps):
        display_current_step(steps[current_step], text, pattern)

def display_current_step(step, text, pattern):
    """Display details of the current algorithm step."""
    
    # Step header
    st.markdown(f"### Step {step['step_number']}: {step['action']}")
    st.write(step['description'])
    
    # Visual representation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text visualization with current window highlighted
        if step['position'] >= 0:
            before = text[:step['position']]
            window = step['window']
            after = text[step['position'] + len(pattern):]
            
            # Color coding based on match status
            if step['match_found']:
                color = "#4CAF50"  # Green for match
                icon = "✅"
            elif step['hash_match']:
                color = "#FF9800"  # Orange for hash match (spurious hit)
                icon = "⚠️"
            else:
                color = "#2196F3"  # Blue for current window
                icon = "🔍"
            
            highlighted_text = f"{before}<span style='background-color: {color}; padding: 2px 4px; border-radius: 3px; color: white;'>{window}</span>{after}"
            st.markdown(f"**Text:** {highlighted_text}", unsafe_allow_html=True)
            st.markdown(f"**Current Window:** `{window}` {icon}")
        else:
            st.markdown(f"**Pattern:** `{pattern}`")
    
    with col2:
        # Hash values
        st.markdown("**Hash Values:**")
        if step['pattern_hash'] is not None:
            st.write(f"Pattern Hash: `{step['pattern_hash']}`")
        if step['text_hash'] is not None:
            st.write(f"Window Hash: `{step['text_hash']}`")
        
        # Match status
        if step['position'] >= 0:
            if step['hash_match']:
                if step['match_found']:
                    st.success("✅ Match Found!")
                else:
                    st.warning("⚠️ Hash Match (Spurious Hit)")
            else:
                st.info("🔍 No Hash Match")
    
    # Detailed calculations
    with st.expander("🔢 View Hash Calculations"):
        if 'calculation' in step:
            st.code(step['calculation'])
        if 'rolling_calculation' in step:
            st.code(step['rolling_calculation'])

def create_algorithm_flow_chart():
    """Create a visual flow chart of the Rabin-Karp algorithm."""
    
    st.subheader("📊 Algorithm Flow Chart")
    
    flow_steps = [
        "1. Calculate hash of pattern",
        "2. Calculate hash of first text window",
        "3. Compare hash values",
        "4. If hashes match → Verify character by character",
        "5. Slide window by one position",
        "6. Calculate rolling hash for new window",
        "7. Repeat steps 3-6 until end of text"
    ]
    
    for i, step in enumerate(flow_steps):
        if i == 0:
            st.markdown(f"🟢 **{step}**")
        elif "Repeat" in step:
            st.markdown(f"🔄 **{step}**")
        elif "Verify" in step:
            st.markdown(f"🔍 **{step}**")
        else:
            st.markdown(f"➡️ **{step}**")
    
    # Algorithm complexity info
    st.info("""
    **Time Complexity:**
    - Average Case: O(n + m)
    - Worst Case: O(nm) (when many spurious hits occur)
    
    **Space Complexity:** O(1)
    
    Where n = text length, m = pattern length
    """)