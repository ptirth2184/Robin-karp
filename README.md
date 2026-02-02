# Rabin-Karp String Matching Algorithm

A comprehensive implementation of the Rabin-Karp string matching algorithm with interactive Streamlit visualization for Design and Analysis of Algorithms (DAA) course.

## 🚀 Features

### Phase 1 ✅
- ✅ Core Rabin-Karp algorithm implementation
- ✅ Rolling hash mechanism
- ✅ Spurious hit detection and handling
- ✅ Interactive Streamlit web interface
- ✅ Real-time pattern matching
- ✅ Algorithm performance statistics
- ✅ Match highlighting and context display

### Phase 2 ✅ (NEW!)
- ✅ **Step-by-step algorithm visualization**
- ✅ **Interactive algorithm walkthrough**
- ✅ **Performance comparison with naive algorithm**
- ✅ **Time complexity analysis and charts**
- ✅ **Hash collision analysis**
- ✅ **Algorithm theory and educational content**
- ✅ **Multiple sample texts for testing**
- ✅ **Hash calculator and parameter tuning**

## 📁 Project Structure

```
Robin-karp/
├── main.py                    # Streamlit web application (Enhanced with tabs)
├── rabin_karp.py             # Core algorithm implementation
├── utils.py                  # Utility functions
├── visualizer.py             # Step-by-step visualization (NEW!)
├── performance_analyzer.py   # Performance analysis & comparison (NEW!)
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Robin-karp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run main.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 🎯 How to Use

### 🔍 Basic Search Tab
1. **Enter Text**: Input the text you want to search in
2. **Enter Pattern**: Specify the pattern to search for
3. **Configure Settings**: Adjust hash function parameters in the sidebar
4. **Search**: Click the search button to find matches
5. **View Results**: See matches, statistics, and hash values

### 🎬 Step-by-Step Visualization Tab
1. **Enter shorter text** for better visualization experience
2. **Use navigation controls** to step through the algorithm
3. **Watch hash calculations** and window sliding in real-time
4. **See spurious hits** and actual matches highlighted
5. **Use auto-play** for automatic progression

### 📈 Performance Analysis Tab
1. **Compare algorithms** - Rabin-Karp vs Naive search
2. **View timing results** and efficiency metrics
3. **Analyze hash collisions** and distribution
4. **Study complexity charts** for different input sizes

### 📚 Algorithm Theory Tab
1. **Learn the theory** behind Rabin-Karp algorithm
2. **Understand hash functions** and parameter selection
3. **Explore real-world applications**
4. **Use the hash calculator** to experiment with values

## 🔧 Algorithm Parameters

- **Base**: Base value for polynomial hash function (default: 256)
- **Prime Modulus**: Prime number for modular arithmetic (default: 101)
- **Case Sensitivity**: Toggle case-sensitive/insensitive search

## 📊 Algorithm Statistics

The application tracks and displays:
- Total matches found
- Character comparisons performed
- Hash calculations executed
- Spurious hits detected
- Algorithm efficiency metrics

## 🎓 Educational Value

This implementation demonstrates:
- Rolling hash technique for efficient string matching
- Hash collision handling
- Time complexity analysis (O(n+m) average case)
- Space complexity optimization
- Real-world algorithm application

## 🔮 Upcoming Features (Future Phases)

- Step-by-step algorithm visualization
- Performance comparison with other algorithms
- Multiple pattern matching
- Advanced hash functions
- Comprehensive testing suite
- Interactive algorithm tutorial

## 👨‍💻 Author

DAA Course Project - Rabin-Karp Algorithm Implementation