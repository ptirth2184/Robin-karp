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

### Phase 4 ✅ (NEW!)
- ✅ **Complete algorithm comparison suite** (Naive, KMP, Boyer-Moore, Z Algorithm)
- ✅ **Comprehensive benchmarking system**
- ✅ **Scalability analysis with different input sizes**
- ✅ **Performance trend analysis and recommendations**
- ✅ **Advanced performance metrics and visualizations**
- ✅ **Algorithm ranking and efficiency scoring**
- ✅ **Standardized test cases for fair comparison**
- ✅ **Configuration management system**

## 📁 Project Structure

```
Robin-karp/
├── main.py                    # Streamlit web application (5 comprehensive tabs)
├── rabin_karp.py             # Core algorithm implementation (Multi-pattern support)
├── utils.py                  # Utility functions
├── hash_functions.py         # Multiple hash function implementations
├── algorithms.py             # Complete algorithm comparison suite (NEW!)
├── visualizer.py             # Step-by-step visualization
├── performance_analyzer.py   # Advanced performance analysis (Enhanced!)
├── config.py                 # Configuration and settings (NEW!)
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
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

### 🎯 Multi-Pattern Search Tab
1. **Enter multiple patterns** (one per line)
2. **Choose search mode** - Individual vs Optimized
3. **View comprehensive results** for all patterns
4. **Analyze efficiency metrics** and success rates

### 📈 Performance Analysis Tab
1. **Basic Comparison** - Compare Rabin-Karp vs Naive algorithm
2. **Algorithm Comparison** - Compare with KMP, Boyer-Moore, Z Algorithm
3. **Benchmark Suite** - Run standardized performance tests
4. **Scalability Analysis** - Test performance with different input sizes
5. **Algorithm Recommendations** - Get suggestions based on your use case

### 🏆 Advanced Features
- **Complete Algorithm Suite**: 5 different string matching algorithms
- **Comprehensive Benchmarking**: Standardized test cases and metrics
- **Performance Visualization**: Charts and graphs for easy comparison
- **Scalability Testing**: Analyze performance trends with input size
- **Smart Recommendations**: Algorithm suggestions based on input characteristics

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
- **Rolling hash technique** for efficient string matching
- **Hash collision handling** and spurious hit detection
- **Algorithm comparison** and performance analysis
- **Time and space complexity** analysis with real data
- **Multiple algorithm implementations** for comprehensive understanding
- **Benchmarking methodologies** and performance metrics
- **Scalability analysis** and optimization techniques
- **Real-world algorithm application** and use case analysis

## 🔮 Project Phases

- **Phase 1** ✅ Core algorithm and basic UI
- **Phase 2** ✅ Visualization and analysis
- **Phase 3** ✅ Advanced features and multi-pattern search
- **Phase 4** ✅ Performance comparison and benchmarking
- **Phase 5** 🔄 Testing and validation (Next)
- **Phase 6** 📋 Documentation and polish (Final)

## 👨‍💻 Author

DAA Course Project - Rabin-Karp Algorithm Implementation