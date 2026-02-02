"""
Configuration settings for the Rabin-Karp project
"""

# Default algorithm parameters
DEFAULT_HASH_PARAMS = {
    'polynomial': {'base': 256, 'prime': 101},
    'simple': {'base': 256, 'prime': 101},
    'djb2': {'base': 33, 'prime': 101},
    'fnv': {'base': 16777619, 'prime': 101}
}

# Prime numbers for hash functions
PRIME_NUMBERS = [
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
    199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
    263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
    317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
    383, 389, 397, 401, 409, 419, 421, 431, 433, 439,
    443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
    503, 509, 521, 523, 541, 547, 557, 563, 569, 571,
    577, 587, 593, 599, 601, 607, 613, 617, 619, 631,
    641, 643, 647, 653, 659, 661, 673, 677, 683, 691,
    701, 709, 719, 727, 733, 739, 743, 751, 757, 761,
    769, 773, 787, 797, 809, 811, 821, 823, 827, 829,
    839, 853, 857, 859, 863, 877, 881, 883, 887, 907,
    911, 919, 929, 937, 941, 947, 953, 967, 971, 977,
    983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033,
    1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093,
    1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
    1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229,
    1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291,
    1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367,
    1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439,
    1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489,
    1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559,
    1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613,
    1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693,
    1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753,
    1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831,
    1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901,
    1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
    1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039
]

# Benchmark configuration
BENCHMARK_CONFIG = {
    'small_text_size': 100,
    'medium_text_size': 1000,
    'large_text_size': 10000,
    'test_patterns': ['a', 'abc', 'pattern', 'longerpattern'],
    'repetitions': 3  # Number of times to run each test for averaging
}

# UI Configuration
UI_CONFIG = {
    'max_visualization_length': 100,
    'default_samples': {
        'Custom': ("The quick brown fox jumps over the lazy dog. The fox is quick and brown.", "fox"),
        'Lorem Ipsum': ("Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "dolor"),
        'Programming': ("def function(param): return param * 2", "param"),
        'DNA': ("ATCGATCGATCGATCGTAGCTAGCT", "ATCG"),
        'Repeated': ("abcabcabcdefabcabcghi", "abc")
    },
    'color_scheme': {
        'match': '#4CAF50',
        'hash_match': '#FF9800',
        'current_window': '#2196F3',
        'no_match': '#F5F5F5'
    }
}

# Algorithm information
ALGORITHM_INFO = {
    'Rabin-Karp': {
        'description': 'Uses rolling hash for efficient pattern matching',
        'time_complexity_avg': 'O(n+m)',
        'time_complexity_worst': 'O(n*m)',
        'space_complexity': 'O(1)',
        'best_for': ['Multiple patterns', 'Large alphabets', 'Rolling hash applications'],
        'invented': 1987,
        'inventors': ['Richard Karp', 'Michael Rabin']
    },
    'Naive (Brute Force)': {
        'description': 'Simple character-by-character comparison',
        'time_complexity_avg': 'O(n*m)',
        'time_complexity_worst': 'O(n*m)',
        'space_complexity': 'O(1)',
        'best_for': ['Small inputs', 'Simple implementation', 'Educational purposes'],
        'invented': 'Ancient',
        'inventors': ['Unknown']
    },
    'KMP (Knuth-Morris-Pratt)': {
        'description': 'Uses failure function to avoid redundant comparisons',
        'time_complexity_avg': 'O(n+m)',
        'time_complexity_worst': 'O(n+m)',
        'space_complexity': 'O(m)',
        'best_for': ['Guaranteed linear time', 'Repeated patterns', 'Streaming data'],
        'invented': 1977,
        'inventors': ['Donald Knuth', 'James Morris', 'Vaughan Pratt']
    },
    'Boyer-Moore (Simplified)': {
        'description': 'Scans pattern from right to left with bad character heuristic',
        'time_complexity_avg': 'O(n/m)',
        'time_complexity_worst': 'O(n*m)',
        'space_complexity': 'O(σ)',
        'best_for': ['Large patterns', 'Natural language text', 'Large alphabets'],
        'invented': 1977,
        'inventors': ['Robert Boyer', 'J Strother Moore']
    },
    'Z Algorithm': {
        'description': 'Uses Z array for linear time pattern matching',
        'time_complexity_avg': 'O(n+m)',
        'time_complexity_worst': 'O(n+m)',
        'space_complexity': 'O(n+m)',
        'best_for': ['Pattern preprocessing', 'Multiple queries', 'String analysis'],
        'invented': 1995,
        'inventors': ['Dan Gusfield']
    }
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'fast_time': 0.001,      # seconds
    'slow_time': 0.1,        # seconds
    'efficient_comparisons': 1000,
    'inefficient_comparisons': 10000
}

# Export settings
EXPORT_CONFIG = {
    'supported_formats': ['CSV', 'JSON', 'TXT'],
    'default_filename': 'rabin_karp_results',
    'include_metadata': True
}