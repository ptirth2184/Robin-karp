"""
Different hash function implementations for Rabin-Karp algorithm
"""

class HashFunction:
    """Base class for hash functions."""
    
    def __init__(self, base=256, prime=101):
        self.base = base
        self.prime = prime
    
    def calculate_hash(self, text, length):
        """Calculate hash for a string of given length."""
        raise NotImplementedError
    
    def rolling_hash(self, old_hash, old_char, new_char, power):
        """Calculate rolling hash."""
        raise NotImplementedError
    
    def calculate_power(self, length):
        """Calculate base^(length-1) % prime."""
        power = 1
        for _ in range(length - 1):
            power = (power * self.base) % self.prime
        return power

class PolynomialHash(HashFunction):
    """Polynomial rolling hash function."""
    
    def __init__(self, base=256, prime=101):
        super().__init__(base, prime)
        self.name = "Polynomial Hash"
        self.description = f"Hash = (c₀×{base}^(n-1) + c₁×{base}^(n-2) + ... + cₙ₋₁) mod {prime}"
    
    def calculate_hash(self, text, length):
        """Calculate polynomial hash."""
        hash_value = 0
        for i in range(length):
            hash_value = (hash_value * self.base + ord(text[i])) % self.prime
        return hash_value
    
    def rolling_hash(self, old_hash, old_char, new_char, power):
        """Calculate rolling hash using polynomial method."""
        new_hash = (old_hash - ord(old_char) * power) % self.prime
        new_hash = (new_hash * self.base + ord(new_char)) % self.prime
        return new_hash

class SimpleHash(HashFunction):
    """Simple additive hash function."""
    
    def __init__(self, base=256, prime=101):
        super().__init__(base, prime)
        self.name = "Simple Additive Hash"
        self.description = f"Hash = (sum of ASCII values × {base}) mod {prime}"
    
    def calculate_hash(self, text, length):
        """Calculate simple additive hash."""
        hash_value = 0
        for i in range(length):
            hash_value = (hash_value + ord(text[i]) * self.base) % self.prime
        return hash_value
    
    def rolling_hash(self, old_hash, old_char, new_char, power):
        """Calculate rolling hash for simple method."""
        new_hash = (old_hash - ord(old_char) * self.base + ord(new_char) * self.base) % self.prime
        return new_hash

class DJB2Hash(HashFunction):
    """DJB2 hash function variant."""
    
    def __init__(self, base=33, prime=101):
        super().__init__(base, prime)
        self.name = "DJB2 Hash"
        self.description = f"Hash = ((hash × {base}) + c) mod {prime}"
    
    def calculate_hash(self, text, length):
        """Calculate DJB2 hash."""
        hash_value = 5381  # DJB2 magic number
        for i in range(length):
            hash_value = ((hash_value * self.base) + ord(text[i])) % self.prime
        return hash_value
    
    def rolling_hash(self, old_hash, old_char, new_char, power):
        """Calculate rolling hash for DJB2."""
        # Remove old character contribution
        new_hash = (old_hash - ord(old_char) * power) % self.prime
        # Add new character
        new_hash = ((new_hash * self.base) + ord(new_char)) % self.prime
        return new_hash

class FNVHash(HashFunction):
    """FNV (Fowler-Noll-Vo) hash function."""
    
    def __init__(self, base=16777619, prime=101):
        super().__init__(base, prime)
        self.name = "FNV Hash"
        self.description = f"Hash = (hash × {base} ⊕ c) mod {prime}"
        self.fnv_offset = 2166136261  # FNV offset basis
    
    def calculate_hash(self, text, length):
        """Calculate FNV hash."""
        hash_value = self.fnv_offset
        for i in range(length):
            hash_value = (hash_value * self.base) % self.prime
            hash_value = hash_value ^ ord(text[i])
        return hash_value % self.prime
    
    def rolling_hash(self, old_hash, old_char, new_char, power):
        """Calculate rolling hash for FNV (simplified)."""
        # Simplified rolling for FNV - not as efficient as polynomial
        new_hash = (old_hash - ord(old_char) * power) % self.prime
        new_hash = (new_hash * self.base + ord(new_char)) % self.prime
        return new_hash

def get_hash_function(hash_type, base=256, prime=101):
    """
    Factory function to get hash function by type.
    
    Args:
        hash_type (str): Type of hash function
        base (int): Base value
        prime (int): Prime modulus
    
    Returns:
        HashFunction: Hash function instance
    """
    hash_functions = {
        "polynomial": PolynomialHash,
        "simple": SimpleHash,
        "djb2": DJB2Hash,
        "fnv": FNVHash
    }
    
    if hash_type.lower() in hash_functions:
        return hash_functions[hash_type.lower()](base, prime)
    else:
        return PolynomialHash(base, prime)  # Default

def compare_hash_functions(text, pattern_length, base=256, prime=101):
    """
    Compare different hash functions on the same text.
    
    Args:
        text (str): Text to analyze
        pattern_length (int): Length of patterns
        base (int): Base value
        prime (int): Prime modulus
    
    Returns:
        dict: Comparison results
    """
    if len(text) < pattern_length:
        return {}
    
    hash_types = ["polynomial", "simple", "djb2", "fnv"]
    results = {}
    
    for hash_type in hash_types:
        hash_func = get_hash_function(hash_type, base, prime)
        hash_values = []
        hash_counts = {}
        
        # Calculate hash for all windows
        for i in range(len(text) - pattern_length + 1):
            window = text[i:i + pattern_length]
            hash_val = hash_func.calculate_hash(window, pattern_length)
            hash_values.append(hash_val)
            
            if hash_val in hash_counts:
                hash_counts[hash_val] += 1
            else:
                hash_counts[hash_val] = 1
        
        # Calculate statistics
        total_windows = len(hash_values)
        unique_hashes = len(hash_counts)
        collisions = sum(1 for count in hash_counts.values() if count > 1)
        collision_rate = (total_windows - unique_hashes) / total_windows * 100 if total_windows > 0 else 0
        
        results[hash_type] = {
            'name': hash_func.name,
            'description': hash_func.description,
            'total_windows': total_windows,
            'unique_hashes': unique_hashes,
            'collision_groups': collisions,
            'collision_rate': collision_rate,
            'hash_values': hash_values[:20],  # First 20 for display
            'distribution_uniformity': calculate_uniformity(hash_values, prime)
        }
    
    return results

def calculate_uniformity(hash_values, prime):
    """
    Calculate how uniformly distributed the hash values are.
    
    Args:
        hash_values (list): List of hash values
        prime (int): Prime modulus (max possible hash value)
    
    Returns:
        float: Uniformity score (0-100, higher is better)
    """
    if not hash_values:
        return 0
    
    # Create buckets
    num_buckets = min(20, prime)
    bucket_size = prime // num_buckets
    buckets = [0] * num_buckets
    
    # Distribute hash values into buckets
    for hash_val in hash_values:
        bucket_idx = min(hash_val // bucket_size, num_buckets - 1)
        buckets[bucket_idx] += 1
    
    # Calculate uniformity (lower variance = higher uniformity)
    expected_per_bucket = len(hash_values) / num_buckets
    variance = sum((count - expected_per_bucket) ** 2 for count in buckets) / num_buckets
    
    # Convert to 0-100 scale (lower variance = higher score)
    max_variance = expected_per_bucket ** 2
    uniformity = max(0, 100 - (variance / max_variance * 100)) if max_variance > 0 else 100
    
    return round(uniformity, 2)