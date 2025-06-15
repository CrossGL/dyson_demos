# Dyson

A high-performance hardware routing and optimization library for scientific computing and computational physics applications.

## Overview

Dyson is a Python package that intelligently routes computational workloads to optimal hardware configurations, providing automatic optimization for complex mathematical operations like solving differential equations, linear algebra operations, and physics simulations.

## Installation

```bash
pip install dyson
```

## Quick Start

### 1. Basic Setup

```python
import dyson
from dyson import DysonRouter
import os
import numpy as np

# Set your API key
os.environ["dyson_api"] = "your_dyson_api_key_here"

# Initialize the router
router = DysonRouter()
```

### 2. Define Your Function

```python
def your_computational_function(matrix_a, matrix_b, vector_c, tol=1e-6, max_iter=1000):
    """
    Your computational function that needs optimization
    """
    result = matrix_a
    for _ in range(max_iter):
        result_new = matrix_a @ (matrix_b @ result)
        result_new /= np.linalg.norm(result_new)
        if np.linalg.norm(result_new - result) < tol:
            break
        result = result_new
    return result
```

### 3. Route Hardware

```python
# Route your function to optimal hardware
hardware = router.route_hardware(
    your_computational_function,
    mode="cost-effective",        # Options: "cost-effective", "performance", "balanced"
    judge=5,                      # Optimization level (1-10)
    run_type="log",              # Options: "log", "silent", "verbose"
    complexity="medium",          # Options: "low", "medium", "high"
    precision="normal",           # Options: "low", "normal", "high"
    multi_device=False,          # Enable multi-device computation
)

print("Hardware Specification:", hardware["spec"])
print("Hardware Type:", hardware["hardware_type"])
```

### 4. Compile and Run

```python
# Compile your function for the target hardware
compiled_function = dyson.run(
    your_computational_function, 
    target_device=hardware["hardware_type"]
)

# Use your optimized function
result = compiled_function(matrix_a, matrix_b, vector_c)
```

## Complete Example: Bethe-Salpeter Equation Solver

```python
import cv2
from urllib.request import urlopen
import numpy as np
import dyson
from dyson import DysonRouter
import os

# Setup
os.environ["dyson_api"] = "your_dyson_api_key_here"
router = DysonRouter()

def bethe_salpeter_equation(G0, K, psi0, tol=1e-6, max_iter=1000):
    """
    Iteratively solves the Bethe-Salpeter equation Psi = G0 * K * Psi.
    
    Parameters:
    - G0: Free two-particle propagator matrix [N, N]
    - K: Interaction kernel matrix [N, N]
    - psi0: Initial guess for the bound state wavefunction [N]
    - tol: Convergence tolerance
    - max_iter: Maximum number of iterations
    
    Returns:
    - psi: Approximate bound state wavefunction
    """
    psi = psi0.copy()
    for _ in range(max_iter):
        psi_new = G0 @ (K @ psi)
        psi_new /= np.linalg.norm(psi_new)  # Normalize
        if np.linalg.norm(psi_new - psi) < tol:
            break
        psi = psi_new
    return psi

# Route to optimal hardware
hardware = router.route_hardware(
    bethe_salpeter_equation,
    mode="cost-effective",
    judge=5,
    run_type="log",
    complexity="medium",
    precision="normal",
    multi_device=False,
)

# Compile for target hardware
compiled_bse_solver = dyson.run(
    bethe_salpeter_equation, 
    target_device=hardware["hardware_type"]
)

# Example usage
N = 100
G0 = np.eye(N)
K = np.random.rand(N, N) * 0.1
psi0 = np.random.rand(N)

# Solve using optimized function
psi = compiled_bse_solver(G0, K, psi0)
print("BSE Solution (norm):", np.linalg.norm(psi))
```

## API Reference

### DysonRouter

The main class for hardware routing and optimization.

```python
router = DysonRouter()
```

### router.route_hardware()

Routes computational functions to optimal hardware configurations.

**Parameters:**
- `function`: The function to optimize
- `mode`: Optimization mode
  - `"cost-effective"`: Prioritizes cost efficiency
  - `"performance"`: Prioritizes maximum performance
  - `"balanced"`: Balances cost and performance
- `judge`: Optimization level (1-10, higher = more aggressive optimization)
- `run_type`: Logging level
  - `"log"`: Standard logging
  - `"silent"`: No output
  - `"verbose"`: Detailed logging
- `complexity`: Computational complexity hint
  - `"low"`: Simple operations
  - `"medium"`: Moderate complexity
  - `"high"`: Complex operations
- `precision`: Required precision level
  - `"low"`: Faster, less precise
  - `"normal"`: Standard precision
  - `"high"`: Maximum precision
- `multi_device`: Enable multi-device computation (bool)

**Returns:**
Dictionary containing:
- `"spec"`: Hardware specification details
- `"hardware_type"`: Target hardware type for compilation

### dyson.run()

Compiles and optimizes functions for target hardware.

**Parameters:**
- `function`: The function to compile
- `target_device`: Target hardware type (from route_hardware result)

**Returns:**
Optimized, compiled version of the input function

## Configuration

### API Key Setup

Set your Dyson API key as an environment variable:

```python
import os
os.environ["dyson_api"] = "your_api_key_here"
```

Or set it in your shell:
```bash
export dyson_api="your_api_key_here"
```

## Best Practices

1. **Function Design**: Ensure your functions are compatible with hardware acceleration (avoid complex control flow when possible)

2. **Data Types**: Use NumPy arrays for optimal performance

3. **Memory Management**: Consider memory usage when setting `complexity` parameter

4. **Iterative Algorithms**: Dyson works particularly well with iterative mathematical algorithms

5. **Benchmarking**: Test both original and compiled versions to measure performance gains

## Use Cases

- **Scientific Computing**: Linear algebra, differential equations, optimization problems
- **Physics Simulations**: Quantum mechanics, statistical mechanics, field theory
- **Machine Learning**: Custom numerical algorithms, specialized computations
- **Signal Processing**: Image processing, signal analysis, filtering operations

## Support

For issues, questions, or feature requests, join our  [discord chanel](https://discord.com/invite/uyRQKXhcyW) .

## License

[License](https://github.com/CrossGL/dyson_demos/blob/main/LICENSE)

---

**Note**: Replace `"your_dyson_api_key_here"` with your actual Dyson API key before running the examples.
