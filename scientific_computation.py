import numpy as np
from dyson import DysonRouter
import os
import dyson

os.environ["dyson_api"] = "dyson_api_key_here"
router = DysonRouter()


def scientific_computation(n_points=10000):
    """
    Scientific computation example with numerical methods
    """

    # Monte Carlo integration of pi
    np.random.seed(42)
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside_circle) / n_points

    # Numerical differentiation
    def f(x):
        return np.sin(x) * np.exp(-x)

    x_vals = np.linspace(0, 5, n_points)
    y_vals = f(x_vals)

    # Numerical derivative
    dy_dx = np.gradient(y_vals, x_vals)

    # FFT analysis
    t = np.linspace(0, 1, n_points)
    signal = (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 10 * t)
        + np.random.normal(0, 0.1, n_points)
    )
    fft = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(n_points, t[1] - t[0])

    dominant_freq = frequencies[np.argmax(np.abs(fft[: n_points // 2]))]

    # Linear algebra operations
    A = np.random.randn(100, 100)
    eigenvalues = np.linalg.eigvals(A)
    condition_number = np.linalg.cond(A)

    return {
        "monte_carlo_pi": pi_estimate,
        "pi_error": abs(pi_estimate - np.pi),
        "max_derivative": float(np.max(np.abs(dy_dx))),
        "dominant_frequency": float(dominant_freq),
        "matrix_condition_number": float(condition_number),
        "max_eigenvalue": float(np.max(np.real(eigenvalues))),
        "computation_points": n_points,
        "signal_to_noise_ratio": float(np.var(signal) / 0.01),  # 0.01 is noise variance
    }


hardware = router.route_hardware(
    scientific_computation,
    mode="cost-effective",
    judge=5,
    run_type="log",
    complexity="medium",
    precision="normal",
    multi_device=False,
)

print(hardware["spec"])
print(hardware["hardware_type"])
compiled_function = dyson.run(scientific_computation, target_device="c3cpu")
# Example usage
result = compiled_function(n_points=10000)
print(result)
# Output the results
print(f"Estimated Pi: {result['monte_carlo_pi']}")
print(f"Error in Pi estimate: {result['pi_error']}")
print(f"Max derivative: {result['max_derivative']}")
print(f"Dominant frequency: {result['dominant_frequency']}")
print(f"Matrix condition number: {result['matrix_condition_number']}")
print(f"Max eigenvalue: {result['max_eigenvalue']}")
print(f"Computation points: {result['computation_points']}")
print(f"Signal to noise ratio: {result['signal_to_noise_ratio']}")
# This code performs a scientific computation involving Monte Carlo integration, numerical differentiation,
