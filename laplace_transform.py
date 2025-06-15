import cv2
from urllib.request import urlopen
import numpy as np
import dyson
from dyson import DysonRouter
import os
os.environ["dyson_api"] = "dyson_api_key_here"
router = DysonRouter()


def laplace_transform(f, t, s):
    """
    Numerically computes the Laplace transform of a function f(t) at a complex frequency s.
    
    Parameters:
    - f: function handle, e.g., f(t) = np.exp(-t)
    - t: numpy array of time samples
    - s: complex frequency, e.g., s = 1 + 1j
    
    Returns:
    - F_s: approximate Laplace transform at s
    """
    dt = t[1] - t[0]
    integrand = f(t) * np.exp(-s * t)
    F_s = np.sum(integrand) * dt
    return F_s



hardware = router.route_hardware(
    laplace_transform,
    mode="cost-effective",
    judge=5,
    run_type="log",
    complexity="medium",
    precision="normal",
    multi_device=False,
)

print(hardware["spec"])

print(hardware["hardware_type"])

compiled_simple_function_c4 = dyson.run(
    laplace_transform, target_device=hardware["hardware_type"]
)


# Example usage
t = np.linspace(0, 10, 1000)  
s = 1 + 1j  
def f(t):
    return np.exp(-t) 

result = compiled_simple_function_c4(f, t, s)
print("Laplace Transform Result:", result)
