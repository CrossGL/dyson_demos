import cv2
from urllib.request import urlopen
import numpy as np
import dyson
from dyson import DysonRouter
import os
os.environ["dyson_api"] = "dyson_api_key_here"
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



hardware = router.route_hardware(
    bethe_salpeter_equation,
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
    bethe_salpeter_equation, target_device=hardware["hardware_type"]
)


# Example system (toy data)
N = 100
G0 = np.eye(N)
K = np.random.rand(N, N) * 0.1
psi0 = np.random.rand(N)

psi = bethe_salpeter_equation(G0, K, psi0)

psi = compiled_simple_function_c4(G0, K, psi0)
print("BSE Solution (norm):", np.linalg.norm(psi))