import dyson
from dyson import DysonRouter
import os

os.environ["dyson_api"] = "dyson_api_key_here"
router = DysonRouter()
import jax.numpy as jnp
from jax import jit
import numpy as np


def h264_compression(frame1, frame2, block_size=8):
    """
    Performs H.264-like compression including DCT, quantization, motion estimation, and reconstruction.

    Parameters:
        frame1 (jax.numpy.ndarray): The current video frame (grayscale, 2D array).
        frame2 (jax.numpy.ndarray): The reference frame for motion estimation (grayscale, 2D array).
        block_size (int, optional): Size of the blocks used for DCT and motion estimation (default: 8).

    Returns:
        tuple:
            - reconstructed_frame (jax.numpy.ndarray): The frame reconstructed after compression.
            - motion_vectors (jax.numpy.ndarray): Motion vectors indicating displacement for each block.

    Steps:
        1. Apply 2D Discrete Cosine Transform (DCT) to each block in frame1.
        2. Quantize the DCT coefficients using a standard quantization matrix.
        3. Dequantize and apply inverse DCT to reconstruct the frame.
        4. Perform motion estimation using block matching between frame1 and frame2.
    """
    N = block_size

    @jit
    def calculate_dct():
        DCT_matrix = jnp.array(
            [
                [jnp.cos((2 * i + 1) * j * jnp.pi / (2 * N)) for j in range(N)]
                for i in range(N)
            ]
        )
        DCT_matrix = DCT_matrix * jnp.sqrt(2 / N)
        DCT_matrix = DCT_matrix.at[0].set(DCT_matrix[0] / jnp.sqrt(2))
        return DCT_matrix

    DCT_matrix = calculate_dct()

    def dct_2d(block):
        return jnp.dot(DCT_matrix, jnp.dot(block, DCT_matrix.T))

    def idct_2d(coeff):
        return jnp.dot(DCT_matrix.T, jnp.dot(coeff, DCT_matrix))

    def quantize(block, Q):
        return jnp.round(block / Q)

    def dequantize(block, Q):
        return block * Q

    Q_matrix = jnp.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )

    height, width = frame1.shape
    motion_vectors = jnp.zeros((height // block_size, width // block_size, 2))
    reconstructed_frame = jnp.zeros_like(frame1)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = frame1[i : i + block_size, j : j + block_size]
            dct_coeff = dct_2d(block)
            quantized = quantize(dct_coeff, Q_matrix)
            dequantized = dequantize(quantized, Q_matrix)
            reconstructed_block = idct_2d(dequantized)
            reconstructed_frame = reconstructed_frame.at[
                i : i + block_size, j : j + block_size
            ].set(reconstructed_block)

            best_match = jnp.array([0, 0])
            min_error = jnp.inf
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    ref_x, ref_y = i + dx, j + dy
                    is_valid_x = (ref_x >= 0) & (ref_x < height - block_size)
                    is_valid_y = (ref_y >= 0) & (ref_y < width - block_size)
                    if is_valid_x & is_valid_y:
                        candidate = frame2[
                            ref_x : ref_x + block_size, ref_y : ref_y + block_size
                        ]
                        error = jnp.sum(jnp.abs(block - candidate))
                        condition = error < min_error
                        min_error = jnp.where(condition, error, min_error)
                        best_match = jnp.where(
                            condition, jnp.array([dx, dy]), best_match
                        )

            motion_vectors = motion_vectors.at[i // block_size, j // block_size].set(
                best_match
            )

    return reconstructed_frame, motion_vectors


hardware = router.route_hardware(
    h264_compression,
    mode="cost-effective",
    judge=5,
    run_type="log",
    complexity="medium",
    precision="normal",
    multi_device=False,
)

print(hardware["spec"])

print(hardware["hardware_type"])

device = hardware["hardware_type"]

compiled_function = dyson.run(h264_compression, target_device=device)
# Example usage
frame1 = jnp.array(np.random.randint(0, 255, (64, 64)), dtype=np.float32)
frame2 = jnp.array(np.random.randint(0, 255, (64, 64)), dtype=np.float32)
reconstructed_frame, motion_vectors = compiled_function(frame1, frame2)
