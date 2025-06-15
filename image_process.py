import cv2
from urllib.request import urlopen
import numpy as np
import dyson
from dyson import DysonRouter
import os

os.environ["dyson_api"] = "dyson_api_key_here"
router = DysonRouter()


def process_video_frame():
    """
    Reads a video frame from an image file, processes it (converts to grayscale),
    and writes the processed frame to another file.

    Parameters:
    - input_path (str): Path to the input image file.
    - output_path (str): Path to save the processed image.
    """

    # Read the image (simulating a video frame)
    image_url = "https://avatars.githubusercontent.com/u/170319640?s=200&v=4"
    resp = urlopen(image_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # The image object

    # Convert to grayscale (basic transformation)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the processed frame
    return gray_frame


hardware = router.route_hardware(
    process_video_frame,
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
    process_video_frame, target_device=hardware["hardware_type"]
)

print(compiled_simple_function_c4())
