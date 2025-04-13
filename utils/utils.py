import math
import random
import string

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(original, stego):
    psnr_value = psnr(original, stego)
    return psnr_value

def calculate_ssim(original, stego):
    ssim_index = ssim(original, stego, data_range=255,channel_axis=-1)
    return ssim_index


def test_max_message_lsb(image_path):
    # Load image with PIL to avoid OpenCV dependencies
    with Image.open(image_path) as img:
        width, height = img.size
        channels = len(img.getbands())

    # Calculate maximum message size (1 bit per channel per pixel)
    max_bits = width * height * channels
    return math.floor(max_bits / 8)

def test_max_message_dct(image_path):
    with Image.open(image_path) as img:
        width, height = img.size

    # DCT uses 1 bit per 8x8 block (from DC coefficient)
    blocks_x = width // 8
    blocks_y = height // 8
    return (blocks_x * blocks_y) // 8


def test_max_message_gan(image_path):
    with Image.open(image_path) as img:
        width, height = img.size

    # Assuming 4 bits per pixel (common in GAN-based steganography)
    return (width * height * 4) // 8


def create_random_string(num_bytes):
    characters = string.ascii_letters + string.digits

    # Generate a random string of characters
    random_string = ''.join(random.choices(characters, k=num_bytes))

    # Verify the byte length
    actual_bytes = len(random_string.encode('utf-8'))

    if actual_bytes != num_bytes:
        raise ValueError(f"Expected {num_bytes} bytes, but got {actual_bytes} bytes")

    return random_string