import zlib
from math import exp
from typing import Union, List, Optional

import torch
from reedsolo import RSCodec, ReedSolomonError
from torch.nn.functional import conv2d

# Initialize Reed-Solomon error correction with 250 parity symbols
reed_solomon = RSCodec(250)

def encode_text_to_bits(text: str) -> List[int]:
    """Encodes text to a bit array with compression and error correction.

    Args:
        text: Input string to encode

    Returns:
        List of bits (0s and 1s) representing the encoded text
    """
    return bytearray_to_bits(compress_and_protect_text(text))

def decode_bits_to_text(bits: List[int]) -> Optional[str]:
    """Decodes bit array to text with error correction and decompression.

    Args:
        bits: List of bits (0s and 1s) to decode

    Returns:
        Decoded string or None if decoding fails
    """
    return recover_and_decompress_text(bits_to_bytearray(bits))

def bytearray_to_bits(data: bytearray) -> List[int]:
    """Converts bytearray to list of bits (MSB first)."""
    bit_stream = []
    for byte in data:
        # Convert each byte to 8-bit binary representation
        bit_stream.extend([(byte >> shift) & 1 for shift in range(7, -1, -1)])
    return bit_stream

def bits_to_bytearray(bits: List[int]) -> bytearray:
    """Converts bit list to bytearray."""
    byte_stream = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        # Pack 8 bits into a byte
        byte_stream.append(sum(bit << (7 - pos) for pos, bit in enumerate(byte_bits)))
    return byte_stream

def compress_and_protect_text(text: str) -> bytearray:
    """Compresses text and adds error correction codes."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    compressed = zlib.compress(text.encode("utf-8"))
    return reed_solomon.encode(compressed)

def recover_and_decompress_text(data: bytearray) -> Optional[str]:
    """Attempts error correction and decompresses protected text."""
    try:
        # Handle Reed-Solomon decoding
        candidate = reed_solomon.decode(data)
        if isinstance(candidate, tuple):
            candidate = candidate[0]
        decoded_text = zlib.decompress(candidate).decode("utf-8")
        return decoded_text
    except (ReedSolomonError, zlib.error, UnicodeDecodeError):
        return None

def create_gaussian_kernel(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    """Generates 2D Gaussian kernel for SSIM calculation."""
    # Create 1D Gaussian distribution
    axis = torch.arange(window_size) - window_size // 2
    gaussian_1d = torch.exp(-axis**2 / (2 * sigma**2))
    gaussian_1d /= gaussian_1d.sum()  # Normalize

    # Create 2D kernel through outer product
    return gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)

def structural_similarity_index(
        image1: torch.Tensor,
        image2: torch.Tensor,
        window_size: int = 11,
        channel: int = 3,
        return_map: bool = False
) -> Union[torch.Tensor, float]:
    """Calculates Structural Similarity Index (SSIM) between two images.

    Args:
        image1: Batch of images (N, C, H, W)
        image2: Batch of images (N, C, H, W)
        window_size: Size of Gaussian kernel
        channel: Number of input channels
        return_map: Return full SSIM map instead of mean value

    Returns:
        SSIM value or map between images
    """
    # Validate inputs
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions")

    # Create Gaussian window
    kernel = create_gaussian_kernel(window_size).to(image1.device)
    kernel = kernel.expand(channel, 1, window_size, window_size)

    # Calculate means
    padding = window_size // 2
    mean1 = conv2d(image1, kernel, padding=padding, groups=channel)
    mean2 = conv2d(image2, kernel, padding=padding, groups=channel)

    # Calculate variances and covariance
    mean_product = mean1 * mean2
    var1 = conv2d(image1**2, kernel, padding=padding, groups=channel) - mean1**2
    var2 = conv2d(image2**2, kernel, padding=padding, groups=channel) - mean2**2
    covariance = conv2d(image1*image2, kernel, padding=padding, groups=channel) - mean_product

    # Stability constants
    C1 = (0.01 * 255)**2  # Luminance stability
    C2 = (0.03 * 255)**2  # Contrast stability

    # SSIM components
    luminance = (2 * mean_product + C1) / (mean1**2 + mean2**2 + C1)
    structure = (2 * covariance + C2) / (var1 + var2 + C2)

    ssim_map = luminance * structure

    return ssim_map.mean() if not return_map else ssim_map
