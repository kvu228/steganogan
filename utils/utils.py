import math

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