"""
helper_functions.py
Functions to use in vector quantization and discrete cosine transforms.

Author: Allison Davis
Date: March 01, 2025

"""

### Import Modules ###
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt


def extract_blocks(image: np.ndarray, block_size: int, blocks_list: list, flatten_block=True):
    """
    A function to divide the input image up into equal size blocks.
    Inputs:
        --image (np.array float32): reference frame to divide
        --block_size (int): desired block size as a single integer. Assumes blocks are divided as MxM squares.
    Returns:
        --blocks (list): divided MxM subimages from anchor frame
        --idxs (list): top-left index corresponding to the position of a block in the anchor frame.
    """
    # Determine how many blocks per image width and image height
    img_height, img_width = image.shape
    num_blocks_x = int(img_height // block_size)
    num_blocks_y = int(img_width // block_size)

    blocks = blocks_list         # store the blocks

   # Iterate through the number of blocks in the x, y directions
    for row_idx in range(num_blocks_x):
        for col_idx in range(num_blocks_y):
            start_x, start_y = row_idx * block_size, col_idx * block_size         # calculate the starting point of the block
            block = image[start_x:start_x+block_size, start_y:start_y+block_size].astype(np.float64)
            if flatten_block:
                block = block.flatten()
            block = block / 255.    # Normalize block to [0, 1]
            blocks.append(block)

    print(f"Number of blocks: {len(blocks)}")
    return blocks


def error_calc(training_vector, codeword):
    difference = np.subtract(training_vector, codeword)
    error = np.sum(difference**2)
    return error

def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1.T - vector2)**2, axis=1))

def reconstruct_image(blocks, original_img, block_size):

    img_height, img_width = original_img.shape
    num_blocks_x = int(img_height // block_size)
    num_blocks_y = int(img_width // block_size)

    reconstructed_img = np.empty((original_img.shape))

    # Iterate through the number of blocks in the x, y directions
    block_num = 0 
    for row_idx in range(num_blocks_x):
        for col_idx in range(num_blocks_y):
            start_x, start_y = row_idx * block_size, col_idx * block_size         # calculate the starting point of the block
            block = blocks[block_num] * 255
            block = np.reshape(block, (block_size, block_size))
            reconstructed_img[start_x:start_x+block_size, start_y:start_y+block_size] = block

            block_num += 1

    return reconstructed_img.astype(np.uint8)

def PSNR(target_frame: np.ndarray, predicted_frame: np.ndarray):
    """
    A function to calculate the peak signal-to-noise ratio between the ground truth and predicted image. Assumes max value possible in image is 255.
    Inputs:
        --target_frame (np.array uint8): frame of interest to reconstruct
        --predicted_frame (np.array uint8): reconstructed image
    Returns:
        --PSNR (float): calculated peak signal-to-noise ratio
    """
    # Convert images to float32
    target_frame = target_frame.astype(np.float32)
    predicted_frame = predicted_frame.astype(np.float32)
    # Calculate the mean square error
    mse = np.mean((target_frame - predicted_frame)**2)
    # Calculate PSNR
    PSNR = 20*log10(255.0/sqrt(mse))
    return PSNR