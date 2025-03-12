"""
dct_functions.py
Author: Allison Davis
Date: March 7, 2025

Functions to assist in the DCT image compression.
"""


### Import Modules ###
import numpy as np
import matplotlib.pyplot as plt
from cv2 import dct, idct
import os

def dct_block(block, num_coeffs: int):
    # A function to get the DCT coefficients of the block, and return the constrained coefficiebnts
    block = block - 128 # rescale
    coeffs = dct(block)
    lim_coeffs = zigzag_coeffs(coeffs, num_coeffs)  # use zigzag pattern to limit coefficients
    return lim_coeffs

def idct_block(coeffs, img_shape):
    # A function to inverse the DCT coefficients
    block = idct(coeffs) + 128  # rescale back
    block = np.clip(block, 0., 1.)  # clip values
    return block

def zigzag_coeffs(coeffs, num_coeffs: int):
    # This code is adapted from the following source: https://algocademy.com/blog/matrix-traversal-mastering-spiral-diagonal-and-zigzag-patterns/

    # Get the shape of the coeffs matrix
    rows, cols = coeffs.shape
    mask = np.zeros((rows, cols), dtype=np.float64)

    GET_COEFFS = True   # a flag to indicate whether the number of coefficients has been collected or not
    count = 0
    going_up = True
    row, col = 0, 0
    
    while GET_COEFFS:
        mask[row][col] = coeffs[row][col]
        if going_up:
            if row > 0 and col < cols - 1:  # check if at the top row and if there are more columns to cover
                # if yes, then keep moving up and right
                row -= 1
                col += 1
            else:
                # if no, then move down and left
                going_up = False
                if col == cols - 1:
                    row += 1
                else:
                    col += 1
        else:
            if row < rows - 1 and col > 0:
                # if yes, move down and left
                row += 1
                col -= 1
            else:
                # f no, move up and right
                going_up = True
                if row == rows - 1:
                    col += 1
                else:
                    row += 1
        count += 1
        
        # once enough coefficients are preserved, end the loop
        if count >= num_coeffs:
            GET_COEFFS = False

    return mask

def plot_dct_results(original_frame, predicted_frame, PSNR=None, coeffs=None, filepath=None, filename=None):
    """
    Saves an image made up of four subplots: anchor frame, target frame, predicted target frame, and motion field.
    Order of anchor and target frame depends on whether forward or backward estimation is occuring.

    """
    os.makedirs(filepath, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(7, 3), dpi=200)
    fig.suptitle(f'{coeffs} Coefficients DCT Image Prediction')

    axs[0].imshow(original_frame, cmap='gray')
    axs[0].set_title(f'Original Image')
    axs[0].axis('off')

    axs[1].imshow(predicted_frame, cmap='gray')
    axs[1].set_title(f'Predicted Image (PSNR: {PSNR:.4f})')
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{filepath}\{filename}.png')
    # plt.show()