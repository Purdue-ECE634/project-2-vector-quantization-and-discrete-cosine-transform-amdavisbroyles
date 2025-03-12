"""
run_dct.py
Author: Allison Davis
Date: March 7, 2025

A program to run DCT on a target image and reconstruct using a limited number of coefficients.
"""

### Import Modules ###
import argparse
import os
import cv2
import numpy as np
import sys
import time
from skimage.metrics import structural_similarity as ssim

from helper_functions import extract_blocks, reconstruct_image, PSNR
from dct_functions import *

def main():
    # Setup command-line argument parsing 
    parser = argparse.ArgumentParser(description="Run Vector Quantization on target image.")
    parser.add_argument("--target_img", type=str,
                        default=f'C:\ece634\sample_image\cat.png',
                        help="Target image to evaluate.")
    parser.add_argument("--dct_size", type=int,
                        default=8,   
                        help="Size of DCT; assumes NxN")
    parser.add_argument("--K", type=int,
                        default=64,
                        help="Set number of coefficients to use for reconstruction.")
    parser.add_argument("--plot_filepath", type=str,   
                        default='experiments/dct',                                      
                        help="Set path to save results to. Default will make an experiments folder in current working directory.")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    dct_size = args.dct_size
    K = args.K

    # Check if video sequence or folder exists; raise error if not found.
    if not os.path.exists(args.target_img):
        raise FileNotFoundError(f"The file {args.target_img} not found.")
    
    target_img = cv2.imread(args.target_img)
    img_path = os.path.dirname(args.target_img)
    img_name = os.path.basename(args.target_img)
    filename = f'{img_name}_BLOCK{dct_size}_COEFFS{K}_dct'

    # Convert frames to grayscale & get frame of interest
    gray_target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Extract the blocks to use for DCT
    blocks = extract_blocks(gray_target_img, dct_size, [], flatten_block=False)
    recon_blocks = []
    for block in blocks:
        coeffs = dct_block(block, K)    # get the limited coefficients
        recon_block = idct_block(coeffs, block.shape)   # use inverse DCT to reconstruct the block
        recon_blocks.append(recon_block)

    # Reconstruct the image
    reconstructed_img = reconstruct_image(recon_blocks, gray_target_img, dct_size)
    
    # Calculate PSNR and SSIM
    psnr = PSNR(gray_target_img, reconstructed_img)
    ssim_score, _ = ssim(gray_target_img, reconstructed_img, full=True)
    print(f'PSNR of Reconstruction: {psnr:.4f} dB')
    print(f'SSIM Score: {ssim_score:.4f}')

    plot_dct_results(gray_target_img, reconstructed_img, PSNR=psnr, coeffs=K, filepath=args.plot_filepath, filename=f'{filename}_plot')
    cv2.imwrite(f'{args.plot_filepath}/{filename}_reconstruction.png', reconstructed_img)

if __name__ == "__main__":
    main()