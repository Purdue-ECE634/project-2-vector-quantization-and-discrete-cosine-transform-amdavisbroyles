"""
run_vector_quantization.py
Author: Allison Davis
Date: February 20, 2025

A program to run vector quantization, using the Generalized Lloyd Algorithm.

Program is intended to be run in terminal, with arguments parsed in.

"""

### Import Modules ###
import argparse
import os
import cv2
import numpy as np
import sys
import time
from skimage.metrics import structural_similarity as ssim

from helper_functions import extract_blocks, PSNR
from vector_quantization_functions import train_lbg_quantization, test_lbg_quantization, reconstruct_image, plot_vq_results, load_codebook

def main():
    # Setup command-line argument parsing 
    parser = argparse.ArgumentParser(description="Run Vector Quantization on target image.")
    parser.add_argument("--target_img", type=str,
                        default=f'C:\ece634\sample_image\\cat.png',
                        help="Target image to evaluate.")
    parser.add_argument("--block_size", type=int,
                        default=4,   
                        help="Block size to use in vector quantization")
    parser.add_argument("--load_codebook", type=bool,
                        default=False,
                        help="Set to True to load existing codebook.")
    parser.add_argument("--levels", type=int,
                        default=128,
                        help="Set number of quantization levels.")
    parser.add_argument("--num_training_imgs", type=int,
                        default=1,
                        help="Set the number of images to train ")
    parser.add_argument("--threshold", type=float,
                        default=1e-3,
                        help="set a threshold to determine convergence")
    parser.add_argument("--MAX_ITER", type=int,
                        default=50,
                        help="Set the maximum number of iterations to run for convergence.")
    parser.add_argument("--plot_filepath", type=str,   
                        default='experiments/vector_quantization/unique_vectors',                                      
                        help="Set path to save results to. Default will make an experiments folder in current working directory.")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    block_size = args.block_size
    L = args.levels
    num_training_imgs = args.num_training_imgs
    threshold = args.threshold
    max_iterations = args.MAX_ITER

    # Check if video sequence or folder exists; raise error if not found.
    if not os.path.exists(args.target_img):
        raise FileNotFoundError(f"The file {args.target_img} not found.")
    
    # Read the target img, get the image name and set filename for later use
    target_img = cv2.imread(args.target_img)
    img_path = os.path.dirname(args.target_img)
    img_name = os.path.basename(args.target_img)
    filename = f'{img_name}_BLOCK{block_size}_NTRAINING{num_training_imgs}_LEVELS{L}_THRESHOLD{threshold}_unique_vector_quantization'

    # Convert frames to grayscale & get frame of interest
    gray_target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # If using more than target image to train, then get a random list of file indices to randomly select training images
    if num_training_imgs > 1:
        filenames = os.listdir(f'C:/ece634/sample_image/')
        file_idx = np.random.choice(len(filenames), (num_training_imgs))

    original_stdout = sys.stdout    # for saving command terminal prints to a text file
    with open(f'{args.plot_filepath}/{filename}.txt', 'w') as f:   
        frames = []
        # if training from scratch
        if not args.load_codebook:
            sys.stdout = f          # set output to text file

            if num_training_imgs == 1:
                frames.append(gray_target_img)
            else:
                frames = []
                for _, idx in enumerate(file_idx):  # get the training images
                    file = filenames[idx]
                    print(f'File: {file}')
                    gray_frame = cv2.imread(f'{img_path}/{file}', cv2.IMREAD_GRAYSCALE)
                    frames.append(gray_frame)

            start_time = time.time()     # start time for algorithm
            blocks = []
            # extract the blocks from the images
            for frame in frames:
                blocks = extract_blocks(frame, block_size, blocks)
            
            # train codebook using blocks
            codebook = train_lbg_quantization(blocks, L, threshold, max_iterations)
        
        # if using existing codebook to quantize
        else:
            filename = filename +'_load_codebook'
            start_time = time.time()
            codebook_path = str(input("Enter path to codebook: "))  # get codebook path from user
            sys.stdout = f
            print(f'Loading Codebook: {codebook_path}')
            codebook = load_codebook(codebook_path)     # parse the codebook to be readable

        # after training or loading a codebook, test quantization on the target image
        test_blocks = []
        test_blocks = extract_blocks(gray_target_img, block_size, test_blocks)
        quantized_blocks = test_lbg_quantization(test_blocks, codebook)
        
        end_time = time.time()
        algorithm_time = end_time - start_time

        print(f'Quantization Time: {algorithm_time:.4f} seconds.')
        start_time = time.time()

        # reconstruct the image using the quantized blocks
        reconstructed_img = reconstruct_image(quantized_blocks, gray_target_img, block_size)
        end_time = time.time()
        print(f'Reconstruction Time: {end_time - start_time:.4f} seconds.')

        # calculate PSNR and SSIM
        psnr = PSNR(gray_target_img, reconstructed_img)
        ssim_score, _ = ssim(gray_target_img, reconstructed_img, full=True)
        print(f'PSNR of Reconstruction: {psnr:.4f} dB')
        print(f'SSIM Score: {ssim_score:.4f}')

        plot_vq_results(gray_target_img, reconstructed_img, num_training_imgs, PSNR=psnr, levels=L, filepath=args.plot_filepath, filename=filename+"_results")
        sys.stdout = original_stdout
    
    # if training from scratch, save the trained codebook to a text file
    if not args.load_codebook:
        with open(f"{args.plot_filepath}/{filename}_codebook.txt", "w") as file:
            for item in codebook:
                file.write(str(item)+"\n")

    cv2.imwrite(f'{args.plot_filepath}/{filename}_reconstruction.png', reconstructed_img)
    print('Done')

if __name__ == "__main__":
    main()