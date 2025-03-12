"""
vector_quantization_functions.py
Author: Allison Davis
Date: February 28, 2025

Functions for use in vector quantization.
"""

### Import Modules ###
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time

def train_lbg_quantization(blocks, L, threshold=1e-5, MAX_ITERS=100):
    """
    A function to train the generalized Lloyd algorithm
    Inputs: blocks, levels, threshold, and maximum iterations to run.
    Output: trained codebook
    """
    total_vectors = len(blocks)
    print(f'Total Number of Vectors: {total_vectors}')

    # Get only the unique vectors to train on. Reduce repetition
    blocks = np.unique(blocks, axis=0).tolist()
    print(f'Total Number of Unique Vectors: {len(blocks)}')
    total_vectors = len(blocks)

    # Initialize codebook g_l with random values
    g_l = random.sample(blocks, L)

    # Get the intial partition regions
    print('Finding Initial Partition Regions')
    partition_start = time.time()
    B_l = find_partition_regions(blocks, g_l, L)
    partition_end = time.time()
    print(f'Time to Partition: {(partition_end-partition_start):.4f} (s)')

    # Get initial distortion
    print('Getting Initial Distortion')
    error = 0
    for level, B in enumerate(B_l):
        B_array = np.array(B)
        if B_array.size == 0:
            error += 0          # if there are no vectors in the partition region, it contributes no error
        else:
            error += error_calc(B_array, g_l[level])
    distortion_0 = error / total_vectors
    
    # Train codebook
    iter = 0
    convergence = float('inf')  # set initial convergence value
    while convergence > threshold:
        print(f'Iter:{iter}')
        g_t = update_codewords(B_l, g_l)    # update the codewords based on the partition regions
        B_l = find_partition_regions(blocks, g_t, L)    # find new partition regions
        
        # calculate new distortion
        error = 0
        for level, B in enumerate(B_l):
            B_array = np.array(B)
            if B_array.size == 0:
                error += 0
            else:
                error += error_calc(B_array, g_t[level])
        distortion_1 = error / total_vectors

        # check convergence values
        convergence_new = np.abs(distortion_0 - distortion_1) / distortion_0
        print(f'Convergence: {convergence_new:,.7f}')
        if convergence_new < convergence:
            print('Convergence towards threshold')
        else:
            print('Divergence from threshold')
        distortion_0 = distortion_1
        convergence = convergence_new

        iter += 1
        # if reached MAX_ITERS, timeout the program.
        if iter >= MAX_ITERS:
            raise TimeoutError("Reached maximum iterations before converging below threshold.")
    print('Finished codebook')
    return g_t

def find_partition_regions(blocks, codebook, L):
    """
    A function to find the partition regions
    Inputs: blocks as vectors, current codebook, number of levels
    Outputs: partition regions
    """
    
    # Break vector array into batches to avoid memory overload for vectorization
    batch_size = 20**5
    best_L = []

    for i in range(0, len(blocks), batch_size):
        batch = blocks[i:i+batch_size]
        blocks_array = np.array(batch, dtype=np.float32)
        codebook_array = np.array(codebook, dtype=np.float32)

        # Calculate the Eucliden distance between vectors and codewords
        diff = blocks_array[:, np.newaxis, :] - codebook_array[np.newaxis, :, :]
        dists = np.sum(diff**2, axis=2)

        # Get the level for the closest distances
        best_L.append(np.argmin(dists, axis=1))

    best_L = np.concatenate(best_L)

    # Save each vector to its corresponding partition region based on the argmin values
    B_l = [[] for _ in range(L)]
    for i, cluster_L in enumerate(best_L):
        B_l[cluster_L].append(blocks[i])

    return B_l

def error_calc(training_vector, codeword):
    # Calculate the error between the training vector and the codeword
    difference = np.subtract(training_vector, codeword)
    error = np.sum(difference**2)
    return error

def euclidean_distance(vector1, vector2):
    # A function to calculate the Euclidean distance of two vectors
    return np.sqrt(np.sum((vector1.T - vector2)**2, axis=1))

def update_codewords(B_l, g_l):
    """
    A function to update the codewords
    Inputs: Partition regions and initial codebook
    Outputs: New codebook
    """

    codewords = []
    for level, B in enumerate(B_l):
        if B:       # check that cluster has arrays
            B_array = np.array(B)
            B_array = np.reshape(B_array, (B_array.shape[0], B_array.shape[1]))
            g_t = (1/len(B_array)) * np.sum(B_array, axis=0)    # calculate an updated codeword
            codewords.append(g_t)
        else:
            # If the partition region has no vectors, assign a new random codeword from the initial codewords
            g_t = random.choice(g_l)
            codewords.append(g_t)
    return codewords

def test_lbg_quantization(blocks, codebook):
    """
    A function to test the quantization on the target image.
    Inputs: blocks as vectors, trained codebook
    Outputs: Quantized blocks
    """

    quantized_blocks = []
    for block in blocks:    # assign each block a quantized value
        vector = np.reshape(block, (-1, 1))

        best_L = None
        best_dist = float('inf')

        # Find the closest codeword
        for l in range(0, len(codebook)):
            dist = euclidean_distance(vector, codebook[l])
            if dist < best_dist:
                best_dist = dist
                best_L = l
        
        quantized_blocks.append(np.reshape(codebook[best_L], block.shape))
    
    return quantized_blocks

def reconstruct_image(quantized_blocks, original_img, block_size):
    """
    A function to reconstruct the image
    Inputs: Quantized blocks, original image (for image shape), block size
    Outputs: Reconstructed image

    This code is based on the reconstruction code from Project 01.
    """
    img_height, img_width = original_img.shape
    num_blocks_x = int(img_height // block_size)
    num_blocks_y = int(img_width // block_size)

    reconstructed_img = np.empty((original_img.shape))

    # Iterate through the number of blocks in the x, y directions
    block_num = 0 
    for row_idx in range(num_blocks_x):
        for col_idx in range(num_blocks_y):
            start_x, start_y = row_idx * block_size, col_idx * block_size         # calculate the starting point of the block
            block = quantized_blocks[block_num] * 255
            block = np.reshape(block, (block_size, block_size))
            reconstructed_img[start_x:start_x+block_size, start_y:start_y+block_size] = block

            block_num += 1
    
    return reconstructed_img.astype(np.uint8)

def plot_vq_results(original_frame, predicted_frame, num_training_imgs, PSNR=None, levels=None, filepath=None, filename=None):
    """
    Saves an image made up of four subplots: anchor frame, target frame, predicted target frame, and motion field.
    Order of anchor and target frame depends on whether forward or backward estimation is occuring.

    """
    os.makedirs(filepath, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(7, 3), dpi=200)
    fig.suptitle(f'{levels} Levels Quantization Image Prediction from {num_training_imgs} Training Image(s)')

    axs[0].imshow(original_frame, cmap='gray')
    axs[0].set_title(f'Original Image')
    axs[0].axis('off')

    axs[1].imshow(predicted_frame, cmap='gray')
    axs[1].set_title(f'Predicted Image (PSNR: {PSNR:.4f})')
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{filepath}/{filename}.png')
    # plt.show()

def load_codebook(filepath):
    """
    A function to load in a pre-trained codebook.
    """
    # Check if codebook exists; raise error if not found.
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} not found.")
    
    f = open(filepath, 'r')
    data = f.read().split("]\n")
    codebook = []
    
    for line in data:
        line = line.replace("\n", "")
        line = line.replace("[", "")
        line_arr = np.fromstring(line, dtype=np.float32, sep=" ")
        if line_arr.size != 0:
            codebook.append(line_arr)

    return codebook