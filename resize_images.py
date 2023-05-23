import os
import cv2
import sys
import numpy as np
import tqdm
import time
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
import multiprocessing
from multiprocessing import Pool
from functools import partial


# Define function to resize images
def resize_image(i, old_folder_path, new_folder_path, img_size, img_ending, bit16_image, resize_images):
    file_paths = [os.path.join(i, f.decode('utf-8')) for f in os.listdir(i) if f.decode('utf-8').endswith(img_ending)]
    image_paths_new = [f.replace(old_folder_path, new_folder_path) for f in file_paths]
    img_dir_new = i.replace(old_folder_path, new_folder_path)
    if not os.path.exists(img_dir_new):
        os.makedirs(img_dir_new)
    for file, image_path_new in zip(file_paths, image_paths_new):
        if not os.path.exists(image_path_new):
            if bit16_image:
                image = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.float32)
                image = image/65535*255
                if resize_images:
                    image = resize(image, output_shape=(img_size, img_size), preserve_range=True)
                imsave(image_path_new, image.astype(np.uint8))
            else:
                image = imread(file)
                if resize_images:
                    image = resize(image, output_shape=(img_size, img_size), preserve_range=True)
                imsave(image_path_new, image.astype(np.uint8))


def process_images_in_parallel(img_list, old_folder_path, new_folder_path, img_size, img_ending, num_cpus, bit16_image, resize_images):
    # Prepare multiprocessing
    num_cpus_ava = multiprocessing.cpu_count()  # Number of available CPUs
    pool = multiprocessing.Pool(processes=num_cpus)  # Initialize a pool of worker processes
    print(f" Processors available: {num_cpus_ava}; using {num_cpus}...")

    # The partial function will make it easier to pass the same arguments to myfunc_wrapper for all images
    wrapped_func = partial(resize_image, old_folder_path=old_folder_path, new_folder_path=new_folder_path, img_size=img_size, img_ending=img_ending, bit16_image=bit16_image, resize_images=resize_images)

    # Use the pool's imap_unordered method to run wrapped_func in parallel for all images in img_list
    # Wrap it with tqdm for progress bar
    for _ in tqdm.tqdm(pool.imap_unordered(wrapped_func, img_list), total=len(img_list)):
        pass

    pool.close()  # Close the pool
    pool.join()  # Wait for all worker processes to finish


if __name__ == "__main__":
    num_cpus = 10
    img_ending = "png"
    bit16_image = True
    img_size = 224
    resize_images = False
    old_folder_path = "/Users/felixkrones/python_projects/data/Padchest/0/"
    new_folder_path = "/Users/felixkrones/python_projects/data/Padchest/0_255/"

    # Get image paths
    img_list = []
    for root, dirs, files in os.walk(old_folder_path):
        for file in files:
            if(file.endswith(img_ending)):
                img_list.append(root)
    img_list = np.unique(img_list)

    print(f"Starting resizing for {len(img_list)} folders ...")
    start_time = time.time()
    process_images_in_parallel(img_list, old_folder_path, new_folder_path, img_size, img_ending, num_cpus, bit16_image, resize_images)
    print(f"Finished resizing {len(img_list)} folders after {round((time.time() - start_time)/60,4)} min")
