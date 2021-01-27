import numpy as np
from PIL import Image
import os
import datetime

# TODO: int or float?

# matches np.float64 from PMD notebook


def image_series_to_npy(image_dir_path, image_names, num_frames, resolution):
    num_images = len(image_names)

    dataset = np.zeros((resolution, resolution, num_images * num_frames))  # initializes a zero array

    for i in range(len(image_names)):
        image = Image.open(os.path.join(image_dir_path, image_names[i]))
        image_array = np.array(image, dtype=np.float64)

        for j in range(num_frames):
            dataset[resolution - 1, resolution - 1, i + j] = image_array[
                resolution - 1, resolution - 1, j]  # adds each frame one at a time

        if i % 100 == 0:  # updates progress
            print(f'{(i / num_images) * 100}% complete...')

    return dataset


def main():
    start = datetime.datetime.now()

    image_dir_path = 'C:/Users/Sand Box/Hugo_Mecp2_CaPop/170920timeseries'
    output_npy_path = 'C:/Users/Sand Box/funtref/npy_output/170920timeseries.npy'

    image_extension = '.tif'

    image_names = [image_name for image_name in os.listdir(image_dir_path) if image_name[-4:] == image_extension]

    num_frames = 3
    resolution = 512  # assumes images are squares

    dataset = image_series_to_npy(image_dir_path, image_names, num_frames, resolution)
    # np.save(output_npy_path, dataset)

    end = datetime.datetime.now()
    print('Time elapsed', end - start)
    return dataset
