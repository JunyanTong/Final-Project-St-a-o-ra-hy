#!/usr/bin/env python

import click
from PIL import Image
import torch
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 

def __int_to_bin(rgb):
    """Convert an integer tuple to a binary (string) tuple.

        :param rgb: An integer tuple (e.g. (220, 110, 96))
    :return: A string tuple (e.g. ("00101010", "11101011", "00010110"))
        """
    r, g, b = rgb
    return (f'{r:08b}',
                f'{g:08b}',
                f'{b:08b}')


def __bin_to_int(rgb):

    r, g, b = rgb
    return (int(r, 2),
                int(g, 2),
                int(b, 2))

def __merge_rgb(rgb1, rgb2):

    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    rgb = (r1[:4] + r2[:4],
            g1[:4] + g2[:4],
            b1[:4] + b2[:4])
    return rgb


def merge(img1, img2):


    # Check the images dimensions
    if img2.size[0] > img1.size[0] or img2.size[1] > img1.size[1]:
        raise ValueError('Image 2 should not be larger than Image 1!')

    # Get the pixel map of the two images
    pixel_map1 = img1.load()
    pixel_map2 = img2.load()

    # Create a new image that will be outputted
    new_image = Image.new(img1.mode, img1.size)
    pixels_new = new_image.load()

    for i in range(img1.size[0]):
        for j in range(img1.size[1]):
            rgb1 = __int_to_bin(pixel_map1[i, j])

            # Use a black pixel as default
            rgb2 = __int_to_bin((0, 0, 0))

            # Check if the pixel map position is valid for the second image
            if i < img2.size[0] and j < img2.size[1]:
                rgb2 = __int_to_bin(pixel_map2[i, j])

            # Merge the two pixels and convert it to a integer tuple
            rgb = __merge_rgb(rgb1, rgb2)

            pixels_new[i, j] = __bin_to_int(rgb)

    return new_image


def unmerge(img):
       
    # Load the pixel map
    pixel_map = img.load()

    # Create the new image and load the pixel map
    new_image = Image.new(img.mode, img.size)
    pixels_new = new_image.load()

    # Tuple used to store the image original size
    original_size = img.size

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            # Get the RGB (as a string tuple) from the current pixel
            r, g, b =__int_to_bin(pixel_map[i, j])

            # Extract the last 4 bits (corresponding to the hidden image)
            # Concatenate 4 zero bits because we are working with 8 bit
            rgb = (r[4:] + '0000',
                    g[4:] + '0000',
                    b[4:] + '0000')

            # Convert it to an integer tuple
            pixels_new[i, j] = __bin_to_int(rgb)

            # If this is a 'valid' position, store it
            # as the last valid position
            if pixels_new[i, j] != (0, 0, 0):
                original_size = (i + 1, j + 1)

    # Crop the image based on the 'valid' pixels
    new_image = new_image.crop((0, 0, 768, 512))

    return new_image

def forward_merge(img1, img2):
    mer_image = merge(img1, img2)
    rec_image = unmerge(mer_image)
    rec_image = torch.Tensor(np.array(rec_image))
    img2 = torch.Tensor(np.array(img2))
    loss = torch.sqrt(torch.nn.functional.mse_loss(rec_image, img2))#np.sqrt(mean_squared_error(np.array(rec_image), np.array(img2)))
    return loss


if __name__ == '__main__':
    root = "data/test/Kodak24/"
    width, height = 768, 512
    for i in range(12):
        path1 = root + "kodim{:0>2d}.png".format(i+1)
        path2 = root + "kodim{:0>2d}.png".format(i+2)
        img1 = Image.open(path1)
        img1 = img1.resize((width, height),Image.ANTIALIAS)
        img2 = Image.open(path2)
        img2 = img2.resize((width, height),Image.ANTIALIAS)
        # print(img1)
        loss = forward_merge(img1, img2)
        print("The {}th pair pictures' rmse is :{}".format(i+1, loss))

