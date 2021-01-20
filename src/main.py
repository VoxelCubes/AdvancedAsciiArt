# -*- coding: utf-8 -*-
"""
Advanced Ascii art generator for any font.

19.01.2021
"""

from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageStat
from math import floor
from scipy import fftpack
import numpy as np
import os

ASCII_CHARS = "⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾⠿⡀⡁⡂⡃⡄⡅⡆⡇⡈⡉⡊⡋⡌⡍⡎⡏⡐⡑⡒⡓⡔⡕⡖⡗⡘⡙⡚⡛⡜⡝⡞⡟⡠⡡⡢⡣⡤⡥⡦⡧⡨⡩⡪⡫⡬⡭⡮⡯⡰⡱⡲⡳⡴⡵⡶⡷⡸⡹⡺⡻⡼⡽⡾⡿⢀⢁⢂⢃⢄⢅⢆⢇⢈⢉⢊⢋⢌⢍⢎⢏⢐⢑⢒⢓⢔⢕⢖⢗⢘⢙⢚⢛⢜⢝⢞⢟⢠⢡⢢⢣⢤⢥⢦⢧⢨⢩⢪⢫⢬⢭⢮⢯⢰⢱⢲⢳⢴⢵⢶⢷⢸⢹⢺⢻⢼⢽⢾⢿⣀⣁⣂⣃⣄⣅⣆⣇⣈⣉⣊⣋⣌⣍⣎⣏⣐⣑⣒⣓⣔⣕⣖⣗⣘⣙⣚⣛⣜⣝⣞⣟⣠⣡⣢⣣⣤⣥⣦⣧⣨⣩⣪⣫⣬⣭⣮⣯⣰⣱⣲⣳⣴⣵⣶⣷⣸⣹⣺⣻⣼⣽⣾⣿"#"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"
ASCII_CHARS_DISABLE = ""##$%*+@[\\]^_`{|}~"
for char in ASCII_CHARS_DISABLE:
    ASCII_CHARS = ASCII_CHARS.replace(char, "")


# implement 2D DCT
def dct2(image):
    pixels = np.array(image, dtype=np.float)
    dct_size = pixels.shape[0]
    dct_data = fftpack.dct(fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
    # Cut down high frequency information
    dct_data[DCT_CUTOFF:, :] = 0
    dct_data[:, DCT_CUTOFF:] = 0
    return dct_data[:DCT_CUTOFF, :DCT_CUTOFF]


# implement 2D IDCT
def idct2(coefficients):
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')


def get_reconstructed_image(raw):
    img = raw.clip(0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    return img


def to_greyscale(image):
    return image.convert("L")


def crop_background(image):
    """
    Crops away everything that is of solid background color (white). Crops to bounding box.
    :param image: pil image
    :return: pil image
    """
    bg = Image.new(image.mode, image.size, "white")
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    return image.crop(bbox)


def char_cutout(font, character, max_size):
    """
    Draws a character and crops to bounding box
    :param font: pil font
    :param character: string
    :param max_size: int
    :return: pil image
    """
    canvas = Image.new('L', (max_size, max_size), "white")
    img_draw = ImageDraw.Draw(canvas)
    img_draw.text((0, 0), character, fill="black", font=font)
    return crop_background(canvas)


def font_thumbnails(font, characters, max_size, width=None, height=None, is_padding=False, square=False):
    """
    Create a list of character cutouts, optionally padding or cropping if dimensions are given.
    :param font: pil font, contains font size
    :param characters: string of character for each to create cutout
    :param max_size: largest size to allow when drawing characters
    :param width: target width. None to fit to largest character
    :param height: target height. None to fit to largest character
    :param is_padding: when True, width and height are added to the max_char dimensions
    :param square: make image square
    :return: (list of pil images, final width, final height)
    """
    thumbs = []
    final_thumbs = []
    max_char_width = 0
    max_char_height = 0

    for char in characters:
        char_image = char_cutout(font, char, max_size)
        c_width, c_height = char_image.size
        max_char_width = max(max_char_width, c_width)
        max_char_height = max(max_char_height, c_height)
        thumbs.append(char_image)

    final_width = max_char_width
    final_height = max_char_height

    if is_padding:
        if width:
            final_width += width
        if height:
            final_height += height
    else:
        if width:
            final_width = width
        if height:
            final_height = height

    if square:
        final_width = max(final_width, final_height)
        final_height = max(final_width, final_height)

    # Crop or pad images
    for image in thumbs:
        char_w, char_h = image.size
        char_image = Image.new('L', (final_width, final_height), "white")
        char_image.paste(image, (floor((final_width - char_w) / 2), floor((final_height - char_h) / 2)))
        final_thumbs.append(char_image)
        char_image.show()

    print("Chunk size: ", final_width, final_height)
    return final_thumbs


def asciify_image_map(image, chunks):
    """
    Compares tiles from source image to find best replacement with ascii tiles.
    :param image: pil image
    :param chunks: list of pil images
    :return: 2d list of indices referring to chunks
    """
    image_w, image_h = image.size
    chunk_width, chunk_height = chunks[0].size
    chunks_x = floor(image_w / chunk_width)
    chunks_y = floor(image_h / chunk_height)

    temp_cutout = None
    best_match = None
    index_map = [[0] * chunks_x for _ in range(chunks_y)]

    if DCT:
        print("\tDiscrete Cosine Transformation...")
        fingerprints = []
        for chunk in chunks:
            cosine_transformation = dct2(chunk)
            fingerprints.append(cosine_transformation)

        # chunks[0].show()
        # dct_copy = fingerprints[1].copy()
        #
        # r_img = idct2(dct_copy)
        # reconstructed_image = get_reconstructed_image(r_img)
        # reconstructed_image.show()
        # print(dct_copy)
        # dct_copy[DCT_CUTOFF:, :] = 0
        # dct_copy[:, DCT_CUTOFF:] = 0
        # dct_copy = dct_copy[:DCT_CUTOFF, :DCT_CUTOFF]
        # print(dct_copy)
        # # Reconstructed image
        # r_img = idct2(dct_copy)
        # reconstructed_image = get_reconstructed_image(r_img)
        # reconstructed_image.show()

        #dct_cropped = np.array([[5272.8399516, -65.5641573, 1161.06710582, 129.37633343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-191.68429971, 92.71092256, 103.49563386, -130.90131562, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 119.54818952, 48.21936934, 56.05308502, -95.25159083, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ -37.58517323, -186.23500539, 101.23847235, 276.32962351, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ])
        # dct_cropped = np.array([[5272.8399516, -65.5641573, 1161.06710582, 129.37633343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-191.68429971, 92.71092256, 103.49563386, -130.90131562, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 119.54818952, 48.21936934, 56.05308502, -95.25159083, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ -37.58517323, -186.23500539, 101.23847235, 276.32962351, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ])
        # r_img = idct2(dct_cropped)
        # reconstructed_image = get_reconstructed_image(r_img)
        # reconstructed_image.show()

    def compare_diff():
        # Generate diff image in memory.
        diff_img = ImageChops.difference(temp_cutout, chunk)
        # Calculate difference as a ratio.
        stat = ImageStat.Stat(diff_img)
        diff_ratio = sum(stat.mean) / (len(stat.mean) * 255)
        chunk_match.append(diff_ratio)

    def compare_dct():
        cutout_dct = dct2(temp_cutout)
        diff_ratio = np.mean(np.abs((cutout_dct - fingerprint)**2))
        chunk_match.append(diff_ratio)
        return 0

    print("\tMatching...")
    last_percent = -10
    for cy in range(0, chunks_y):

        percent = round((cy / chunks_y) * 100)
        if percent - last_percent >= 10:
            last_percent = percent
            print(f"\t\t{percent}%")

        for cx in range(0, chunks_x):
            temp_cutout = image.crop((chunk_width * cx, chunk_height * cy, chunk_width * (cx + 1), chunk_height * (cy + 1)))
            chunk_match = []

            if ALLOW_WHITE:
                pixels = np.array(temp_cutout)
                if pixels.mean() > WHITE_THRESHOLD:
                    index_map[cy][cx] = -1
                    continue

            if DCT:
                for fingerprint in fingerprints:
                    compare_dct()

            else:
                for chunk in chunks:
                    compare_diff()

            best_match = chunk_match.index(min(chunk_match))
            index_map[cy][cx] = best_match

        # temp_cutout.show()
        # chunks[best_match].show()
    print(f"\t\t100%")

    return index_map


def assemble_from_chunks(chunks, chunk_map):
    """
    Assembles ascii tiles according to the 2d map holding the corresponding indices for chunks
    :param chunks: list of pil images
    :param chunk_map: 2d list of indices referring to chunks
    :return: the ascii art pil image
    """
    rows = len(chunk_map)
    cols = len(chunk_map[0])
    chunk_width, chunk_height = chunks[0].size
    canvas = Image.new("L", (cols * chunk_width, rows * chunk_height), "white")
    print("Final image dimensions", cols * chunk_width, rows * chunk_height)

    for cy in range(0, rows):
        for cx in range(0, cols):
            if chunk_map[cy][cx] > 0:
                chunk = chunks[chunk_map[cy][cx]]
                canvas.paste(chunk, (chunk_width * cx, chunk_height * cy))

    canvas.show()
    return canvas

DCT = True
DCT_CUTOFF = 3  # higher retains more detail
ALLOW_WHITE = True
WHITE_THRESHOLD = 240
width = None
height = None
file_name = "test.png"#"vol12\\LabyrinthENtyped.png"
font = ImageFont.truetype("c:\windows\\fonts\\ARIALUNI.TTF", 15)#Tensura\\..\\fonts\\BebasNeue-Regular.ttf"
print("Generating character thumbnails...")
thumbs = font_thumbnails(font, ASCII_CHARS, 50, width=width, height=height, square=False)
print("Loading image...")
original = Image.open(".\\image_in\\"+file_name)  # Dialemma.jpg")#
grayscale = to_greyscale(original)
print("Matching chunks...")
chunk_map = asciify_image_map(grayscale, thumbs)
print("Assembling...")
image = assemble_from_chunks(thumbs, chunk_map)

if input("Save image? [y/n]").upper() == "Y":
    image.save(os.getcwd()+"\\image_out\\"+file_name, "png")

print("done")
