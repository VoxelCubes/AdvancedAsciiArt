# -*- coding: utf-8 -*-
"""
Advanced Ascii art generator for any font.
Helper file for static functions that don't rely on global variables.

19.01.2021
"""


from enum import Enum, auto
from math import floor

import numpy as np
from PIL import Image, ImageDraw, ImageChops
from scipy import fftpack


class Metric(Enum):
    DCT = auto()
    LUM = auto()
    MIX = auto()


class Confirmation(Enum):
    FIRST = auto()
    EACH = auto()
    NONE = auto()


def default_config():
    return {
        "general": {
            "image_path": "cc0_house_tweaked.jpg",  # check in image_in, then absolute path
            "prompt_confirmation": "first",  # first, each, none
            "pad_to_original_size": True,
            "pad_centered": True,
            "logging": "info",
            "progress_interval": 25,
            "allowed_file_types": [".bmp", ".png", ".jpg", ".jpeg", ".tiff"],
            "ignore_invalid_types": True,
        },
        "font_settings": {
            "font_path": "courbd.ttf",  # check \fonts\ then windows\fonts\ then absolute path
            "ASCII_whitelist": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;?@[\\]^_`{|}~",
            "ASCII_blacklist": "",
            "font_size": 20,
            "auto_size": True,
            "character_width": 2,
            "character_height": 2,
            "size_as_padding": False,
            "force_square": False,
        },
        "method": {
            "metric": "mix",
            "white_threshold": 250,
            "normalize_luminosity": True,
            "DCT_cutoff_low": 3,
            "DCT_cutoff_high": 2,
            "Mix_threshold": 0.50,
        },
    }


def test_file_extension(file_name, valid_extensions):
    """
    Test if file has extension in config
    :param file_name: pathlib2.Path
    :param valid_extensions: [string,...], list of valid extension strings
    :return True if valid extension
    """
    return file_name.suffix in valid_extensions


def dct2(image, cutoff_low, cutoff_high):
    """
    Performs a discrete cosine transform on the given image data,
    then truncates results to the first/last n factors.
    :param image: pil image
    :param cutoff_low: int, how many low frequencies to preserve
    :param cutoff_high: int, how many high frequencies to preserve
    :return: reduced numpy 2D array
    """
    pixels = np.array(image, dtype=np.float)
    dct_data = fftpack.dct(fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
    # Cut down high frequency information
    dct_data[cutoff_low:-cutoff_high, :] = 0
    dct_data[:, cutoff_low:-cutoff_high] = 0
    dct_cutoff = cutoff_low + cutoff_high
    dct_minimized = dct_data[:dct_cutoff, :dct_cutoff] + dct_data[-dct_cutoff:, -dct_cutoff:]
    return dct_minimized


def to_greyscale(image):
    """
    Converts RGB into a single luminance channel.
    :param image: pil image
    :return: pil image
    """
    return image.convert("L")


def luminosity_avg(image):
    """
    Calculates the arithmetic mean of the image luminosity
    :param image: pil image in mode "L"
    :return: numpy float
    """
    pixels = np.array(image, dtype=np.float)
    return pixels.mean(dtype=np.float)


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


def pad_image_to_size(image, size, centered):
    """
    Pads image on the right and bottom with white.
    :param image: pil image
    :param size: (width, height)
    :param centered: bool
    :return: pil image
    """
    bg = Image.new("L", size, "white")
    if centered:
        bg.paste(image, ((size[0] - image.size[0]) // 2, (size[1] - image.size[1]) // 2))
    else:
        bg.paste(image, (0, 0))
    return bg


def char_cutout(font, character, max_size):
    """
    Draws a character and crops to bounding box
    :param font: pil font
    :param character: string
    :param max_size: (int, int), the size of the canvas to draw on before cropping
    :return: pil image
    """
    canvas = Image.new("L", (max_size, max_size), "white")
    img_draw = ImageDraw.Draw(canvas)
    img_draw.text((0, 0), character, fill="black", font=font)
    return crop_background(canvas)


def font_thumbnails(font, characters, max_size, width=None, height=None,
                    is_padding=False, square=False, logger=None):
    """
    Create a list of character cutouts, optionally padding or cropping if dimensions are given.
    :param font: pil font, contains font size
    :param characters: string of character for each to create cutout
    :param max_size: (int, int), largest size to allow when drawing characters
    :param width: int, target width. None to fit to largest character
    :param height: int, target height. None to fit to largest character
    :param is_padding: when True, width and height are added to the max_char dimensions
    :param square: make image square
    :param logger: logger to write to
    :return: list of pil images
    """
    thumbs = []
    final_thumbs = []
    max_char_width = 0
    max_char_height = 0

    # Create list of images where each character is cropped to its own minimum size
    for char in characters:
        char_image = char_cutout(font, char, max_size)
        c_width, c_height = char_image.size
        # Find the largest character dimensions to determine auto size
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
        char_image = Image.new("L", (final_width, final_height), "white")
        char_image.paste(image, (floor((final_width - char_w) / 2), floor((final_height - char_h) / 2)))
        final_thumbs.append(char_image)

    logger.debug(f"Chunk size: {final_width} {final_height}")
    return final_thumbs


def assemble_from_chunks(chunks, chunk_map, logger):
    """
    Assembles ascii tiles according to the 2d map holding the corresponding indices for chunks
    :param chunks: list of pil images
    :param chunk_map: list[list], 2d list array of indices referring to chunks
    :param logger: logger to write to
    :return: the ascii art pil image
    """
    rows = len(chunk_map)
    cols = len(chunk_map[0])
    chunk_width, chunk_height = chunks[0].size
    canvas = Image.new("L", (cols * chunk_width, rows * chunk_height), "white")
    logger.debug(f"Final image dimensions without padding {cols * chunk_width} {rows * chunk_height}")
    for cy in range(0, rows):
        for cx in range(0, cols):
            if chunk_map[cy][cx] >= 0:
                chunk = chunks[chunk_map[cy][cx]]
                canvas.paste(chunk, (chunk_width * cx, chunk_height * cy))

    return canvas
