# -*- coding: utf-8 -*-
"""
Advanced Ascii art generator for any font.

19.01.2021
"""

import os
import sys
import json
import logging
import time
import logzero
from logzero import logger
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageChops
from math import floor
from scipy import fftpack
from enum import Enum, auto




class Metric(Enum):
    DCT = auto()
    LUM = auto()


def set_log_level(level):
    """
    Adapting the log level for information display through process.
    :param level: str - [debug/info/warning/error]
    """
    level_table = {
        'debug': logging.DEBUG,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'error': logging.ERROR
    }
    log_level = level_table[level.lower()]
    logzero.loglevel(log_level)


def dct2(image):
    """
    Performs a discrete cosine transform on the given image data,
    then truncates results to the first/last n factors,
    as configured by DCT_CUTOFF_LOW/DCT_CUTOFF_HIGH.
    :param image: pil image
    :return: reduced numpy 2D array
    """
    pixels = np.array(image, dtype=np.float)
    dct_data = fftpack.dct(fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
    # Cut down high frequency information
    dct_data[cfg_dct_cut_low:-cfg_dct_cut_high, :] = 0
    dct_data[:, cfg_dct_cut_low:-cfg_dct_cut_high] = 0
    dct_cutoff = cfg_dct_cut_low + cfg_dct_cut_high
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
    index_map = [[0] * chunks_x for _ in range(chunks_y)]
    fingerprints = []
    luminosity = []

    if cfg_metric == Metric.DCT:
        print("\tDiscrete Cosine Transformation...")
        for chunk in chunks:
            fingerprints.append(dct2(chunk))

    elif cfg_metric == Metric.LUM:
        print("\tLuminosity metric...")
        for chunk in chunks:
            luminosity.append(luminosity_avg(chunk))
        # Map range to 0-255
        if cfg_normalize_lum:
            luminosity /= max(luminosity)
            luminosity -= min(luminosity)
            luminosity *= 255 / max(luminosity)

    def compare_dct():
        # Calculate average difference between compressed transformations.
        diff_ratio = np.mean(np.abs(cutout_dct - fingerprint))
        chunk_match.append(diff_ratio)

    def compare_lum():
        # Calculate average difference in luminosity
        difference = abs(luminance - cutout_lum)
        chunk_match.append(difference)

    print("\tMatching...")
    last_percent = -10
    for cy in range(0, chunks_y):

        percent = round((cy / chunks_y) * 100)
        if percent - last_percent >= 10:
            last_percent = percent
            print(f"\t\t{percent}%")

        for cx in range(0, chunks_x):
            temp_cutout = image.crop(
                (chunk_width * cx, chunk_height * cy, chunk_width * (cx + 1), chunk_height * (cy + 1)))
            chunk_match = []

            if cfg_white_thresh < 255:
                pixels = np.array(temp_cutout)
                if pixels.mean() > cfg_white_thresh:
                    index_map[cy][cx] = -1
                    continue

            if cfg_metric == Metric.DCT:
                cutout_dct = dct2(temp_cutout)
                for fingerprint in fingerprints:
                    compare_dct()
            elif cfg_metric == Metric.LUM:
                cutout_lum = luminosity_avg(temp_cutout)
                for luminance in luminosity:
                    compare_lum()

            best_match = chunk_match.index(min(chunk_match))
            index_map[cy][cx] = best_match

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
            if chunk_map[cy][cx] >= 0:
                chunk = chunks[chunk_map[cy][cx]]
                canvas.paste(chunk, (chunk_width * cx, chunk_height * cy))

    return canvas


if __name__ == "__main__":
    # Set dir to project directory, in order to guarantee relative paths
    dir_name = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    os.chdir(dir_name)

    # Load config, restore if necessary
    config = {}
    try:
        with open("config.json") as config_file:
            config = json.load(config_file)
            print("Loading config.json")
            raise OSError

    except (OSError, ValueError):
        logger.warning("WARNING: config.json missing or corrupted. Restoring defaults.")
        config = {
            "general": {
                "image_path": "CleanCover.jpg",
                "prompt_confirmation": "first",  # first, each, none
                "pad_to_original_size": True,
                "logging": "info",
            },
            "font_settings": {
                "font_path": "Tensura.ttf",  # check \fonts\ then windows\fonts\ then absolute path
                "ASCII_whitelist": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;?@[\\]^_`{|}~",
                "ASCII_blacklist": "#$%*+@[\\]^_`{|}~",
                "font_size": 20,
                "auto_size": True,
                "character_width": 10,
                "character_height": 10,
                "size_as_padding": False,
                "force_square": False,
            },
            "method": {
                "metric": "dct",
                "white_threshold": 254,
                "normalize_luminosity": True,
                "DCT_cutoff_low": 3,
                "DCT_cutoff_high": 2,
            },
        }
        try:
            with open("config.json", 'w', encoding="utf-8") as outfile:
                json.dump(config, outfile, indent=4, ensure_ascii=False)
        except OSError:
            logger.error("Could not save config! Using defaults.")
    # End try

    # Load config
    cfg_img_path = config["general"]["image_path"]
    cfg_prompt_conf = config["general"]["prompt_confirmation"]
    cfg_pad_img = config["general"]["pad_to_original_size"]
    cfg_log_level = config["general"]["logging"]

    cfg_font_path = config["font_settings"]["font_path"]
    cfg_ascii_w = config["font_settings"]["ASCII_whitelist"]
    cfg_ascii_b = config["font_settings"]["ASCII_blacklist"]
    cfg_font_size = config["font_settings"]["font_size"]
    cfg_auto_size = config["font_settings"]["auto_size"]
    cfg_char_width = config["font_settings"]["character_width"]
    cfg_char_height = config["font_settings"]["character_height"]
    cfg_is_padding = config["font_settings"]["size_as_padding"]
    cfg_is_square = config["font_settings"]["force_square"]

    cfg_raw_metric = config["method"]["metric"]
    cfg_white_thresh = config["method"]["white_threshold"]
    cfg_normalize_lum = config["method"]["normalize_luminosity"]
    cfg_dct_cut_low = config["method"]["DCT_cutoff_low"]
    cfg_dct_cut_high = config["method"]["DCT_cutoff_high"]

    set_log_level(cfg_log_level)

    if cfg_raw_metric.upper() == "DCT":
        cfg_metric = Metric.DCT
    elif cfg_raw_metric.upper() == "LUM":
        cfg_metric = Metric.LUM
    else:
        logger.error("Unrecognized metric selected!")
        sys.exit()

    logger.info("Starting...")

# TODO pad to original size

# ASCII_CHARS = "⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾⠿⡀⡁⡂⡃⡄⡅⡆⡇⡈⡉⡊⡋⡌⡍⡎⡏⡐⡑⡒⡓⡔⡕⡖⡗⡘⡙⡚⡛⡜⡝⡞⡟⡠⡡⡢⡣⡤⡥⡦⡧⡨⡩⡪⡫⡬⡭⡮⡯⡰⡱⡲⡳⡴⡵⡶⡷⡸⡹⡺⡻⡼⡽⡾⡿⢀⢁⢂⢃⢄⢅⢆⢇⢈⢉⢊⢋⢌⢍⢎⢏⢐⢑⢒⢓⢔⢕⢖⢗⢘⢙⢚⢛⢜⢝⢞⢟⢠⢡⢢⢣⢤⢥⢦⢧⢨⢩⢪⢫⢬⢭⢮⢯⢰⢱⢲⢳⢴⢵⢶⢷⢸⢹⢺⢻⢼⢽⢾⢿⣀⣁⣂⣃⣄⣅⣆⣇⣈⣉⣊⣋⣌⣍⣎⣏⣐⣑⣒⣓⣔⣕⣖⣗⣘⣙⣚⣛⣜⣝⣞⣟⣠⣡⣢⣣⣤⣥⣦⣧⣨⣩⣪⣫⣬⣭⣮⣯⣰⣱⣲⣳⣴⣵⣶⣷⣸⣹⣺⣻⣼⣽⣾⣿"#"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"
ASCII_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"
# ASCII_CHARS = "ナフマキビウデントニムイヘゼサベィゾチェヒズヤテシザヂァドグモリォノツボプケネダペロゥジハブオワスバガタョュャアルユパソエギセゴヅホゲメピポヌミラカコヨレク"
ASCII_CHARS_DISABLE = "#$%*+@[\\]^_`{|}~"
# ASCII_CHARS = "|—_#+O',.-/\\=[]"
# ASCII_CHARS_DISABLE = ""
for char in ASCII_CHARS_DISABLE:
    ASCII_CHARS = ASCII_CHARS.replace(char, "")

tic = time.perf_counter()

file_name = "cover.jpeg"  # "CleanCover.jpg"
font = ImageFont.truetype("fonts\\Tensura.ttf", cfg_font_size)  # c:\windows\\fonts\\ARIALUNI.TTFBebasNeue-Regular.ttf"
print("Generating character thumbnails...")
thumbs = font_thumbnails(font, ASCII_CHARS, cfg_font_size * 2, width=cfg_char_width, height=cfg_char_height,
                         square=False)
print("Loading image...")
original = Image.open("image_in\\" + file_name)  # Dialemma.jpg")#
grayscale = to_greyscale(original)
print("Matching chunks...")
chunk_map = asciify_image_map(grayscale, thumbs)
print("Assembling...")
image = assemble_from_chunks(thumbs, chunk_map)

toc = time.perf_counter()
print(f"Done in {toc - tic:0.1f} seconds")

image.show()

if input("Save image? [y/n]").upper() == "Y":
    image.save(os.getcwd() + "\\image_out\\" + file_name, "png")

print("done")
