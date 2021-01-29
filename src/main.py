# -*- coding: utf-8 -*-
"""
Advanced Ascii art generator for any font.
Main file to run.
19.01.2021
"""

import os
import sys
import json
import logging
import time
import logzero
import numpy as np

from pathlib2 import Path
from PIL import Image, ImageDraw, ImageFont, ImageChops
from math import floor
from scipy import fftpack

# Enums
from helpers import Metric, Confirmation
# Functions
from helpers import default_config, test_file_extension, dct2, assemble_from_chunks, \
                    luminosity_avg, font_thumbnails, to_greyscale, pad_image_to_size


def add_new_image_from_file(path):
    """
    Try to add the image after testing if the extension is valid.
    :param path: absolute path
    """
    if test_file_extension(path):
        originals.append(Image.open(str(path)))
        filenames.append(path.name)
    elif cfg_ignore_inv_ext:
        logger.warning(f"Unsupported file extension! skipping {path.name}")
    else:
        logger.error(f"Unsupported file extension in {path.name}")
        sys.exit()


def set_log_level(level):
    """
    Adapting the log level for information display through process.
    :param level: str - [debug/info/warning/error]
    """
    global logger
    level_table = {
        'debug': logging.DEBUG,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'error': logging.ERROR
    }
    log_level = level_table[level.lower()]
    if log_level == logging.DEBUG:
        formatter = logzero.LogFormatter(
            fmt="%(color)s[%(levelname)1.1s %(module)s:%(lineno)d]%(end_color)s %(message)s")
    else:
        formatter = logzero.LogFormatter(
            fmt="%(color)s[%(levelname)1.1s]%(end_color)s %(message)s")
    logger = logzero.setup_logger(formatter=formatter, level=log_level)


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

    logger.debug(f"Handling {chunks_x * chunks_y} chunks.")

    temp_cutout = None
    index_map = [[0] * chunks_x for _ in range(chunks_y)]
    fingerprints = []
    luminosity = []

    if cfg_metric == Metric.DCT:
        logger.debug("\tDiscrete Cosine Transformation...")
        for chunk in chunks:
            fingerprints.append(dct2(chunk, cfg_dct_cut_low, cfg_dct_cut_high))

    elif cfg_metric == Metric.LUM:
        logger.debug("\tLuminosity metric...")
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

    if 0 < cfg_progress_interval < 100:
        last_percent = -cfg_progress_interval
    else:
        last_percent = 1000  # Disable progress reports

    for cy in range(0, chunks_y):

        percent = round((cy / chunks_y) * 100)
        if percent - last_percent >= cfg_progress_interval:
            last_percent = percent
            logger.info(f"\t\t{percent}%")

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
                cutout_dct = dct2(temp_cutout, cfg_dct_cut_low, cfg_dct_cut_high)
                for fingerprint in fingerprints:
                    compare_dct()
            elif cfg_metric == Metric.LUM:
                cutout_lum = luminosity_avg(temp_cutout)
                for luminance in luminosity:
                    compare_lum()

            best_match = chunk_match.index(min(chunk_match))
            index_map[cy][cx] = best_match

    if 0 < cfg_progress_interval < 100:
        logger.info(f"\t\t100%")

    return index_map


def make_ascii_art(original):
    """
    Using config (cfg_) values from main
    :return: pil image
    """
    tic = time.perf_counter()

    logger.debug("Generating character thumbnails...")
    thumbs = font_thumbnails(font, cfg_ascii_w, cfg_font_size * 2,
                             width=cfg_char_width, height=cfg_char_height,
                             square=cfg_is_square, is_padding=cfg_is_padding)
    logger.debug("Matching chunks...")
    chunk_map = asciify_image_map(to_greyscale(original), thumbs)
    logger.debug("Assembling...")
    image_assembled = assemble_from_chunks(thumbs, chunk_map)

    if cfg_pad_img:
        logger.debug(f"Padding image from {image_assembled.size} to {original.size}")
        image_assembled = pad_image_to_size(image_assembled, original.size, cfg_pad_centered)

    toc = time.perf_counter()
    logger.info(f"Done in {toc - tic:0.1f} seconds.")

    return image_assembled


if __name__ == "__main__":
    # Set dir to project directory, in order to guarantee relative paths
    cwd = Path(sys.argv[0]).parents[1]
    os.chdir(cwd)

    # Set up logger to use reduced date and time tag
    global logger
    set_log_level("warning")

    """
    --------------------Config--------------------
    """
    # Load config, restore if necessary
    config = {}
    try:
        with open("config.json") as config_file:
            config = json.load(config_file)
            print("Loading config.json")

    except (OSError, ValueError):
        logger.warning("WARNING: config.json missing or corrupted. Restoring defaults.")
        config = default_config()
        try:
            with open("config.json", 'w', encoding="utf-8") as outfile:
                json.dump(config, outfile, indent=4, ensure_ascii=False)
        except OSError:
            logger.error("Could not save config! Using defaults.")
    # End try

    # Load config
    cfg_img_path = Path(config["general"]["image_path"])
    cfg_raw_prompt_conf = config["general"]["prompt_confirmation"]
    cfg_pad_img = config["general"]["pad_to_original_size"]
    cfg_pad_centered = config["general"]["pad_centered"]
    cfg_log_level = config["general"]["logging"]
    cfg_progress_interval = config["general"]["progress_interval"]
    cfg_extensions = config["general"]["allowed_file_types"]
    cfg_ignore_inv_ext = config["general"]["ignore_invalid_types"]

    cfg_font_path = Path(config["font_settings"]["font_path"])
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

    # Handle settings
    set_log_level(cfg_log_level)

    if cfg_raw_metric.upper() == "DCT":
        cfg_metric = Metric.DCT
    elif cfg_raw_metric.upper() == "LUM":
        cfg_metric = Metric.LUM
    else:
        logger.error("Unrecognized metric selected!")
        sys.exit()

    if cfg_raw_prompt_conf.upper() == "FIRST":
        cfg_prompt_conf = Confirmation.FIRST
    elif cfg_raw_prompt_conf.upper() == "EACH":
        cfg_prompt_conf = Confirmation.EACH
    elif cfg_raw_prompt_conf.upper() == "NONE":
        cfg_prompt_conf = Confirmation.NONE
    else:
        logger.error("Unrecognized confirmation prompt selected!")
        sys.exit()

    # Purge blacklisted chars from whitelist
    for char in cfg_ascii_b:
        cfg_ascii_w = cfg_ascii_w.replace(char, "")

    if cfg_auto_size and not cfg_is_padding:
        cfg_char_width  = None
        cfg_char_height = None

    """
    --------------------Files--------------------
    """
    originals = []
    filenames = []
    # Test existence of files
    try:
        if not cfg_img_path.is_absolute():
            input_path = Path(cwd, "image_in", cfg_img_path)
        else:
            input_path = cfg_img_path

        # Files
        if input_path.is_file():
            add_new_image_from_file(input_path)
        # Directories
        elif input_path.is_dir():
            for img in input_path.glob("*"):
                add_new_image_from_file(img)
        else:
            logger.error("Could not resolve path.")
            raise OSError
    except OSError:
        logger.error("Could not load image/s!")
        sys.exit()

    try:
        input_path_local = Path(cwd, "fonts", cfg_font_path)
        input_path_windows = Path("C:\\windows\\fonts\\", cfg_font_path)

        if input_path_local.is_file():
            font = ImageFont.truetype(str(input_path_local), cfg_font_size)
        elif input_path_windows.is_file():
            font = ImageFont.truetype(str(input_path_windows), cfg_font_size)
        elif cfg_font_path.is_file():
            font = ImageFont.truetype(str(cfg_font_path), cfg_font_size)
        else:
            raise OSError
    except OSError:
        logger.error("Could not load font!")
        sys.exit()

    # File preview
    if len(filenames) > 1:
        print(f"-----\n{len(filenames)} Files:")
        for file_name in filenames:
            print(file_name)
        print("-----")
    elif len(filenames) == 0:
        logger.error("No valid files found.")
        sys.exit()

    """
    --------------------ASCII art--------------------
    """
    ascii_art = []

    for index, image in enumerate(originals):
        logger.info(f"Image {index+1} / {len(originals)} : {filenames[index]}")
        ascii_art.append(make_ascii_art(image))
        # Ask if configured to
        if cfg_prompt_conf == Confirmation.EACH:
            ascii_art[index].show()
            if input("Save image? [y/n]").upper() == "Y":
                try:
                    file_path = str(cwd) + "\\image_out\\" + filenames[index]
                    ascii_art[index].save(file_path, "png")
                    print(f"Saved image in {file_path}")
                except OSError:
                    logger.error(f"Could not save image in {file_path}!")
                    sys.exit()

            if index != len(originals)-1 and input("Continue with next image? [y/n]").upper() == "Y":
                continue
            else:
                break

        elif index == 0 and cfg_prompt_conf == Confirmation.FIRST:
            ascii_art[index].show()
            if input("Save image? [y/n]").upper() == "Y":
                try:
                    file_path = str(cwd) + "\\image_out\\" + filenames[index]
                    ascii_art[index].save(file_path, "png")
                    print(f"Saved image in {file_path}")
                except OSError:
                    logger.error(f"Could not save image in {file_path}!")
                    sys.exit()

            if index != len(originals) - 1 and input("Continue with next images? [y/n]").upper() == "Y":
                continue
            else:
                break

        else:
            try:
                file_path = str(cwd) + "\\image_out\\" + filenames[index]
                ascii_art[index].save(file_path, "png")
                print(f"Saved image in {file_path}")
            except OSError:
                logger.error(f"Could not save image in {file_path}!")
                sys.exit()

    logger.info("Done.")
