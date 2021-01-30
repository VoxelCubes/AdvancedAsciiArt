# -*- coding: utf-8 -*-
"""
Advanced Ascii art generator for any font.
Main file to run.
19.01.2021
"""

import json
import logging
import os
import sys
import time
from math import floor

import logzero
import numpy as np
from PIL import Image, ImageFont
from pathlib2 import Path

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
    if test_file_extension(path, cfg_extensions):
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
    :param level: str, [debug/info/warning/error]
    """
    # Ensuring the global logger is overwritten
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
    :return: list[list], 2d list of indices referring to chunks
    """
    image_w, image_h = image.size
    chunk_width, chunk_height = chunks[0].size
    chunks_x = floor(image_w / chunk_width)
    chunks_y = floor(image_h / chunk_height)

    logger.debug(f"Handling {chunks_x * chunks_y} chunks.")

    # Initialize list with independent nested lists
    index_map = [[0] * chunks_x for _ in range(chunks_y)]
    # Cache for pre-computed character cutouts
    fingerprints = []
    luminosity = []

    if cfg_metric == Metric.DCT or cfg_metric == Metric.MIX:
        # Pre-compute the transformation of each chunk.
        # Save it with its index because MIX metric may skip many chunks.
        logger.debug("\tDiscrete Cosine Transformation...")
        for i, chunk in enumerate(chunks):
            fingerprints.append((dct2(chunk, cfg_dct_cut_low, cfg_dct_cut_high), i))

    if cfg_metric == Metric.LUM or cfg_metric == Metric.MIX:
        # Pre-compute the luminance of each chunk.
        # Save it with its index because MIX metric may skip many chunks.
        logger.debug("\tLuminosity metric...")
        for i, chunk in enumerate(chunks):
            luminosity.append((luminosity_avg(chunk), i))
        # Map range to 0-255
        if cfg_normalize_lum:
            luminosity, _ = zip(*luminosity)  # Unzip included iterator
            luminosity /= max(luminosity)
            luminosity -= min(luminosity)
            luminosity *= 255 / max(luminosity)
            luminosity = list(zip(luminosity, range(len(luminosity))))

    # Helper functions only used locally
    def compare_dct(chunk_print):
        # Calculate average difference between compressed transformations, save with original index
        diff_ratio = np.mean(np.abs(cutout_dct - chunk_print[0]))
        chunk_match.append((diff_ratio, chunk_print[1]))

    def compare_lum(chunk_lum):
        # Calculate average difference in luminosity, save with original index
        difference = abs(cutout_lum - chunk_lum[0])
        chunk_match.append((difference, chunk_lum[1]))

    # Initialize progress counter as configured
    if 0 < cfg_progress_interval < 100:
        last_percent = -cfg_progress_interval
    else:
        last_percent = 1000  # Disable progress reports

    # Loop through all image cutouts and find best match from among the character chunks
    for cy in range(0, chunks_y):

        # Display the current progress at defined intervals
        percent = round((cy / chunks_y) * 100)
        if percent - last_percent >= cfg_progress_interval:
            last_percent = percent
            logger.info(f"\t\t{percent}%")

        for cx in range(0, chunks_x):
            temp_cutout = image.crop(
                (chunk_width * cx, chunk_height * cy, chunk_width * (cx + 1), chunk_height * (cy + 1)))
            # Initialize chunk match list
            chunk_match = []

            # Set tiles to white if bright enough.
            # Chunks with index -1 won't be drawn, leaving a blank white box
            if cfg_white_thresh < 256:
                pixels = np.array(temp_cutout)
                if pixels.mean() >= cfg_white_thresh:
                    index_map[cy][cx] = -1
                    continue
            # DCT metric
            if cfg_metric == Metric.DCT:
                cutout_dct = dct2(temp_cutout, cfg_dct_cut_low, cfg_dct_cut_high)
                for fingerprint in fingerprints:
                    compare_dct(fingerprint)
            # LUM metric
            elif cfg_metric == Metric.LUM:
                cutout_lum = luminosity_avg(temp_cutout)
                for luminance in luminosity:
                    compare_lum(luminance)
            # MIX metric
            elif cfg_metric == Metric.MIX:
                cutout_lum = luminosity_avg(temp_cutout)
                cutout_dct = dct2(temp_cutout, cfg_dct_cut_low, cfg_dct_cut_high)
                # Limit the choices of the DCT metric my first selecting rough matches with LUM metric
                for luminance in luminosity:
                    compare_lum(luminance)

                fingerprints_by_priority = [x for (y, x) in sorted(
                    zip(chunk_match, fingerprints),
                    key=lambda pair: pair[0])]
                # Limit to suggested range
                fingerprints_by_priority = fingerprints_by_priority[0:cfg_mix_threshold]
                chunk_match = []
                for fingerprint in fingerprints_by_priority:
                    compare_dct(fingerprint)

            # Save best matching character chunk, meaning least difference
            # Use the index bound to the match tuple
            # Because the index won't match with the current index in MIX metric
            best_match = min(chunk_match, key=lambda pair: pair[0])
            index_map[cy][cx] = best_match[1]

    if 0 < cfg_progress_interval < 100:
        logger.info(f"\t\t100%")

    return index_map


def make_ascii_art(original):
    """
    Takes raw input image and outputs the final ascii image.
    Using config (cfg_) values from main
    :return: pil image
    """
    tic = time.perf_counter()

    logger.debug("Generating character thumbnails...")
    thumbs = font_thumbnails(font, cfg_ascii_w, cfg_font_size * 2,
                             width=cfg_char_width, height=cfg_char_height,
                             square=cfg_is_square, is_padding=cfg_is_padding, logger=logger)
    logger.debug("Matching chunks...")
    chunk_map = asciify_image_map(to_greyscale(original), thumbs)
    logger.debug("Assembling...")
    image_assembled = assemble_from_chunks(thumbs, chunk_map, logger)

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

    # Load config as variables to prevent dict key typos
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
    cfg_mix_threshold = config["method"]["Mix_threshold"]

    # Handle settings
    set_log_level(cfg_log_level)

    if cfg_raw_metric.upper() == "DCT":
        cfg_metric = Metric.DCT
    elif cfg_raw_metric.upper() == "LUM":
        cfg_metric = Metric.LUM
    elif cfg_raw_metric.upper() == "MIX":
        cfg_metric = Metric.MIX
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
        cfg_char_width = None
        cfg_char_height = None

    # Convert mix threshold to absolute value if percentage,
    # or make sure it's less than the number of allowed characters
    if cfg_metric == Metric.MIX:
        if cfg_mix_threshold < 1:
            cfg_mix_threshold = round(cfg_mix_threshold * len(cfg_ascii_w))
        cfg_mix_threshold = np.clip(round(cfg_mix_threshold), 1, len(cfg_ascii_w) - 1)
        logger.debug(f"Set MIX threshold at the best {cfg_mix_threshold} picks.")

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
        logger.info(f"Image {index + 1} / {len(originals)} : {filenames[index]}")
        ascii_art.append(make_ascii_art(image))
        # May remain undefined, though unlikely.
        file_path = None
        # Check if consent is needed
        get_consent = cfg_prompt_conf == Confirmation.EACH or (index == 0 and cfg_prompt_conf == Confirmation.FIRST)

        if get_consent:
            ascii_art[index].show()

        if not get_consent or input("Save image? [y/n]").upper() == "Y":
            # Attempt to save image
            try:
                file_path = str(cwd) + "\\image_out\\" + filenames[index]
                ascii_art[index].save(file_path, "png")
                print(f"Saved image in {file_path}")
            except OSError:
                logger.error(f"Could not save image in {file_path}!")
                sys.exit()

        if get_consent:
            if index != len(originals) - 1 and input("Continue with next image? [y/n]").upper() == "Y":
                continue
            else:
                break

    logger.info("Done.")
