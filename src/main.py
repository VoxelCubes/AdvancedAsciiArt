"""
Advanced Ascii art generator for any font.

19.01.2021
"""

from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageChops
import sys
import string

ASCII_CHARS_ALL = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;?@[\\]^_`{|}~ "
ASCII_CHARS_DISABLE = ""




def resize(image, new_width = 50):
    width, height = image.size
    new_height = new_width * height // width
    return image.resize((new_width, new_height))

def to_greyscale(image):
    return image.convert("L")

def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = "";
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel//25];
    return ascii_str


def main():
    path = None
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")
    #resize image
    image = resize(image);
    #convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img=""
    #Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i:i+img_width] + "\n"
    #save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img);

def crop_background(image, back_color):
    bg = Image.new(image.mode, image.size, back_color)
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()

    return image.crop(bbox)


width = 50
height = 50
color_t = "black"
color_b = "white"
font = ImageFont.truetype("\\..\\fonts\\Tensura.ttf", 50)
char_w, char_h = font.getsize("A")

print(char_w, char_h)

canvas = Image.new('RGB', (width*2, height*2), color_b)
img_draw = ImageDraw.Draw(canvas)
img_draw.text((0,0), "A", fill=color_t, font=font)
#round(width/2), round(height/2)
canvas.show()
canvas = crop_background(canvas, ImageColor.getrgb(color_b))

canvas.show()


if __name__ == '__main__':
    pass

