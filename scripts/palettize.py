import modules.scripts as scripts
import hitherdither
import gradio as gr

import cv2
import numpy as np
from PIL import Image
import os
import requests
import colorsys
from io import BytesIO
from itertools import product

from modules import images
from modules.processing import process_images
from modules.ui import create_refresh_button
from modules.shared import opts

script_dir = scripts.basedir()


def refreshPalettes():
    palettes = ["None", "Automatic"]
    palettes.extend(os.listdir('./extensions/sd-palettize/palettes/'))
    return palettes


def adjust_gamma(image, gamma=1.0):
    # Create a lookup table for the gamma function
    gamma_map = [255 * ((i / 255.0) ** (1.0 / gamma)) for i in range(256)]
    gamma_table = bytes([(int(x / 255.0 * 65535.0) >> 8)
                        for x in gamma_map] * 3)

    # Apply the gamma correction using the lookup table
    return image.point(gamma_table)


def kCentroid(image: Image, width: int, height: int, centroids: int):
    image = image.convert("RGB")
    downscaled = np.zeros((height, width, 3), dtype=np.uint8)
    wFactor = image.width/width
    hFactor = image.height/height
    for x, y in product(range(width), range(height)):
        tile = image.crop((x*wFactor, y*hFactor, (x*wFactor)+wFactor, (y*hFactor)+hFactor)
                          ).quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")
        color_counts = tile.getcolors()
        most_common_color = max(color_counts, key=lambda x: x[0])[1]
        downscaled[y, x, :] = most_common_color
    return Image.fromarray(downscaled, mode='RGB')


def determine_best_k(image, max_k):
    # Convert the image to RGB mode
    image = image.convert("RGB")

    # Prepare arrays for distortion calculation
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    # Calculate distortion for different values of k
    distortions = []
    for k in range(1, max_k + 1):
        quantized_image = image.quantize(
            colors=k, method=2, kmeans=k, dither=0)
        centroids = np.array(quantized_image.getpalette()
                             [:k * 3]).reshape(-1, 3)

        # Calculate distortions
        distances = np.linalg.norm(
            pixel_indices[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        distortions.append(np.sum(min_distances ** 2))

    # Calculate the rate of change of distortions
    rate_of_change = np.diff(distortions) / np.array(distortions[:-1])

    # Find the elbow point (best k value)
    if len(rate_of_change) == 0:
        best_k = 2
    else:
        elbow_index = np.argmax(rate_of_change) + 1
        best_k = elbow_index + 2

    return best_k

# Runs cv2 k_means quantization on the provided image with "k" color indexes
# def remove_background_tempImg(image, threshold):
#     # Get the first pixel color
#     first_pixel_color = image[0][0]

#     # Create a mask to remove the background color
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     for x in range(image.shape[1]):
#         for y in range(image.shape[0]):
#             pixel_color = image[y][x]
#             distance = sum([abs(int(pixel_color[i]) - int(first_pixel_color[i])) for i in range(3)])
#             if distance > threshold:
#                 mask[y][x] = 255

#     # Apply the mask to remove the background color and make it transparent
 
#     # Convert the image to RGBA mode to support transparency
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
#     image[:, :, 3] = mask
#     return image

def remove_background_processed(image, threshold):
    # Get the first pixel color
    first_pixel_color = image.getpixel((0, 0))

    # Create a mask to remove the background color
    mask = Image.new("L", image.size, 0)
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            pixel_color = image.getpixel((x, y))
            distance = sum([abs(pixel_color[i] - first_pixel_color[i]) for i in range(3)])
            if distance > threshold:
                mask.putpixel((x, y), 255)

    # Convert the image to RGBA mode to support transparency
    image = image.convert("RGBA")

    # Apply the mask to remove the background color and make it transparent
    image = Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), mask)
    return image

def desaturate_image(image, desaturation_value):
    image = image.convert('RGBA')
    pixels = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a != 0:
                h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                s = max(0, s - desaturation_value)
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                pixels[x, y] = (int(r * 255), int(g * 255), int(b * 255), a)
    return image

def palettize(input, colors, palImg, dithering, strength):
    img = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).convert("RGB")

    dithering += 1

    if palImg is not None:
        palImg = cv2.cvtColor(palImg, cv2.COLOR_BGR2RGB)
        palImg = Image.fromarray(palImg).convert("RGB")
        numColors = len(palImg.getcolors(16777216))
    else:
        numColors = colors

    palette = []

    threshold = (16*strength)/4

    if palImg is not None:

        numColors = len(palImg.getcolors(16777216))

        if strength > 0:
            img = adjust_gamma(img, 1.0-(0.02*strength))
            for i in palImg.getcolors(16777216):
                palette.append(i[1])
            palette = hitherdither.palette.Palette(palette)
            img_indexed = hitherdither.ordered.bayer.bayer_dithering(
                img, palette, [threshold, threshold, threshold], order=2**dithering).convert('RGB')
        else:
            for i in palImg.getcolors(16777216):
                palette.append(i[1][0])
                palette.append(i[1][1])
                palette.append(i[1][2])
            palImg = Image.new('P', (256, 1))
            palImg.putpalette(palette)
            img_indexed = img.quantize(
                method=1, kmeans=numColors, palette=palImg, dither=0).convert('RGB')
    elif colors > 0:

        if strength > 0:
            img_indexed = img.quantize(
                colors=colors, method=1, kmeans=colors, dither=0).convert('RGB')
            img = adjust_gamma(img, 1.0-(0.03*strength))
            for i in img_indexed.convert("RGB").getcolors(16777216):
                palette.append(i[1])
            palette = hitherdither.palette.Palette(palette)
            img_indexed = hitherdither.ordered.bayer.bayer_dithering(
                img, palette, [threshold, threshold, threshold], order=2**dithering).convert('RGB')

        else:
            img_indexed = img.quantize(
                colors=colors, method=1, kmeans=colors, dither=0).convert('RGB')

    result = cv2.cvtColor(np.asarray(img_indexed), cv2.COLOR_RGB2BGR)
    return result


class Script(scripts.Script):
    def title(self):
        return "Palettize"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):

        clusters = gr.Slider(minimum=2, maximum=128, step=1,
                             label='Colors in palette', value=24)
        with gr.Row():
            downscale = gr.Checkbox(
                label='Downscale before processing', value=True)
            original = gr.Checkbox(label='Show original images', value=False)
        with gr.Row():
            upscale = gr.Checkbox(label='Save 1:1 pixel image', value=False)
            kcentroid = gr.Checkbox(
                label='Use K-Centroid algorithm for downscaling', value=True)
        with gr.Row():
            scale = gr.Slider(minimum=2, maximum=32, step=1,
                              label='Downscale factor', value=8)
        with gr.Row():
            desaturation = gr.Checkbox(label='Desaturation of the image', value=False)
            desaturation_value = gr.Slider(minimum=0, maximum=1, step=0.05,
                                label='desaturation value', value=0.1)
        with gr.Row():
            transparent = gr.Checkbox(label='Make background transparent', value=False)
            threshold = gr.Slider(minimum=0, maximum=255, step=1,
                                label='Threshold for background removal', value=10)
      
        with gr.Row():
            dither = gr.Dropdown(choices=["Bayer 2x2", "Bayer 4x4", "Bayer 8x8"],
                                 label="Matrix Size", value="Bayer 8x8", type="index")
            ditherStrength = gr.Slider(
                minimum=0, maximum=10, step=1, label='Dithering Strength', value=0)
        with gr.Row():
            paletteDropdown = gr.Dropdown(
                choices=refreshPalettes(), label="Palette", value="None", type="value")
            create_refresh_button(paletteDropdown, refreshPalettes, lambda: {
                                  "choices": refreshPalettes()}, None)
        with gr.Row():
            paletteURL = gr.Textbox(
                max_lines=1, placeholder="Image URL (example:https://lospec.com/palette-list/pear36-1x.png)", label="Palette URL")
        with gr.Row():
            palette = gr.Image(label="Palette image")

        return [downscale, original, upscale, kcentroid, scale, transparent, threshold, desaturation, desaturation_value, paletteDropdown, paletteURL, palette, clusters, dither, ditherStrength]

    def run(self, p, downscale, original, upscale, kcentroid, scale, transparent, threshold, desaturation, desaturation_value, paletteDropdown, paletteURL, palette, clusters, dither, ditherStrength):

        if ditherStrength > 0:
            print(
                f'Palettizing output to {clusters} colors with order {2**(dither+1)} dithering...')
        else:
            print(f'Palettizing output to {clusters} colors...')

        if paletteDropdown != "None" and paletteDropdown != "Automatic":
            palette = cv2.cvtColor(cv2.imread(
                "./extensions/sd-palettize/palettes/"+paletteDropdown), cv2.COLOR_RGB2BGR)

        if paletteURL != "":
            try:
                palette = np.array(Image.open(BytesIO(requests.get(
                    paletteURL).content)).convert("RGB")).astype(np.uint8)
            except:
                print("An error occured fetching image from URL")

        processed = process_images(p)

        generations = p.batch_size*p.n_iter

        grid = False

        if opts.return_grid and p.batch_size*p.n_iter > 1:
            generations += 1
            grid = True

        originalImgs = []
        transparent_images = []

        for i in range(generations):
            # Converts image from "Image" type to numpy array for cv2

            img = np.array(processed.images[i]).astype(np.uint8)

            if original:
                originalImgs.append(processed.images[i])

            if downscale:
                if kcentroid:
                    img = Image.fromarray(cv2.cvtColor(
                        img, cv2.COLOR_BGR2RGB)).convert("RGB")
                    img = cv2.cvtColor(np.asarray(kCentroid(
                        img, int(img.width/scale), int(img.height/scale), 2)), cv2.COLOR_RGB2BGR)
                else:
                    img = cv2.resize(img, (int(
                        img.shape[1]/scale), int(img.shape[0]/scale)), interpolation=cv2.INTER_LINEAR)

            if paletteDropdown == "Automatic":
                palImg = Image.fromarray(cv2.cvtColor(
                    img, cv2.COLOR_BGR2RGB)).convert("RGB")
                best_k = determine_best_k(palImg, 64)
                palette = cv2.cvtColor(np.asarray(palImg.quantize(
                    colors=best_k, method=1, kmeans=best_k, dither=0).convert('RGB')), cv2.COLOR_RGB2BGR)

            tempImg = palettize(img, clusters, palette, dither, ditherStrength)

            if downscale:
                img = cv2.resize(tempImg, (int(
                    img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)


            if not upscale:
                tempImg = img

            processed.images[i] = Image.fromarray(img)
            images.save_image(Image.fromarray(tempImg), p.outpath_samples, "palettized",
                            processed.seed + i, processed.prompt, opts.samples_format, info=processed.info, p=p)

            if transparent:
                transimage = remove_background_processed(processed.images[i], threshold)
                if desaturation:
                    processed.images[i] = desaturate_image(processed.images[i], desaturation_value)
                    transimage = desaturate_image(transimage, desaturation_value)
                images.save_image(transimage, p.outpath_samples, "palettized_transparent", processed.seed + i, processed.prompt, opts.samples_format, info=processed.info, p=p)
                transparent_images.append(transimage) 
                if upscale:
                    np_img = np.array(transimage)
                    downscaled_img = cv2.resize(np_img, (int(np_img.shape[1]/scale), int(np_img.shape[0]/scale)), interpolation=cv2.INTER_LINEAR)
                    downscaled_pil_img = Image.fromarray(downscaled_img)
                    images.save_image(downscaled_pil_img, p.outpath_samples, "palettized_transparent_downscaled", processed.seed + i, processed.prompt, opts.samples_format, info=processed.info, p=p)
            else:
                if desaturation:
                     processed.images[i] = desaturate_image(processed.images[i], desaturation_value)

            if grid:
                processed.images[0] = images.image_grid(processed.images[1:generations], p.batch_size)
              

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "palettized",
                              prompt=p.prompt, seed=processed.seed, grid=True, p=p)

        if original:
            processed.images.extend(originalImgs)

        if transparent:
             processed.images.extend(transparent_images)

        return processed

    

