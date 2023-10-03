import modules.scripts as scripts
import hitherdither
import gradio as gr

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import os
import requests
import colorsys
from io import BytesIO
from itertools import product


from modules import images
from modules.processing import process_images
from modules.ui import create_refresh_button
from modules.shared import opts
from sklearn.cluster import KMeans
from scipy import stats
script_dir = scripts.basedir()
from rembg import remove
import io


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

def remove_average_color(image: Image, threshold: int) -> Image:
    # Get the average color of the first 3x3 pixels in the top left corner
    average_color = [0, 0, 0]
    for x in range(3):
        for y in range(3):
            pixel_color = image.getpixel((x, y))
            average_color[0] += pixel_color[0]
            average_color[1] += pixel_color[1]
            average_color[2] += pixel_color[2]
    average_color = [int(c / 9) for c in average_color]

    # Create a mask to remove the average color
    mask = Image.new("L", image.size, 255) # Change initial value of mask to 255
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            pixel_color = image.getpixel((x, y))
            distance = sum([abs(pixel_color[i] - average_color[i]) for i in range(3)])
            if distance <= threshold:
                mask.putpixel((x, y), 0) # Change value of mask to 0

    # Convert the image to RGBA mode to support transparency
    image = image.convert("RGBA")

    # Apply the mask to remove the average color and make it transparent
    image = Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), mask)

    return image


# def remove_shadow(image: Image, lower_color, upper_color):
#     # Check if the image is a numpy array
#     if not isinstance(image, np.ndarray):
#         # Convert the image to a numpy array
#         image = np.array(image)

#     # Convert the image to HSV color space
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define range of color in HSV
#     lower_color = np.array(lower_color)
#     upper_color = np.array(upper_color)

#     # Threshold the HSV image to get only the specified colors
#     mask = cv2.inRange(hsv, lower_color, upper_color)

#     # Get the number of pixels that have a color within the specified range
#     num_pixels = np.sum(mask)

#     # Print the number of pixels
#     print(f"Number of pixels with color within range: {num_pixels}")

#     # Invert the mask
#     mask = cv2.bitwise_not(mask)

#     # Bitwise-AND mask and original image
#     res = cv2.bitwise_and(image, image, mask=mask)

#     # Convert numpy array to PIL Image
#     resImage = Image.fromarray(res)

#     return resImage


def remove_color_range(image: Image, color1: tuple, color2: tuple) -> Image:

    # Convert the image to grayscale
    grayscale_image = ImageOps.grayscale(image)

    # Apply thresholding to the grayscale image
    _, thresholded_image = cv2.threshold(np.array(grayscale_image), 127, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to the thresholded image
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(thresholded_image, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    # Convert the thresholded image back to a PIL image
    thresholded_image = Image.fromarray(eroded_image)

    # Invert the mask
    inverted_mask = ImageOps.invert(thresholded_image)

    # Convert the image to HSV color space
    hsv_image = image.convert("HSV")

    # Create a mask to remove the color range
    mask = Image.new("L", image.size, 255) # Change initial value of mask to 255
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            pixel_color = hsv_image.getpixel((x, y))
            if color1[0] <= pixel_color[0] <= color2[0] and color1[1] <= pixel_color[1] <= color2[1] and color1[2] <= pixel_color[2] <= color2[2]:
                if inverted_mask.getpixel((x, y)) ==  (0, 0, 0):
                    mask.putpixel((x, y), 0) # Change value of mask to 0

    # Convert the image to RGBA mode to support transparency
    image = image.convert("RGBA")

    # Apply the mask to remove the color range and make it transparent
    image = Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), mask)

    # Convert the image back to RGB mode
    image = image.convert("RGB")

    return inverted_mask




# def remove_color_range(image: Image, color1: tuple, color2: tuple) -> Image:
#     # Create a mask to remove the color range
#     mask = Image.new("L", image.size, 255) # Change initial value of mask to 255
#     for x in range(image.size[0]):
#         for y in range(image.size[1]):
#             pixel_color = image.getpixel((x, y))
#             if color1[0] <= pixel_color[0] <= color2[0] and color1[1] <= pixel_color[1] <= color2[1] and color1[2] <= pixel_color[2] <= color2[2]:
#                 mask.putpixel((x, y), 0) # Change value of mask to 0

#     # Convert the image to RGBA mode to support transparency
#     image = image.convert("RGBA")

#     # Apply the mask to remove the color range and make it transparent
#     image = Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), mask)

#     return image


def remove_shadow(image, color1, color2):
    # Convert the input image to RGB mode
    image = image.convert('RGB')
    
    # Get the dimensions of the image
    width, height = image.size

    # Create a mask to remove the average color
    mask = Image.new("L", image.size, 255) # Change initial value of mask to 255
    # Loop through each pixel in the image
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            r, g, b = image.getpixel((x, y))
            
            # Check if the pixel is between the two colors
            if not (color1[0] <= r <= color2[0] and color1[1] <= g <= color2[1] and color1[2] <= b <= color2[2]):
                # If not, set the pixel to black
                mask.putpixel((x, y), 0) # Change value of mask to 0

    # Get the number of pixels that have a color within the specified range
    num_pixels = np.sum(mask)

    # Print the number of pixels
    print(f"Number of pixels with color within range: {num_pixels}")

    # Convert the image to RGBA mode to support transparency
    image = image.convert("RGBA")

    # Apply the mask to remove the average color and make it transparent
    image = Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), mask)

    # Return the modified image
    return image

def make_background_transparent_contour(image, threshold):
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(image) 
    img = open_cv_image[:, :, ::-1].copy() 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold the image
    otsu_threshold, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    otsu_threshold = otsu_threshold - threshold

    print("Obtained threshold: ", otsu_threshold)

    otsu_threshold, thresh = cv2.threshold(gray, otsu_threshold, 255, cv2.THRESH_BINARY_INV)
    
    edged = cv2.Canny(thresh, 10, 255)

    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=1)

    # find the contours in the dilated image
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an all black mask
    mask = np.zeros_like(img)

    # Draw the contours on the mask with white
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Reduce the outer shape of the mask by two pixels
    mask = cv2.erode(mask, kernel, iterations=2)

    # Convert mask to 3-channels
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Create a 4-channel image (RGBA) from the original image
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    # Apply the mask to the image
    img_rgba[mask==0] = [0,0,0,0]

    # # Draw the contours on the image
    # cv2.drawContours(img_rgba, contours, -1, (0,255,0,255), 2)

    # Convert the result to PIL format
    result = Image.fromarray(img_rgba)

    return result


# def remove_background_processed(image, threshold):
#     # Convert the image to a NumPy array
#     image = np.array(image)

#     # Calculate the most common pixel color in the image
#     mode = stats.mode(image.reshape(-1, 3), axis=0)[0][0]

#     # Calculate the distance between each pixel and the most common pixel color
#     distance = np.sqrt(np.sum((image - mode) ** 2, axis=2))

#     # Create a mask to remove pixels that are within the threshold distance of the most common pixel color
#     mask = (distance > threshold).astype(np.uint8) * 255

#     # Convert the image to RGBA mode to support transparency
#     image = Image.fromarray(image).convert("RGBA")

#     # Apply the mask to remove the background color and make it transparent
#     image = Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), Image.fromarray(mask))
#     return image

# def remove_background_processed(image, threshold):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

#     # Apply Otsu's thresholding method
#     ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Create a mask to remove the background color
#     mask = Image.fromarray(thresh)

#     # Convert the image to RGBA mode to support transparency
#     image = image.convert("RGBA")

#     # Apply the mask to remove the background color and make it transparent
#     image = Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), mask)
#     return image


def remove_background_processed(image, threshold, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10):
    # Remove the background using rembg
    img = np.array(image)
    img = img[:, :, :3]
    img = Image.fromarray(img)
    input = io.BytesIO()
    img.save(input, format='PNG')
    input = input.getvalue()
    output = remove(input, alpha_matting=alpha_matting, alpha_matting_foreground_threshold=alpha_matting_foreground_threshold, alpha_matting_background_threshold=alpha_matting_background_threshold)
    output = Image.open(io.BytesIO(output)).convert('RGBA')
    
    # Apply threshold to alpha channel
    data = np.array(output)
    alpha = data[:, :, 3]
    alpha[alpha < threshold] = 0
    data[:, :, 3] = alpha
    output = Image.fromarray(data)
    
    return output


def change_contrast(image: Image, contrast_value: float):
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(contrast_value)
    return enhanced_image

def change_brightness(image: Image, brightness_value: float):
    enhancer = ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(brightness_value)
    return enhanced_image

def change_color(image: Image, color_value: float):
    enhancer = ImageEnhance.Color(image)
    enhanced_image = enhancer.enhance(color_value)
    return enhanced_image

def change_sharpness(image: Image, sharpness_value: float):
    enhancer = ImageEnhance.Sharpness(image)
    enhanced_image = enhancer.enhance(sharpness_value)
    return enhanced_image

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

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class Script(scripts.Script):
    def title(self):
        return "Palettize"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        limit = gr.Checkbox(label='Limit colors', value=True)
        clusters = gr.Slider(minimum=2, maximum=256, step=1,
                             label='Colors in palette', value=128)
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
            contrast = gr.Checkbox(label='Contrast of the image', value=False)
            contrast_value = gr.Slider(minimum=0, maximum=2, step=0.1,
                                label='contrast value', value=1)
        with gr.Row():
            brightness = gr.Checkbox(label='Brightness of the image', value=False)
            brightness_value = gr.Slider(minimum=0, maximum=2, step=0.1,
                                label='brightness value', value=1)
        with gr.Row():
            color = gr.Checkbox(label='Color of the image', value=False)
            color_value = gr.Slider(minimum=0, maximum=2, step=0.1,
                                label='color value', value=1)
        with gr.Row():
            sharpness = gr.Checkbox(label='Sharpness of the image', value=False)
            sharpness_value = gr.Slider(minimum=0, maximum=2, step=0.1,
                                label='sharpness value', value=1)     
        with gr.Row():
            transparentColor = gr.Checkbox(label='Remove background with color of top left color', value=False)
            thresholdColor = gr.Slider(minimum=0, maximum=255, step=1,
                                label='Threshold for postbackground removal', value=30)  
        with gr.Row():
            transparentRembg = gr.Checkbox(label='Remove background with rembg', value=False)
            thresholdRembg = gr.Slider(minimum=0, maximum=255, step=1,
                                label='Threshold', value=0)
        # with gr.Row():
            alpha_matting_foreground_threshold = gr.Slider(minimum=0, maximum=500, step=1,
                                label='alpha_matting_foreground_threshold', value=0)
        # with gr.Row():    
            alpha_matting_background_threshold = gr.Slider(minimum=0, maximum=100, step=1,
                                label='alpha_matting_background_threshold', value=0)
        with gr.Row():
            transparentContour = gr.Checkbox(label='Remove background with contour', value=False)
            thresholdContour = gr.Slider(minimum=0, maximum=255, step=1,
                                label='Threshold Contour', value=65)
        with gr.Row():
            removeShadows = gr.Checkbox(label='remove Shadows', value=False)
            shadowsLower_val = gr.ColorPicker(initial_color="red", label='shadowsLower_val (darker)', value="#828282")
            shadowsUpper_val = gr.ColorPicker(initial_color="red", label='shadowsUpper_val (lighter)', value="#ffffff")
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

        return [downscale, original, upscale, kcentroid, scale, transparentRembg, thresholdRembg, transparentContour, thresholdContour, transparentColor, thresholdColor, removeShadows, shadowsLower_val, shadowsUpper_val, contrast, contrast_value, brightness, brightness_value, color, color_value, sharpness, sharpness_value,paletteDropdown, paletteURL, palette, limit, clusters, dither, ditherStrength, alpha_matting_foreground_threshold, alpha_matting_background_threshold] 

    def run(self, p, downscale, original, upscale, kcentroid, scale, transparentRembg, thresholdRembg, transparentContour, thresholdContour, transparentColor, thresholdColor, removeShadows, shadowsLower_val, shadowsUpper_val, contrast, contrast_value, brightness, brightness_value, color, color_value, sharpness, sharpness_value, paletteDropdown, paletteURL, palette, limit, clusters, dither, ditherStrength, alpha_matting_foreground_threshold, alpha_matting_background_threshold):

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
            if limit:
                tempImg = palettize(img, clusters, palette, dither, ditherStrength)
            else:
                tempImg = img

            if downscale:
                img = cv2.resize(tempImg, (int(
                    img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)


            if not upscale:
                tempImg = img

            processed.images[i] = Image.fromarray(img)
            images.save_image(Image.fromarray(tempImg), p.outpath_samples, "palettized",
                            processed.seed + i, processed.prompt, opts.samples_format, info=processed.info, p=p)

            if transparentRembg | transparentColor | transparentContour| removeShadows | contrast | brightness | color | sharpness:
                transimage = processed.images[i]
                if transparentColor:
                    transimage = remove_average_color(transimage, thresholdColor)
                if transparentRembg:
                    transimage = remove_background_processed(transimage, thresholdRembg, True, alpha_matting_foreground_threshold, alpha_matting_background_threshold)
                if transparentContour:
                    transimage = make_background_transparent_contour(transimage, thresholdContour)
                if removeShadows:
                    transimage = remove_color_range(transimage, hex_to_rgb(shadowsLower_val), hex_to_rgb(shadowsUpper_val))
                if contrast:
                    processed.images[i] = change_contrast(processed.images[i], contrast_value)
                    transimage = change_contrast(transimage, contrast_value)
                if brightness:
                    processed.images[i] = change_brightness(processed.images[i], brightness_value)
                    transimage = change_brightness(transimage, brightness_value)
                if color:
                    processed.images[i] = change_color(processed.images[i], color_value)
                    transimage = change_color(transimage, color_value)
                if sharpness:
                    processed.images[i] = change_sharpness(processed.images[i], sharpness_value)
                    transimage = change_sharpness(transimage, sharpness_value)
 

                images.save_image(transimage, p.outpath_samples, "palettized_transparent", processed.seed + i, processed.prompt, opts.samples_format, info=processed.info, p=p)
                transparent_images.append(transimage) 
                if upscale:
                    np_img = np.array(transimage)
                    downscaled_img = cv2.resize(np_img, (int(np_img.shape[1]/scale), int(np_img.shape[0]/scale)), interpolation=cv2.INTER_LINEAR)
                    downscaled_pil_img = Image.fromarray(downscaled_img)
                    images.save_image(downscaled_pil_img, p.outpath_samples, "palettized_transparent_downscaled", processed.seed + i, processed.prompt, opts.samples_format, info=processed.info, p=p)
            else:
                if contrast:
                    processed.images[i] = change_contrast(processed.images[i], contrast_value)
                if brightness:
                    processed.images[i] = change_brightness(processed.images[i], brightness_value)
                if color:
                    processed.images[i] = change_color(processed.images[i], color_value)
                if sharpness:
                    processed.images[i] = change_sharpness(processed.images[i], sharpness_value)

            if grid:
                processed.images[0] = images.image_grid(processed.images[1:generations], p.batch_size)
              

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "palettized",
                              prompt=p.prompt, seed=processed.seed, grid=True, p=p)

        if original:
            processed.images.extend(originalImgs)

        if transparentRembg | transparentColor | transparentContour | removeShadows:
             processed.images.extend(transparent_images)

        return processed

    

