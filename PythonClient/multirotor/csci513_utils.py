import setup_path
import airsim
import time

import numpy as np
import os
import tempfile
import pprint
import cv2
import io

# pyrovision imports
import argparse
import json

import gradio as gr
import onnxruntime
from huggingface_hub import hf_hub_download
from PIL import Image

def sendEmail(client, imageBytes):
    completed = False
    try:
        gps_data = client.getGpsData()
        geo_point = gps_data.gnss.geo_point

        coords = '{},{}'.format(geo_point.latitude, geo_point.longitude)
        # function(coords, imageBytes)
        completed = True
    except: 
        print("Error with sending email. Printing out GPS coords for now.")
        print(coords)

    return completed


def captureBandImages(client, tmp_dir, ort_session, cfg, count):
    img = client.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene), airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
    filename = os.path.join(tmp_dir, str("nav"))
    response = img[0] # PNG format -> for saving screenshots
    rgba_response = img[1] # RGBA format -> for navigation

    airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    img1d = np.fromstring(rgba_response.image_data_uint8, dtype=np.uint8)
    img_rgb_o = img1d.reshape(rgba_response.height, rgba_response.width, 3)

    img_rgb = img_rgb_o[:,:,::-1]

    imgp = Image.fromarray(img_rgb, 'RGB') #Image.open(io.BytesIO(rgba_response.image_data_uint8))

    # slice the image so we only check what we are headed into (and not what is down on the ground below us).
    # MJ - edit to check only the bottom part of the image
    bottom = np.vsplit(img_rgb, 2)[1]

    # now look at 4 horizontal bands (far left, left, right, far right) and see which is most open.
    # the depth map uses black for far away (0) and white for very close (255), so we invert that
    # to get an estimate of distance.
    bands = np.hsplit(bottom, 4);
    #print(bands)

    likelihoods = []
    #test = predict(Image.fromarray(bands[0], 'RGB'), ort_session, cfg)
    for band in bands:
        filename2 = os.path.join(tmp_dir, str(count))
        band_invert = band[:,:,::-1]
        cv2.imwrite(os.path.normpath(filename2 + '_bands.png'), band_invert) # write to png
        result = navigatePredict(Image.fromarray(band, 'RGB'), ort_session, cfg)
        likelihoods.append(result)
        count += 1

    
    mostLikely = np.max(likelihoods)
    mostLikelyInd = np.argmax(likelihoods)
    #print(mostLikely)
    return mostLikelyInd, mostLikely, response.image_data_uint8


def preprocess_image(pil_img: Image.Image, cfg) -> np.ndarray:
    """Preprocess an image for inference
    Args:
        pil_img: a valid pillow image
    Returns:
        the resized and normalized image of shape (1, C, H, W)
    """

    # Resizing (PIL takes (W, H) order for resizing)
    img = pil_img.resize(cfg["input_shape"][-2:][::-1], Image.BILINEAR)
    # (H, W, C) --> (C, H, W)
    img = np.asarray(img).transpose((2, 0, 1)).astype(np.float32) / 255
    # Normalization
    img -= np.array(cfg["mean"])[:, None, None]
    img /= np.array(cfg["std"])[:, None, None]
    #print("Completing preprocessing of input image.")

    return img[None, ...]

def predict(image, ort_session, cfg):
    # Preprocessing
    np_img = preprocess_image(image, cfg)
    ort_input = {ort_session.get_inputs()[0].name: np_img}

    # Inference
    ort_out = ort_session.run(None, ort_input)
    # Post-processing
    probs = 1 / (1 + np.exp(-ort_out[0][0]))

    return {class_name: float(conf) for class_name, conf in zip(cfg["classes"], probs)}

def navigatePredict(image, ort_session, cfg):
    resultForImg = predict(image, ort_session, cfg)
    #print(resultForImg)
    likelihood = resultForImg['Wildfire'] # should be the percentrage
    #print(likelihood)
    return likelihood


def predictIfFire(count = 0):
    # get images of the surroudnigns
    img = client.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene), airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
    filename = os.path.join(tmp_dir, str(count))
    response = img[0] # PNG format -> for saving screenshots
    rgba_response = img[1] # RGBA format -> for navigation

    airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    img1d = np.fromstring(rgba_response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(rgba_response.height, rgba_response.width, 3)
    img_rgb = img_rgb[:,:,::-1]
    imgp = Image.fromarray(img_rgb, 'RGB') #Image.open(io.BytesIO(rgba_response.image_data_uint8))
    
    resultForImg = predict(imgp)
    print(resultForImg)
    likelihood = resultForImg['Wildfire'] # should be the percentrage
    count += 1
    print(likelihood)
    return count, likelihood

def likelihoodCheck(likelihood):
    if likelihood >= 0.9:
        print("FIREEEEEEE")
        print("do something special and go forward")
        return True
    return False