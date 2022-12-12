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

#def captureBandImages


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
    print("Completing preprocessing of input image.")

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
    print(resultForImg)
    likelihood = resultForImg['Wildfire'] # should be the percentrage
    print(likelihood)
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