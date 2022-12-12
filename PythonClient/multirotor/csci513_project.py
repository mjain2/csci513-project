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

from csci513_utils import *

# setup the details needed for detection before connecting to the simulator
# TODO: move to another helper function probably
REPO = "pyronear/rexnet1_0x"

# Download model config & checkpoint
with open(hf_hub_download(REPO, filename="config.json"), "rb") as f:
    cfg = json.load(f)

ort_session = onnxruntime.InferenceSession(hf_hub_download(REPO, filename="model.onnx"))

#def preprocess_image(pil_img: Image.Image) -> np.ndarray:
#    """Preprocess an image for inference
#    Args:
#        pil_img: a valid pillow image
#    Returns:
#        the resized and normalized image of shape (1, C, H, W)
#    """

#    # Resizing (PIL takes (W, H) order for resizing)
#    img = pil_img.resize(cfg["input_shape"][-2:][::-1], Image.BILINEAR)
#    # (H, W, C) --> (C, H, W)
#    img = np.asarray(img).transpose((2, 0, 1)).astype(np.float32) / 255
#    # Normalization
#    img -= np.array(cfg["mean"])[:, None, None]
#    img /= np.array(cfg["std"])[:, None, None]
#    print("Completing preprocessing of input image.")

#    return img[None, ...]

#def predict(image):
#    # Preprocessing
#    np_img = preprocess_image(image)
#    ort_input = {ort_session.get_inputs()[0].name: np_img}

#    # Inference
#    ort_out = ort_session.run(None, ort_input)
#    # Post-processing
#    probs = 1 / (1 + np.exp(-ort_out[0][0]))

#    return {class_name: float(conf) for class_name, conf in zip(cfg["classes"], probs)}

#def getPath():
#    path = []
#    distance = 0
#    #while x < self.boxsize:
#    #    distance += self.boxsize
#    #    path.append(Vector3r(x, self.boxsize, z))
#    #    x += self.stripewidth
#    #    distance += self.stripewidth
#    #    path.append(Vector3r(x, self.boxsize, z))
#    #    distance += self.boxsize
#    #    path.append(Vector3r(x, -self.boxsize, z))
#    #    x += self.stripewidth
#    #    distance += self.stripewidth
#    #    path.append(Vector3r(x, -self.boxsize, z))
#    #    distance += self.boxsize

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

imu_data = client.getImuData()
s = pprint.pformat(imu_data)
print("imu_data: %s" % s)

barometer_data = client.getBarometerData()
s = pprint.pformat(barometer_data)
print("barometer_data: %s" % s)

magnetometer_data = client.getMagnetometerData()
s = pprint.pformat(magnetometer_data)
print("magnetometer_data: %s" % s)

gps_data = client.getGpsData()
s = pprint.pformat(gps_data)
print("gps_data: %s" % s)

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to move vehicle to (-10, 10, -10) at 5 m/s')
# client.moveToPositionAsync(20450, -19220, 11640, 5).join()
client.moveToPositionAsync(-10, 10, -10, 5).join()


client.hoverAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

z = -10
print("make sure we are hovering at {} meters...".format(-z))
client.moveToZAsync(z, 1).join()

x = 10

"""
Default behavior: survey area, turning every 5 pictures
If wildfire < 0.5 : Continue default behavior
If 0.5 < wildfire < 0.9: turn towards fire, move towards fire until:
    1. wildfire > 0.9 : send location data etc
    2. wildfire < 0.3 : resume default behavior
if wildfire > 0.9 : send location data, image, and confidence via email etc
"""

for i in range(1, 30):
    img = client.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene), airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
    filename = os.path.join(tmp_dir, str(i))
    response = img[0] # PNG format
    rgba_response = img[1]

    print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
    airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    #print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
    #img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
    #img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
    #cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
    img1d = np.fromstring(rgba_response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(rgba_response.height, rgba_response.width, 3)
    #print(img1d.shape)
    #print(img_rgb.shape)
    # print('Retrieved and saved images: %d' % len(response))
    img_rgb  = img_rgb[...,::-1].copy()

    imgp = Image.fromarray(img_rgb, 'RGB') 
    #imgp.save("C:/Users/WangC/AppData/Local/Temp/airsim_drone/test.png")
    resultForImg = predict(imgp, ort_session, cfg)
    print(resultForImg['Wildfire'])
    #print(resultForImg)
    print("flying on path...")
    # result = client.moveOnPathAsync([airsim.Vector3r(125,0,z),
    #                                 airsim.Vector3r(125,-130,z),
    #                                 airsim.Vector3r(0,-130,z),
    #                                 airsim.Vector3r(0,0,z)],
    #                         12, 120,
    #                         airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 20, 1).join()


    if i % 5 != 0:
        client.moveByVelocityAsync(0, x, 0, 5, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0))
        time.sleep(5) 
        # client.simPause(True)
    else:
        client.moveByVelocityAsync(-10, 0, 0, 5, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0))
        x = x * -1
        # client.simPause(True)
        time.sleep(5) 

airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
