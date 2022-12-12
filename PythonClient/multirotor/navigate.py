# use open cv to show new images from AirSim 

import setup_path 
import airsim

# requires Python 3.5.3 :: Anaconda 4.4.0
# pip install opencv-python
import cv2
import time
import math
import sys
import numpy as np
import os
import tempfile
import pprint
import io

# pyrovision imports
import argparse
import json

import gradio as gr
import onnxruntime
from huggingface_hub import hf_hub_download
from PIL import Image
from csci513_utils import *

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

airsim.wait_key('Press any key to move vehicle to (-10, 10, -10) at 5 m/s')
# client.moveToPositionAsync(20450, -19220, 11640, 5).join()
client.moveToPositionAsync(-10, 10, -10, 5).join()


client.hoverAsync().join()


# setup the details needed for detection before connecting to the simulator
REPO = "pyronear/rexnet1_0x"

# Download model config & checkpoint
with open(hf_hub_download(REPO, filename="config.json"), "rb") as f:
    cfg = json.load(f)

ort_session = onnxruntime.InferenceSession(hf_hub_download(REPO, filename="model.onnx"))
tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise
# you must first press "1" in the AirSim view to turn on the depth capture

# get depth image
yaw = 0
pi = 3.14159265483
vx = 0
vy = 0
driving = 0
help = False
count = 0
ind = 0
while True:
    # this will return png width= 256, height= 144
    #result = client.simGetImage("0", airsim.ImageType.DepthVis)
    #if (result == "\0"):
    #    if (not help):
    #        help = True
    #        print("Please press '1' in the AirSim view to enable the Depth camera view")
    #else:    
    #rawImage = np.fromstring(result, np.int8)
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
    print(bands)

    likelihoods = []
    #test = predict(Image.fromarray(bands[0], 'RGB'), ort_session, cfg)
    for band in bands:
        filename2 = os.path.join(tmp_dir, str(count))
        band_invert = band[:,:,::-1]
        cv2.imwrite(os.path.normpath(filename2 + '_bands.png'), band_invert) # write to png
        result = navigatePredict(Image.fromarray(band, 'RGB'), ort_session, cfg)
        likelihoods.append(result)
        count += 1

    mostLikely = np.argmax(likelihoods)
    print(mostLikely)

    
    pitch, roll, yaw  = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)

    if (max(likelihoods) > 0.5):
        
        # we have a 90 degree field of view (pi/2), we've sliced that into 5 chunks, each chunk then represents
        # an angular delta of the following pi/10.
        change = 0
        driving = max(likelihoods)
        if (min == 0):
            change = -2 * pi / 10
        elif (min == 1):
            change = -pi / 10
        elif (min == 2):
            change = 0 # center strip, go straight
        elif (min == 3):
            change = pi / 10
        else:
            change = 2*pi/10
    
        yaw = (yaw + change)
        vx = math.cos(yaw);
        vy = math.sin(yaw);
        print ("switching angle", math.degrees(yaw), vx, vy, min)
    
    if (vx == 0 and vy == 0):
        vx = math.cos(yaw);
        vy = math.sin(yaw);

    while True:
        client.moveByVelocityAsync(vx, vy, 0, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()
        time.sleep(2)

        ##print ("distance=", current)
        #if i % 5 == 0:
        #    client.moveByVelocityAsync(0, x, 0, 5, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0))
        #    time.sleep(5) 
        #    # client.simPause(True)
        #else:
        #    client.moveByVelocityAsync(-10, 0, 0, 5, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0))
        #    x = x * -1
        #    # client.simPause(True)
        #    time.sleep(5) 


        #cv2.rectangle(png, (x,0), (x+50,50), (0,255,0), 2)
        #cv2.imshow("Top", png)

    key = cv2.waitKey(1) & 0xFF;
    if (key == 27 or key == ord('q') or key == ord('x')):
        break;
