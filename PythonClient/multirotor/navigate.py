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
from tkinter import messagebox


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

def discoverFire():
    # get depth image
    yaw = 0
    pi = 3.14159265483
    vx = 0
    vy = 0
    count = 0
    x = 10 # default
    for i in range(1, 30):
        mostLikelyInd, mostLikely, navImage = captureBandImages(client, tmp_dir, ort_session, cfg, count)
        pitch, roll, yaw  = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)

        if (mostLikely > 0.5):
        
            # we have a 90 degree field of view (pi/2), we've sliced that into 5 chunks, each chunk then represents
            # an angular delta of the following pi/10.
            change = 0
            driving = mostLikely
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

            while True:
                client.moveByVelocityAsync(vx, vy, 0, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()
                time.sleep(2)
                mostLikelyInd, mostLikely, fireImage = captureBandImages(client, tmp_dir, ort_session, cfg, 100)
                if mostLikely > 0.9:
                    print(mostLikely)
                    print("INSERT THE EMAIL CODE HERE")
                    messagebox.showerror("Fire Found!", "Sending email with image and GPS locations")
                    sendEmailWrapper(client, mostLikely, fireImage) 

                    client.hoverAsync().join()
                    time.sleep(10)
                    return mostLikely;
                elif mostLikely < 0.3:
                    print("Lower likelihood of fire, do some surveying.")
                    break;

        print("Surveying...")
        if i % 5 != 0:
            client.moveByVelocityAsync(0, x, 0, 5, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0))
            time.sleep(5) 
        else:
            client.moveByVelocityAsync(-10, 0, 0, 5, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0))
            x = x * -1
            time.sleep(5) 

likelihoodScore = discoverFire()
print("ALERT RAISED - FIRE FOUND WITH LIKELIHOOD SCORE OF " + str(likelihoodScore))

airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)