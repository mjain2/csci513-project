import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
import io
import time

# pyrovision imports
import argparse
import json

import gradio as gr
import onnxruntime
from huggingface_hub import hf_hub_download
from PIL import Image

# setup the details needed for detection before connecting to the simulator
# TODO: move to another helper function probably
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

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
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

def predict(image):
    # Preprocessing
    np_img = preprocess_image(image)
    ort_input = {ort_session.get_inputs()[0].name: np_img}

    # Inference
    ort_out = ort_session.run(None, ort_input)
    # Post-processing
    probs = 1 / (1 + np.exp(-ort_out[0][0]))

    return {class_name: float(conf) for class_name, conf in zip(cfg["classes"], probs)}

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

def start():
    print("arming the drone...")
    client.armDisarm(True)

    landed = client.getMultirotorState().landed_state
    if landed == airsim.LandedState.Landed:
        print("taking off...")
        client.takeoffAsync().join()

    landed = client.getMultirotorState().landed_state
    if landed == airsim.LandedState.Landed:
        print("takeoff failed - check Unreal message log for details")
        return
        
    # AirSim uses NED coordinates so negative axis is up.
    x = -boxsize
    z = -altitude

    print("climbing to altitude: " + str(altitude))
    client.moveToPositionAsync(0, 0, z, velocity).join()

    print("flying to first corner of survey box")
    client.moveToPositionAsync(x, -boxsize, z, velocity).join()
        
    # let it settle there a bit.
    client.hoverAsync().join()
    time.sleep(2)

    # after hovering we need to re-enabled api control for next leg of the trip
    client.enableApiControl(True)

    # now compute the survey path required to fill the box 
    path = []
    path2 = []
    path3 = []
    path4 = []
    distance = 0
    while x < boxsize:
        distance += boxsize 
        path.append(airsim.Vector3r(x, boxsize, z))
        x += stripewidth            
        distance += stripewidth 
        path2.append(airsim.Vector3r(x, boxsize, z))
        distance += boxsize 
        path3.append(airsim.Vector3r(x, -boxsize, z)) 
        x += stripewidth  
        distance += stripewidth 
        path4.append(airsim.Vector3r(x, -boxsize, z))
        distance += boxsize 
        
    print("starting survey, estimated distance is " + str(distance))
    trip_time = distance / velocity
    print("estimated survey time is " + str(trip_time))
    count = 0
    try:
        
        #result = client.moveOnPathAsync(path, velocity, trip_time, airsim.DrivetrainType.ForwardOnly, 
        #    airsim.YawMode(False,0), velocity + (velocity/2), 1).join()
        #count, likelihood = predictIfFire(count)
        #likelihoodCheck(likelihood)
        #result2 = client.moveOnPathAsync(path2, velocity, trip_time, airsim.DrivetrainType.ForwardOnly, 
        #    airsim.YawMode(False,0), velocity + (velocity/2), 1).join()
        #count, likelihood2 = predictIfFire(count)
        #likelihoodCheck(likelihood2)

        #result3 = client.moveOnPathAsync(path3, velocity, trip_time, airsim.DrivetrainType.ForwardOnly, 
        #    airsim.YawMode(False,0), velocity + (velocity/2), 1).join()
        #count, likelihood3 = predictIfFire(count)
        #likelihoodCheck(likelihood3)

        #result4 = client.moveOnPathAsync(path4, velocity, trip_time, airsim.DrivetrainType.ForwardOnly, 
        #    airsim.YawMode(False,0), velocity + (velocity/2), 1).join()
        #count, likelihood4 = predictIfFire(count)
        #likelihoodCheck(likelihood4)

        count, likelihood = predictIfFire(count)
        boolL = likelihoodCheck(likelihood)
        while not boolL:
            for i in range(20):
                result = client.moveByVelocityZAsync(x, 1, 0, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1).join()
                count, likelihood = predictIfFire(count)
                boolL = likelihoodCheck(likelihood)

        print("Found fire with confidence above 90%. Exiting simulation.")
        client.hoverAsync().join()
        while True:
            time.sleep(2)
            key = cv2.waitKey(1) & 0xFF;
            if (key == 27 or key == ord('q') or key == ord('x')):
                print("Breaking")
                break
    except:
        errorType, value, traceback = sys.exc_info()
        print("moveOnPath threw exception: " + str(value))
        pass

    print("flying back home")
    client.moveToPositionAsync(0, 0, z, velocity).join()
        
    if z < -5:
        print("descending")
        client.moveToPositionAsync(0, 0, -5, 2).join()

    print("landing...")
    client.landAsync().join()

    print("disarming.")
    client.armDisarm(False)


boxsize = 5
stripewidth = 2
altitude = 30
velocity = 5
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

start()



'''
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

for i in range(0, 30):
    img = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene), airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
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
    print(img1d.shape)
    print(img_rgb.shape)
    # print('Retrieved and saved images: %d' % len(response))

    imgp = Image.fromarray(img_rgb, 'RGB') #Image.open(io.BytesIO(rgba_response.image_data_uint8))
    resultForImg = predict(imgp)
    print(resultForImg)
    print("flying on path...")
    result = client.moveOnPathAsync([airsim.Vector3r(125,0,z),
                                    airsim.Vector3r(125,-130,z),
                                    airsim.Vector3r(0,-130,z),
                                    airsim.Vector3r(0,0,z)],
                            12, 120,
                            airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 20, 1).join()


#airsim.wait_key('Press any key to take images')
## get camera images from the car
#responses = client.simGetImages([
#    airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
#    airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
#    airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
#    airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
#print('Retrieved images: %d' % len(responses))


#for idx, response in enumerate(responses):

#    filename = os.path.join(tmp_dir, str(idx))

#    if response.pixels_as_float:
#        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#        airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
#    elif response.compress: #png format
#        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
#    else: #uncompressed array
#        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
#        img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
#        cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
'''