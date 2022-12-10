Steps:

- Find the LndscapeMountains sln and open in visual studio
- Hit F5 and that should open the simulator

To update the airsim code (ie, programmatically change things);
- Update the airsim repo
  - update a python file to connect API to simulator and have it automatically try and detect things
- Run build.cmd again??
- Copy over the Plugins folder back into the landscapemountains location??

Anaconda:
pip install msgpack-rpc-python 
pip install airsim

pip install pyrovision
pip install torchvision
pip install pillow
pip install gradio
pip install onnxruntime

 Images location: C:\Users\molle\AppData\Local\Temp\airsim_drone

To run:
1. Start the simulation by running the landscape solution
2. Press play on simulation, then 'possess' so viewing experience is good
3. In the anaconda cmd line, once packages have been installed, run the python script needed.

Run python script: 
 cd C:\Users\molle\OneDrive\Documents\usc\Fall2022\CSCI513\Group\Unreal\AirSim\PythonClient\multirotor\
 python .\hello_drone.py



 Todo:
- add logic to keep set distance from ground    
- *if fire: 
  - if wildfire > 0.9 stop and send signal/print
  - split image into 3
    - if in 1/3 -> easy
    - if in 2/3 -> middle + side
    - side + side -> pick one and go 30 degrees
    - if left/right: 30degrees
    - if middle + right/left: 15 degrees
  - moveTOwardsFire
- else:
  - default path moving