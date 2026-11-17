import sys
import time
import imageio
from picamera2 import Picamera2
from libcamera import Transform

name = sys.argv[1]        
shutter = int(sys.argv[2])  

fileName = "data/" + name + ".png"
camera = Picamera2()
config = camera.create_still_configuration(
    main={"size": (2560, 1440)},
    transform=Transform(hflip=1, vflip=1),              
    controls={"AeEnable": False, "ExposureTime": shutter, 
              "AnalogueGain": 1.0,                       
              "AwbEnable": False, "ColourGains": (1.5, 1.5)})  
camera.configure(config)
camera.start()
time.sleep(1)               
camera.capture_file(fileName)
meta = camera.capture_metadata()
camera.stop()
print("выдержка %d мкс, усиление %.2f" % (meta["ExposureTime"], meta["AnalogueGain"]))

photo = imageio.imread(fileName)[:, :, :3]
top = photo.max(axis=(0, 1))
print("%s: R=%d G=%d B=%d" % (name, top[0], top[1], top[2]))
if top.max() == 255:
    print("Пересвет! Уменьшите выдержку")
