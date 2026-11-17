import sys
import time
from picamera2 import Picamera2
from libcamera import Transform

name = sys.argv[1]          # имя снимка без .png
shutter = int(sys.argv[2])  # выдержка, мкс

fileName = "data/" + name + ".png"
camera = Picamera2()
config = camera.create_still_configuration(
    main={"size": (2560, 1440)},
    transform=Transform(hflip=1, vflip=1),              # поворот на 180°
    controls={"AeEnable": False, "ExposureTime": shutter,  # выдержка, мкс
              "AnalogueGain": 1.0,                       # усиление, как ISO 100
              "AwbEnable": False, "ColourGains": (1.5, 1.5)})  # баланс белого
camera.configure(config)
camera.start()
time.sleep(1)               # дать камере применить настройки

request = camera.capture_request()   # один кадр: и файл, и массив, и настройки
request.save("main", fileName)
photo = request.make_array("main")   # массив (строки, столбцы, R G B)
meta = request.get_metadata()
request.release()
camera.stop()

print("выдержка %d мкс, усиление %.2f" % (meta["ExposureTime"], meta["AnalogueGain"]))
top = photo.max(axis=(0, 1))
print("%s: R=%d G=%d B=%d" % (name, top[0], top[1], top[2]))
if top.max() == 255:
    print("Пересвет! Уменьшите выдержку")
