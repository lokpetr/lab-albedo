import subprocess
import sys
import imageio

name = sys.argv[1]          # имя снимка без .png
shutter = sys.argv[2]       # выдержка, мкс

fileName = "data/" + name + ".png"
subprocess.run(["raspistill", "-n", "-t", "1000", "-w", "2560", "-h", "1440", "-rot", "180",
                "-ss", shutter, "-ISO", "100", "-awb", "off", "-awbg", "1.5,1.5",
                "-e", "png", "-o", fileName], check=True)

photo = imageio.imread(fileName)[:, :, :3]
top = photo.max(axis=(0, 1))
print("%s: R=%d G=%d B=%d" % (name, top[0], top[1], top[2]))
if top.max() == 255:
    print("Пересвет! Уменьшите выдержку")