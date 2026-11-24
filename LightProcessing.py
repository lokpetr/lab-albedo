
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import lightFunctions as light

mercuryFile = "рт1(new)(100000).png"                     
sheetFiles = ["white-tungsten.png", "red-tungsten.png", "yellow-tungsten.png",
              "green-tungsten.png", "blue-tungsten.png"]  
linFiles = {12500: "lin-12500.png", 25000: "lin-25000.png",
            50000: "lin-50000.png", 100000: "lin-100000.png"}   

dataPath = "data/"
plotsPath = "plots/"
os.makedirs(plotsPath, exist_ok=True)
sheets = ["white", "red", "yellow", "green", "blue"]
names = ["белый", "красный", "жёлтый", "зелёный", "синий"]
colors = ["w", "r", "gold", "g", "b"]


def readRGB(fileName, g=None):
    """Средние R, G, B по строкам спектра; g — пересчёт пикселей в свет."""
    photo = light.straighten(imageio.imread(dataPath + fileName))
    cut = photo[light.ROWS, light.COLS, 0:3]
    if g is not None:
        cut = g(cut)
    rgb = np.mean(cut, axis=1)
    return rgb[::-1] if light.FLIP else rgb   

mercury = light.readIntensity(dataPath + mercuryFile, plotsPath + "mercury.png",
                              "ртутная лампа", "белый лист")
red, green, blue = readRGB(mercuryFile).T

peaks, _ = find_peaks(blue + green, prominence=20)
pixels = [peaks[0], peaks[1], peaks[2], np.argmax(red)]  
lines = [435.8, 487.7, 546.1, 611.6]
print("пиксели пиков:", np.array(pixels))

fit = np.polyfit(pixels, lines, 1)
wavelength = np.polyval(fit, np.arange(len(mercury)))
print("λ = %.3f·n + %.1f нм" % (fit[0], fit[1]))

plt.figure()
plt.plot(pixels, lines, "o", label="линии ртути")
plt.plot(wavelength, label="λ = %.3f·n + %.1f нм" % (fit[0], fit[1]))
plt.xlabel("Номер пикселя")
plt.ylabel("Длина волны, нм")
plt.legend()
plt.savefig(plotsPath + "calibration.png")
intensity = [light.readIntensity(dataPath + f, plotsPath + s + ".png", "лампа накаливания", n + " лист")
             for f, s, n in zip(sheetFiles, sheets, names)]

plt.figure(figsize=(10, 5))
plt.gca().set_facecolor("lightgrey")
for I, n, c in zip(intensity, names, colors):
    plt.plot(wavelength, I, color=c, label=n + " лист")
plt.xlabel("Длина волны, нм")
plt.ylabel("Яркость")
plt.legend()
plt.savefig(plotsPath + "intensities.png")

whiteSafe = np.maximum(intensity[0], 1)
plt.figure(figsize=(10, 5))
plt.gca().set_facecolor("lightgrey")
for I, n, c in zip(intensity, names, colors):
    plt.plot(wavelength, I / whiteSafe, color=c, label=n + " лист")
plt.xlim(430, 650)
plt.ylim(0, 1.1)
plt.xlabel("Длина волны, нм")
plt.ylabel("Альбедо")
plt.legend()
plt.savefig(plotsPath + "albedos.png")

def X(v):
    x = np.asarray(v, dtype=float) / 255
    return np.stack([x, x**2, x**3], axis=-1)

times = sorted(linFiles)
lin = [readRGB(linFiles[t]).ravel() for t in times]
rows = []
for t1, t2, v1, v2 in zip(times, times[1:], lin, lin[1:]):
    ok = (v1 > 1) & (v2 < 250)                  
    rows.append(X(v2[ok]) - t2 / t1 * X(v1[ok]))
A = np.vstack(rows + [100 * X(128)])
b = np.append(np.zeros(len(A) - 1), 100)
coef = np.linalg.lstsq(A, b, rcond=None)[0]
g = lambda v: X(v) @ coef

weights = [0.2989, 0.5866, 0.1144]
linear = [readRGB(f, g) @ weights for f in sheetFiles]

plt.figure(figsize=(10, 5))
plt.gca().set_facecolor("lightgrey")
for I, L, n, c in zip(intensity[1:], linear[1:], names[1:], colors[1:]):
    plt.plot(wavelength, I / whiteSafe, "--", color=c, label=n + ": по PNG")
    plt.plot(wavelength, L / np.maximum(linear[0], g(1)), color=c, label=n + ": после пересчёта")
plt.xlim(430, 650)
plt.ylim(0, 1.1)
plt.xlabel("Длина волны, нм")
plt.ylabel("Альбедо")
plt.legend()
plt.savefig(plotsPath + "albedos_linear.png")
