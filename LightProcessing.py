
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.signal import find_peaks
import lightFunctions as light

dataPath = "light/data/"
plotsPath = "light/plots/"
sheets = ["white", "red", "yellow", "green", "blue"]
colors = ["w", "r", "gold", "g", "b"]


def readRGB(name, g=None):
    """Средние R, G, B по строкам спектра; g — пересчёт пикселей в свет."""
    photo = light.straighten(imageio.imread(dataPath + name))
    cut = photo[light.ROWS, light.COLS, 0:3]
    if g is not None:
        cut = g(cut)
    return np.mean(cut, axis=1)



mercury = light.readIntensity(dataPath + "white-mercury.png", plotsPath + "mercury.png",
                              "ртутная лампа", "белый лист")
red, green, blue = readRGB("white-mercury.png").T

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

intensity = [light.readIntensity(dataPath + s + "-tungsten.png", plotsPath + s + "-tungsten.png",
                                 "лампа накаливания", s) for s in sheets]

plt.figure(figsize=(10, 5))
plt.gca().set_facecolor("lightgrey")
for I, s, c in zip(intensity, sheets, colors):
    plt.plot(wavelength, I, color=c, label=s)
plt.xlabel("Длина волны, нм")
plt.ylabel("Яркость")
plt.legend()
plt.savefig(plotsPath + "intensities.png")

whiteSafe = np.maximum(intensity[0], 1)
plt.figure(figsize=(10, 5))
plt.gca().set_facecolor("lightgrey")
for I, s, c in zip(intensity, sheets, colors):
    plt.plot(wavelength, I / whiteSafe, color=c, label=s)
plt.xlim(430, 650)
plt.ylim(0, 1.1)
plt.xlabel("Длина волны, нм")
plt.ylabel("Альбедо")
plt.legend()
plt.savefig(plotsPath + "albedos.png")

def X(v):
    x = np.asarray(v, dtype=float) / 255
    return np.stack([x, x**2, x**3], axis=-1)

lin = [readRGB("lin-%d.png" % t).ravel() for t in (12500, 25000, 50000, 100000)]
v1, v2 = np.concatenate(lin[:-1]), np.concatenate(lin[1:])
ok = (v1 > 1) & (v2 < 250)
A = np.vstack([X(v2[ok]) - 2 * X(v1[ok]), 100 * X(128)])
b = np.append(np.zeros(ok.sum()), 100)
coef = np.linalg.lstsq(A, b, rcond=None)[0]
g = lambda v: X(v) @ coef

weights = [0.2989, 0.5866, 0.1144]
linear = [readRGB(s + "-tungsten.png", g) @ weights for s in sheets]

plt.figure(figsize=(10, 5))
plt.gca().set_facecolor("lightgrey")
for I, L, s, c in zip(intensity[1:], linear[1:], sheets[1:], colors[1:]):
    plt.plot(wavelength, I / whiteSafe, "--", color=c, label=s + " по PNG")
    plt.plot(wavelength, L / np.maximum(linear[0], g(1)), color=c, label=s + " после пересчёта")
plt.xlim(430, 650)
plt.ylim(0, 1.1)
plt.xlabel("Длина волны, нм")
plt.ylabel("Альбедо")
plt.legend()
plt.savefig(plotsPath + "albedos_linear.png")