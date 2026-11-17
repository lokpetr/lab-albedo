# lightProcessing.py — калибровка (п2.4), интенсивности и альбедо (п2.5)
# Рядом должны лежать lightFunctions.py и папка data со снимками с Raspberry Pi.
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import lightFunctions as light

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
    return rgb[::-1] if light.FLIP else rgb    # красный сверху → переворачиваем


# ---------- п2.4 Калибровка по длине волны
mercury = light.readIntensity(dataPath + "white-mercury.png", plotsPath + "mercury.png",
                              "ртутная лампа", "белый лист")
red, green, blue = readRGB("white-mercury.png").T

# 435,8, 487,7 и 546,1 нм — первые три пика синего+зелёного,
# 611,6 нм — самый яркий пик красного канала (+1 балл)
peaks, _ = find_peaks(blue + green, prominence=20)
pixels = [peaks[0], peaks[1], peaks[2], np.argmax(red)]   # или впишите свои вручную
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

# ---------- п2.5 Интенсивности
intensity = [light.readIntensity(dataPath + s + "-tungsten.png", plotsPath + s + "-tungsten.png",
                                 "лампа накаливания", n + " лист") for s, n in zip(sheets, names)]

plt.figure(figsize=(10, 5))
plt.gca().set_facecolor("lightgrey")
for I, n, c in zip(intensity, names, colors):
    plt.plot(wavelength, I, color=c, label=n + " лист")
plt.xlabel("Длина волны, нм")
plt.ylabel("Яркость")
plt.legend()
plt.savefig(plotsPath + "intensities.png")

# ---------- п2.5 Альбедо
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

# ---------- п2.5 Пересчёт по тоновой кривой (+1 балл)
# g(v) = a·x + b·x² + c·x³, x = v/255. Свет на снимке с выдержкой t2 в t2/t1 раз
# больше, чем на t1: g(v2) − (t2/t1)·g(v1) = 0 во всех точках, и ещё g(128) = 1.
def X(v):
    x = np.asarray(v, dtype=float) / 255
    return np.stack([x, x**2, x**3], axis=-1)

times = sorted(int(f[4:-4]) for f in os.listdir(dataPath) if f.startswith("lin-"))
lin = [readRGB("lin-%d.png" % t).ravel() for t in times]
rows = []
for t1, t2, v1, v2 in zip(times, times[1:], lin, lin[1:]):
    ok = (v1 > 1) & (v2 < 250)                   # без нуля и без пересвета
    rows.append(X(v2[ok]) - t2 / t1 * X(v1[ok]))
A = np.vstack(rows + [100 * X(128)])
b = np.append(np.zeros(len(A) - 1), 100)
coef = np.linalg.lstsq(A, b, rcond=None)[0]
g = lambda v: X(v) @ coef

weights = [0.2989, 0.5866, 0.1144]
linear = [readRGB(s + "-tungsten.png", g) @ weights for s in sheets]

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
