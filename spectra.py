from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

ROWS = (320, 685)
COLS = (950, 990)
BG = (1130, 1180)


def load_img(path):
    with Image.open(path) as img:
        return np.asarray(img.convert('RGB'), dtype=np.float32)


def get_rgb(img, rows=ROWS, cols=COLS, bg=BG):
    if isinstance(img, (str, Path)):
        img = load_img(img)
    h, w = img.shape[:2]
    y1, y2 = rows
    x1, x2 = cols
    b1, b2 = bg
    if not (0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w and 0 <= b1 < b2 <= w):
        raise ValueError('Область спектра не помещается в изображение')
    rgb = img[y1:y2, x1:x2, :3].mean(axis=1)
    base = np.median(img[y1:y2, b1:b2, :3], axis=1)
    return np.maximum(rgb-base, 0)


def intensity(path, plot=None, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rgb = get_rgb(path)
    if plot is not None:
        y = np.arange(*ROWS)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, color in enumerate(('red', 'green', 'blue')):
            ax.plot(y, rgb[:, i], c=color, label=('R', 'G', 'B')[i])
        ax.set(xlabel='Строка, пиксели', ylabel='Яркость, усл. ед.', title=title)
        ax.legend()
        fig.tight_layout()
        Path(plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot, dpi=180)
        plt.close(fig)
    return rgb.sum(axis=1)


def smooth(rgb, sigma=2):
    return gaussian_filter1d(rgb, sigma, axis=0)


def find_cols(img):
    h, w = img.shape[:2]
    y1, y2 = int(h*.22), int(h*.65)
    x1, x2 = int(w*.22), int(w*.78)
    avg = img[y1:y2, x1:x2, :3].mean(axis=(0, 2))
    avg = gaussian_filter1d(avg, 9)
    x = x1+int(np.argmax(avg))
    return max(0, x-17), min(w, x+18)


def auto_rgb(img, cols):
    h, w = img.shape[:2]
    x1, x2 = cols
    rgb = img[:, x1:x2, :3].mean(axis=1)
    b1 = min(w-32, x2+max(55, int(.08*w)))
    if b1 <= x2:
        b1 = max(0, x1-max(100, int(.08*w))-32)
    b2 = min(w, b1+32)
    base = np.median(img[:, b1:b2, :3], axis=1)
    return np.maximum(rgb-base, 0)
