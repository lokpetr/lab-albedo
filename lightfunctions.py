import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import ndimage
from cycler import cycler


# Где спектр на снимке 2560x1440: строки — вдоль спектра (синий сверху),
# столбцы — поперёк. Координаты подобраны для установки, на которой сняты
# примеры данных; для своих снимков их нужно проверить и при необходимости поправить.
ROWS = slice(430, 870)
COLS = slice(1190, 1290)
BACKGROUND_COLS = slice(1130, 1350)

# Камера стоит чуть повёрнутой относительно щели, и линии спектра на снимке
# наклонены. Поворачиваем участок со спектром обратно на этот угол, градусы.
ANGLE = 5

# True — если на снимке красный край сверху, а синий снизу: спектр переворачивается,
# чтобы длина волны росла с номером пикселя. ROWS и COLS — как на исходном снимке.
FLIP = False


def straighten(photo):
    top, bottom = ROWS.start - 60, ROWS.stop + 60
    left, right = BACKGROUND_COLS.start - 60, BACKGROUND_COLS.stop + 60
    photo = photo[:, :, 0:3].astype(float)
    # поворот вокруг центра участка, то есть вокруг середины спектра
    photo[top:bottom, left:right] = ndimage.rotate(photo[top:bottom, left:right], ANGLE, reshape=False, order=1)
    return photo


def readIntensity(photoName, plotName, lamp, surface):
    photo = straighten(imageio.imread(photoName))
    background = photo[ROWS, BACKGROUND_COLS, 0:3].swapaxes(0, 1)

    cut = photo[ROWS, COLS, 0:3].swapaxes(0, 1)
    rgb = np.mean(cut, axis=(0))
    if FLIP:
        rgb = rgb[::-1]
        background = background[:, ::-1]
    luma = 0.2989 * rgb[:, 0] + 0.5866 * rgb[:, 1] + 0.1144 * rgb[:, 2]

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    fig = plt.figure(figsize=(10, 5), dpi=200)

    plt.title('Интенсивность отражённого излучения\n' + '{} / {}'.format(lamp, surface))
    plt.xlabel('Относительный номер пикселя')
    plt.ylabel('Яркость')

    plt.plot(rgb, label=['r', 'g', 'b'], linewidth=2)
    plt.plot(luma, 'w', label='I', linewidth=2)
    plt.legend()

    # фон — сам снимок спектра; яркость растянута только для наглядности, на данные не влияет
    background = background.astype(float)
    background = (background / max(background.max(), 1)) ** 0.6
    plt.imshow(background, origin='lower', aspect='auto', extent=[0, len(luma), 0, 1.15 * max(rgb.max(), 1)])

    plt.savefig(plotName)

    return luma
