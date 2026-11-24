from spectra import load_img, get_rgb, intensity, smooth, find_cols, auto_rgb
from spectra import ROWS as ROW_RANGE, COLS as COL_RANGE, BG as BG_RANGE

ROWS = slice(*ROW_RANGE)
COLS = slice(*COL_RANGE)
BACKGROUND_COLS = slice(*BG_RANGE)


def readIntensity(path, plot=None, lamp='', sheet=''):
    return intensity(path, plot, (lamp+' '+sheet).strip())
