import sys
import csv
import re
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectra import load_img, get_rgb, smooth, find_cols, auto_rgb


def get_exp(path):
    m = re.search(r'\((\d+)\)\.png$',str(path))
    return int(m.group(1)) if m else None


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Запуск: python3 albedo_foto.py snimok.png belyi.png [kalibrovka.json]')

    file, ref = Path(sys.argv[1]), Path(sys.argv[2])
    cal = Path(sys.argv[3]) if len(sys.argv)>3 else ref.parent/'kalibrovka.json'
    a, b = None, None
    cfg = None
    if cal.exists():
        cfg = json.loads(cal.read_text(encoding='utf-8'))
        a = float(cfg['a'])
        b = float(cfg['b'])

    exp1,exp2 = get_exp(file),get_exp(ref)
    if exp1 and exp2 and exp1 != exp2:
        raise SystemExit('Выдержки разные. Для сравнения нужны снимки с одинаковой выдержкой')
    if not exp1 or not exp2:
        print('Не удалось проверить выдержки по названиям файлов')

    img = load_img(file)
    white = load_img(ref)
    if img.shape != white.shape:
        raise SystemExit('Размеры снимков не совпадают')
    h,w = white.shape[:2]

    if cfg and [w,h] == cfg['razmer']:
        rows = tuple(cfg['stroki'])
        cols = tuple(cfg['stolbcy'])
        bg = tuple(cfg['fon'])
        rgb = get_rgb(img,rows,cols,bg)
        ref_rgb = get_rgb(white,rows,cols,bg)
        yy = np.arange(*rows)
    else:
        cols = find_cols(white)
        rgb = auto_rgb(img,cols)
        ref_rgb = auto_rgb(white,cols)
        yy = np.arange(h)
        if cfg:
            print('Размер кадра изменился, шкала длин волн неприменима')
        a, b = None, None

    if np.any(img[:,cols[0]:cols[1],:3] >= 255):
        print('Внимание: на исследуемом спектре есть пересвеченные пиксели')
    if np.any(white[:,cols[0]:cols[1],:3] >= 255):
        print('Внимание: белый спектр пересвечен, результаты могут быть неверны')

    rgb = smooth(rgb)
    ref_rgb = smooth(ref_rgb)
    sig = rgb.sum(axis=1)
    ref_sig = ref_rgb.sum(axis=1)
    cutoff = max(12, float(ref_sig.max())*.07)
    valid = ref_sig >= cutoff
    if a is not None:
        x = a*yy+b
        valid &= (x>=430)&(x<=660)
        xlabel = 'Длина волны, нм'
    else:
        x = yy
        xlabel = 'Строка изображения, пиксели'

    alb = np.divide(sig,ref_sig,out=np.full(len(sig),np.nan),where=ref_sig>0)
    alb[~valid] = np.nan
    if np.count_nonzero(np.isfinite(alb)) < 20:
        raise SystemExit('Не удалось выделить достаточно яркий спектр')
    if np.nanmax(alb) > 1.02:
        print('Есть значения больше 1: проверьте освещение и положение листа')

    out = Path('rezultat_albedo')
    out.mkdir(exist_ok=True)
    name = file.stem.replace(' ','_')
    with (out/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow([xlabel,'сигнал листа','сигнал белого','относительное альбедо'])
        for i in range(len(x)):
            writer.writerow([float(x[i]),float(sig[i]),float(ref_sig[i]),float(alb[i])])

    fig,ax = plt.subplots(figsize=(9,5))
    ax.plot(x,alb,color='#242424',lw=1.8)
    ax.set(xlabel=xlabel,ylabel='Относительное альбедо',title=file.stem)
    ax.set_ylim(0,max(1.1,min(1.5,np.nanmax(alb)*1.1)))
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out/(name+'.png'),dpi=180)
    plt.close(fig)
    print('График:',out/(name+'.png'))
    print('Таблица:',out/(name+'.csv'))


if __name__=='__main__':
    main()
