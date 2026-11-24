import sys
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter, peak_widths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectra import get_rgb, smooth, ROWS, COLS, BG

SHEETS = {
    'white': ('white1(100000).png', '#303030'),
    'red': ('red(100000).png', '#c62828'),
    'yellow': ('yellow(100000).png', '#c99b00'),
    'green': ('green(100000).png', '#2c8b57'),
    'blue': ('blue(100000).png', '#2c60c9')
}
LABELS = {'white':'Белый', 'red':'Красный', 'yellow':'Жёлтый', 'green':'Зелёный', 'blue':'Синий'}
HG = ['рт1(100000).png', 'рт1(new)(100000).png', 'рт2(100000).png']
EXPS = {25000:'white3(25000).png', 50000:'white2(50000).png',
       100000:'white1(100000).png', 200000:'white4(200000).png'}
LINES = [(435.8, 2, 360, 400), (487.7, 1, 421, 458),
         (546.1, 1, 495, 522), (611.6, 0, 580, 620)]


def peak_at(vals, channel, start, end):
    y = np.arange(*ROWS)
    values = savgol_filter(vals[:, channel], 11, 2)
    peaks, _ = find_peaks(values, prominence=1, distance=8)
    peaks = [i for i in peaks if start <= y[i] <= end]
    if peaks:
        return int(y[max(peaks, key=lambda i: values[i])])
    region = np.where((y >= start) & (y <= end))[0]
    return int(y[region[np.argmax(values[region])]])


def calibrate(data):
    coords = []
    for file in HG:
        coords.append([peak_at(data[file], ch, a, b) for _, ch, a, b in LINES])
    x = np.mean(coords, axis=0)
    nm = np.array([z[0] for z in LINES])
    a, b = np.polyfit(x, nm, 1)
    err = float(np.sqrt(np.mean((a*x + b - nm)**2)))
    z = savgol_filter(data[HG[0]][:, 1], 11, 2)
    idx_hg = peak_at(data[HG[0]], 1, 495, 522) - ROWS[0]
    width = float(peak_widths(z, [idx_hg], rel_height=.5)[0][0] * abs(a))
    return float(a), float(b), x, nm, err, width


def tone(data):
    t = np.array(sorted(EXPS), dtype=float)
    vals = np.stack([data[EXPS[int(values)]] for values in t])
    y = np.arange(*ROWS)
    region = (y >= 380) & (y <= 640)
    alpha = []
    log_t = np.log(t)
    for i in range(3):
        values = vals[:, :, i]
        mask = region & np.all((values >= 3) & (values < 225), axis=0)
        if np.count_nonzero(mask) < 15:
            alpha.append(1.0)
            continue
        z = np.log(values[:,mask])
        num = np.sum((log_t[:,None] - log_t.mean())*(z-z.mean(axis=0)), axis=0)
        den = np.sum((log_t-log_t.mean())**2)
        alpha.append(float(np.median(num/den)))
    alpha = np.array(alpha)
    alpha = np.where((alpha > .2)&(alpha < 2.5), alpha, 1)
    return alpha, t, vals


def plot_all(folder, curves, wave, refl, fit, hg_rgb, series, corr):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5,5))
    for name,(file,color) in SHEETS.items():
        ax.plot(wave, curves[name].sum(axis=1), label=LABELS[name], color=color, lw=1.9)
    ax.set(xlim=(430,660),xlabel='Длина волны, нм',ylabel='Интенсивность, усл. ед.')
    ax.grid(alpha=.3);ax.legend(ncol=5,frameon=False)
    fig.tight_layout();fig.savefig(folder/'intensities.png',dpi=180);plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5,5))
    for name,(file,color) in SHEETS.items():
        ax.plot(wave, refl[name], label=LABELS[name],color=color,lw=1.9)
    ax.set(xlim=(430,660),ylim=(0,1.15),xlabel='Длина волны, нм',
           ylabel='Относительное альбедо')
    ax.grid(alpha=.3);ax.legend(ncol=5,frameon=False)
    fig.tight_layout();fig.savefig(folder/'albedos.png',dpi=180);plt.close(fig)

    a,b,x,nm,err,width = fit
    fig,ax=plt.subplots(figsize=(8.5,4.6))
    ax.scatter(x,nm,color='black',label='Опорные максимумы')
    xx=np.linspace(min(x)-20,max(x)+20,100)
    ax.plot(xx,a*xx+b,color='#666666',label='Аппроксимация')
    for px,wl in zip(x,nm):
        ax.annotate(f'{wl:.1f} нм',(px,wl),xytext=(6,7),textcoords='offset points')
    ax.set(xlabel='Строка снимка',ylabel='Длина волны, нм')
    ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(folder/'calibration.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.6))
    yy=np.arange(*ROWS)
    for i,c in enumerate(('red','green','blue')):
        ax.plot(yy,hg_rgb[:,i],color=c,label=('R','G','B')[i])
    for px in x:
        ax.axvline(px,ls='--',c='gray',alpha=.5)
    ax.set(xlim=(340,635),xlabel='Строка снимка',ylabel='Яркость, усл. ед.')
    ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(folder/'mercury.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,4.6))
    alpha,t,vals = series
    for i,c in enumerate(('red','green','blue')):
        idx=np.argmin(np.abs(wave - (620,530,450)[i]))
        ax.plot(t/1000,vals[:,idx,i],'-o',color=c,label=('R','G','B')[i])
    ax.set(xlabel='Выдержка, мс',ylabel='Яркость, усл. ед.')
    ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(folder/'ton.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.5,5))
    for name,(_,color) in SHEETS.items():
        ax.plot(wave,corr[name],label=LABELS[name],color=color,lw=1.9)
    ax.set(xlim=(430,660),ylim=(0,1.15),xlabel='Длина волны, нм',
           ylabel='Альбедо после поправки')
    ax.grid(alpha=.3);ax.legend(ncol=5,frameon=False)
    fig.tight_layout();fig.savefig(folder/'albedos_ton.png',dpi=180);plt.close(fig)


def main():
    folder = Path(sys.argv[1]) if len(sys.argv)>1 else Path('data')
    root = Path(sys.argv[2]) if len(sys.argv)>2 else Path('.')
    out = root/'data'
    out.mkdir(parents=True,exist_ok=True)
    items = list(SHEETS.values())
    names = set([file for file,_ in items] + HG + list(EXPS.values()))
    data = {}
    for file in names:
        path = folder/file
        if not path.exists():
            raise SystemExit('Нет файла: '+str(path))
        data[file] = get_rgb(path)

    fit = calibrate(data)
    a,b,x,nm,err,width = fit
    y = np.arange(*ROWS)
    wave = a*y+b
    curves = {name:smooth(data[file]) for name,(file,_) in SHEETS.items()}
    white = curves['white']
    white_sum = white.sum(axis=1)
    valid = (wave >= 430)&(wave <= 660)&(white_sum >= 12)
    alpha,t,vals = tone(data)
    power = np.clip(1/alpha,.65,2.5)
    white2 = np.sum((white/255)**power,axis=1)
    refls = {}
    corr_refl = {}
    for name in SHEETS:
        values = curves[name]
        ratio = np.divide(values.sum(axis=1),white_sum,out=np.full(len(y),np.nan),where=white_sum>0)
        values2 = np.sum((values/255)**power,axis=1)
        ratio2 = np.divide(values2,white2,out=np.full(len(y),np.nan),where=white2>0)
        ratio[~valid] = np.nan
        ratio2[~valid] = np.nan
        refls[name], corr_refl[name] = ratio,ratio2

    plot_all(root/'plots',curves,wave,refls,fit,data[HG[0]],(alpha,t,vals),corr_refl)
    with (out/'reflectance_curves.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['Строка','Длина волны, нм','Белый сигнал']+
                     [f'{LABELS[name]} альбедо' for name in SHEETS] + [f'{LABELS[name]} с поправкой' for name in SHEETS])
        for i in range(len(y)):
            writer.writerow([int(y[i]),float(wave[i]),float(white_sum[i])] +
                         [float(refls[name][i]) for name in SHEETS] +
                         [float(corr_refl[name][i]) for name in SHEETS])
    with (out/'signal.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['Длина волны, нм']+[LABELS[name] for name in SHEETS])
        for i in range(len(y)):
            writer.writerow([float(wave[i])]+[float(curves[name][i].sum()) for name in SHEETS])
    info = {'a':a,'b':b,'razmer':[2048,1152], 'stroki':list(ROWS),
            'stolbcy':list(COLS),'fon':list(BG),
            'tochki':x.tolist(),'volny':nm.tolist(),
            'srednee_otklonenie_nm':err,'shirina_zelenoi_linii_nm':width,
            'stepeni':power.tolist()}
    (out/'kalibrovka.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Калибровка: %.4f * y + %.2f нм' % (a,b))
    print('Расхождение по опорным точкам: %.2f нм' % err)
    print('Ширина зелёной линии: %.1f нм' % width)
    print('Поправка по выдержкам:',power.round(3).tolist())
    print('Графики:',(root/'plots').resolve())


if __name__ == '__main__':
    main()
