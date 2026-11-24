import sys
import csv
import subprocess
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Запуск: python3 lightMeasure.py <имя> <выдержка_мкс>')
    name = sys.argv[1]
    if not name or Path(name).name != name or name.endswith('.png'):
        raise SystemExit('Укажите имя без папок и расширения .png')
    try:
        exp = int(sys.argv[2])
    except ValueError:
        raise SystemExit('Выдержка должна быть числом в микросекундах')
    if not 1 <= exp <= 6000000:
        raise SystemExit('Неподходящая выдержка')

    folder = Path('data')
    folder.mkdir(exist_ok=True)
    path = folder/(name+'.png')
    cmd = ['raspistill', '-n', '-t', '1000', '-w', '2560', '-h', '1440',
           '-rot', '180', '-ss', str(exp), '-ISO', '100', '-awb', 'off',
           '-awbg', '1.5,1.5', '-e', 'png', '-o', str(path)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit('На Raspberry Pi не найдена программа raspistill')

    with Image.open(path) as img:
        rgb = np.asarray(img.convert('RGB'))
    peak = rgb.max(axis=(0, 1))
    print('%s: R=%d G=%d B=%d' % (name, peak[0], peak[1], peak[2]))
    if np.any(peak == 255):
        print('Пересвет, уменьшите выдержку')
    elif max(peak) < 150:
        print('Можно увеличить выдержку, сигнал слабый')
    elif max(peak) <= 220:
        print('Выдержка подходит, пересвета нет')
    else:
        print('Пересвета нет, но лучше немного уменьшить выдержку')

    log = folder/'nastroiki.csv'
    new = not log.exists()
    with log.open('a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['дата', 'файл', 'выдержка_мкс', 'ISO', 'баланс_белого', 'R', 'G', 'B'])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M'), str(path), exp,
                         100, '1.5,1.5', *map(int, peak)])
    print('Сохранено:', path)


if __name__ == '__main__':
    main()
