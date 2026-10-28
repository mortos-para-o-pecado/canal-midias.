# Junta tudo numa cena: (rosto animado, se houver) + chama/fumaça/poeira + câmera em profundidade.
# Uso: python3 vida.py pintura.jpg prof.png saida.mp4 --cena latour_vela [--rosto rosto.mp4] [--seg 5] [--larg 720]
import argparse, json, math, subprocess, sys, os
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from efeitos import Chama, Fumaca, Poeira, Clarao, aplica_campos
from camada import Lama, prepara, camera_camadas

CENAS = {   # coordenadas na imagem original
    'latour_vela': dict(camera='orbita', forca=2.0,
                        chama=dict(caixa=(645, 295, 685, 412), base_y=410, ponta_y=300, amp=3.5),
                        fumaca=dict(caixa=(630, 90, 700, 300), amp=5.0),
                        clarao=dict(centro=(665, 345), raio=430, forca=0.045),   # coordenadas na pintura em 1004x1280
                        camadas=dict(limiar=0.15)),
    # paulo_valentin: a profundidade não separa a cabeça da parede; fica com a câmera antiga (sem camadas)
    'paulo_valentin': dict(camera='orbita', forca=1.0, poeira=dict(n=60, caixa=(250, 0, 1073, 700))),
    'lutero_cranach': dict(camera='dolly', forca=1.0, camadas=dict(limiar=0.25)),
    'paulo_rembrandt': dict(camera='orbita', forca=1.0, poeira=dict(n=40, caixa=(0, 0, 685, 500)), camadas=dict(limiar=0.15)),
}

def camera_quadro(q, d, t, modo, W, forca=1.0):
    H0, W0 = q.shape[:2]; H = int(round(W * H0 / W0 / 2)) * 2; M = 1.12; Wb, Hb = int(W * M), int(H * M)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32); ox, oy = (Wb - W) / 2, (Hb - H) / 2
    X, Y = xs + ox, ys + oy; dd = cv2.remap(d, X, Y, cv2.INTER_LINEAR); cx, cy = Wb / 2, Hb / 2; k = forca
    if modo == 'dolly': zoom, dx, dy = 1 + 0.06 * k * t, 0.0, 0.0
    elif modo == 'orbita': zoom, dx, dy = 1 + 0.02 * k * t, (t - 0.5) * 0.05 * k * W, 0.0
    else: zoom, dx, dy = 1 + 0.03 * k * t, 0.0, (0.5 - t) * 0.04 * k * H
    z = 1 + (zoom - 1) * (0.25 + 1.5 * dd)
    sx = (cx + (X - cx) / z - dx * (0.15 + 1.2 * dd)).astype(np.float32)
    sy = (cy + (Y - cy) / z - dy * (0.15 + 1.2 * dd)).astype(np.float32)
    src = cv2.resize(q, (Wb, Hb), interpolation=cv2.INTER_AREA)
    return cv2.remap(src, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

def prepara_prof(prof, W, shape):
    H0, W0 = shape[:2]; H = int(round(W * H0 / W0 / 2)) * 2; M = 1.12; Wb, Hb = int(W * M), int(H * M)
    d = cv2.resize(cv2.imread(prof, cv2.IMREAD_GRAYSCALE), (Wb, Hb), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    d = cv2.GaussianBlur(d, (0, 0), 3)
    return cv2.GaussianBlur(np.maximum(d, cv2.dilate(d, np.ones((9, 9), np.uint8)) * 0.98), (0, 0), 2), H

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('img'); p.add_argument('prof'); p.add_argument('saida'); p.add_argument('--cena', required=True)
    p.add_argument('--rosto', default=''); p.add_argument('--seg', type=float, default=5); p.add_argument('--fps', type=int, default=24)
    p.add_argument('--larg', type=int, default=720); p.add_argument('--sem-camera', action='store_true')
    p.add_argument('--forca', type=float, default=None); p.add_argument('--sem-camadas', action='store_true')
    a = p.parse_args(); c = CENAS[a.cena]
    img = cv2.cvtColor(cv2.imread(a.img), cv2.COLOR_BGR2RGB)
    rosto = None
    if a.rosto:   # quadros do rosto animado (mesmo tamanho da pintura)
        cap = cv2.VideoCapture(a.rosto); rosto = []
        while True:
            ok, f = cap.read()
            if not ok: break
            rosto.append(cv2.cvtColor(cv2.resize(f, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
    N = len(rosto) if rosto else int(a.seg * a.fps)
    efeitos = []
    if 'chama' in c: efeitos.append(Chama(img.shape, **c['chama']))
    if 'fumaca' in c: efeitos.append(Fumaca(img.shape, **c['fumaca']))
    if 'poeira' in c: efeitos.append(Poeira(img, **c['poeira']))
    if 'clarao' in c: efeitos.append(Clarao(img, **c['clarao']))
    d, H = prepara_prof(a.prof, a.larg, img.shape)
    forca = a.forca if a.forca is not None else c.get('forca', 1.0)
    cam = None
    if 'camadas' in c and not a.sem_camadas and not a.sem_camera:   # figura e fundo separados, parede preenchida atrás
        h0, w0 = img.shape[:2]
        pr0 = cv2.resize(cv2.imread(a.prof, cv2.IMREAD_GRAYSCALE), (w0, h0), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        soltos = [c[k]['caixa'] for k in ('chama', 'fumaca') if k in c]
        cam = prepara(img, pr0, Lama(), limiar=c['camadas'].get('limiar'), soltos=soltos,
                      cache=os.path.splitext(a.saida)[0] + '_fundo.png')
    W = a.larg if not a.sem_camera else a.larg
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{a.larg}x{H}', '-r', str(a.fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.saida]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        t = i / a.fps
        q = rosto[i] if rosto else img
        q = aplica_campos(q, efeitos, t)
        tc = 0.5 - 0.5 * math.cos(math.pi * i / (N - 1))
        if a.sem_camera: f = cv2.resize(q, (a.larg, H), interpolation=cv2.INTER_AREA)
        elif cam is not None: f = camera_camadas(q, cam, tc, c['camera'], a.larg, forca)
        else: f = camera_quadro(q, d, tc, c['camera'], a.larg, forca)
        pr.stdin.write(np.ascontiguousarray(f).tobytes())
    pr.stdin.close(); pr.wait(); print('ok', a.saida, N, 'quadros')
