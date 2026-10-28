# Câmera 2.5D em CAMADAS: separa a figura (perto) do fundo (longe) pela profundidade, preenche o fundo escondido atrás
# da figura com LaMa (Apache 2.0, ONNX do opencv_zoo) e move as duas camadas em velocidades diferentes.
# Assim a borda da figura não "estica como borracha": a figura desliza inteira sobre um fundo que existe atrás dela.
# Nada na figura é redesenhado; só a faixa de parede escondida atrás dela é preenchida.
import os, math
import numpy as np, cv2
os.environ.setdefault('ORT_LOGGING_LEVEL', '3')
import onnxruntime as ort

_AQUI = os.path.dirname(os.path.abspath(__file__))
LAMA = os.environ.get('LAMA_ONNX', os.path.join(_AQUI, '..', 'lama', 'lama.onnx'))

URL_LAMA = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/inpainting_lama/inpainting_lama_2025jan.onnx'
SHA_LAMA = '7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2'

def baixa_lama(caminho=LAMA):
    # modelo LaMa em ONNX do opencv_zoo (licença Apache 2.0), 92 MB; confere a impressão digital antes de usar
    import hashlib, urllib.request
    if not os.path.exists(caminho):
        os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
        urllib.request.urlretrieve(URL_LAMA, caminho + '.tmp'); os.replace(caminho + '.tmp', caminho)
    if hashlib.sha256(open(caminho, 'rb').read()).hexdigest() != SHA_LAMA: raise RuntimeError('lama.onnx diferente do esperado')
    return caminho

class Lama:
    def __init__(self, caminho=LAMA):
        baixa_lama(caminho)
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.s = ort.InferenceSession(caminho, so, providers=['CPUExecutionProvider'])
    def pedaco(self, rgb, mascara):
        im = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255
        m = (mascara > 0).astype(np.float32)[None, None]
        out = self.s.run(None, {'image': im, 'mask': m})[0][0].transpose(1, 2, 0)
        return np.clip(out, 0, 255).astype(np.uint8)
    def preenche(self, rgb, mascara, lado=512, passo=384):
        """Preenche só os pixels da máscara, em janelas de 512 px na resolução original (sem perder nitidez)."""
        img = rgb.copy(); falta = (mascara > 0).copy(); h, w = falta.shape
        ys = list(range(0, max(1, h - lado), passo)) + [max(0, h - lado)]
        xs = list(range(0, max(1, w - lado), passo)) + [max(0, w - lado)]
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                m = falta[y:y + lado, x:x + lado]
                if not m.any(): continue
                cr = img[y:y + lado, x:x + lado]; ph, pw = lado - cr.shape[0], lado - cr.shape[1]
                crp = cv2.copyMakeBorder(cr, 0, ph, 0, pw, cv2.BORDER_REFLECT); mp = np.pad(m, ((0, ph), (0, pw)))
                r = self.pedaco(crp, mp.astype(np.uint8))[:cr.shape[0], :cr.shape[1]]
                cr[m] = r[m]; falta[y:y + lado, x:x + lado] = False
        return img

def separa(prof, limiar=None, minimo=0.01):
    """Máscara da figura: profundidade acima do limiar (Otsu se não for dado), sem ilhas pequenas."""
    d8 = (np.clip(prof, 0, 1) * 255).astype(np.uint8)
    if limiar is None: limiar = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255
    m = (prof > limiar).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= minimo * m.size: keep[lab == i] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return keep, limiar

def prepara(img, prof, lama, limiar=None, faixa=0.06, cache=None, soltos=()):
    """Devolve: fundo preenchido, alfa suave da figura, profundidade do fundo e da figura (todas no tamanho da pintura).
    soltos = caixas (x0,y0,x1,y1) de coisas claras e soltas na frente da parede (chama, fumaça): a parede atrás delas é
    preenchida e elas viajam junto com a figura, recortadas quadro a quadro pela diferença de claridade."""
    h, w = img.shape[:2]
    figura, limiar = separa(prof, limiar)
    caixas = np.zeros((h, w), np.uint8)
    for (x0, y0, x1, y1) in soltos: caixas[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8] = 1
    k = max(3, int(faixa * w))                               # quanto a figura pode andar sobre o fundo
    borda = cv2.dilate(figura, np.ones((7, 7), np.uint8))   # a beirada da figura ainda tem cor misturada
    miolo = cv2.erode(figura, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    buraco = ((borda > 0) & ~(miolo > 0)) | ((caixas > 0) & ~(miolo > 0))   # faixa a preencher atrás da figura e das caixas
    if cache and os.path.exists(cache):
        fundo = cv2.cvtColor(cv2.imread(cache), cv2.COLOR_BGR2RGB)
    else:
        fundo = lama.preenche(img, buraco)
        if cache: cv2.imwrite(cache, cv2.cvtColor(fundo, cv2.COLOR_RGB2BGR))
    alfa = cv2.GaussianBlur(figura.astype(np.float32), (0, 0), 1.6)
    # profundidade do fundo: tira a figura e espalha a profundidade da parede para dentro do buraco
    p8 = (np.clip(prof, 0, 1) * 255).astype(np.uint8)
    pf = cv2.inpaint(cv2.resize(p8, (w // 4, h // 4)), cv2.resize(borda * 255, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST), 9, cv2.INPAINT_TELEA)
    prof_fundo = cv2.GaussianBlur(cv2.resize(pf, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255, (0, 0), 4)
    # profundidade da figura: a da própria figura, estendida para fora para a borda andar junto
    pfig = np.where(figura > 0, prof, 0).astype(np.float32)
    for _ in range(3): pfig = np.maximum(pfig, cv2.dilate(pfig, np.ones((15, 15), np.uint8)) * (pfig == 0))
    if caixas.any():   # o que está solto na frente da parede anda com a profundidade da figura logo abaixo (a vela)
        base = cv2.dilate(caixas, np.ones((61, 61), np.uint8)) & figura
        val = float(np.median(prof[base > 0])) if base.any() else float(prof[caixas > 0].max())
        pfig = np.where((caixas > 0) & (figura == 0), val, pfig)
    prof_fig = cv2.GaussianBlur(pfig, (0, 0), 3)
    Lf = cv2.cvtColor(fundo, cv2.COLOR_RGB2LAB).astype(np.float32)
    return dict(fundo=fundo, alfa=alfa, prof_fundo=prof_fundo, prof_fig=prof_fig, figura=figura, buraco=buraco, limiar=limiar,
                caixas=(caixas > 0) & (figura == 0), Lf=Lf)

def alfa_quadro(q, cam):
    """Alfa do quadro atual: a figura + o que estiver claro dentro das caixas soltas (a chama que balança, a fumaça)."""
    a = cam['alfa']
    if not cam['caixas'].any(): return a
    ys, xs = np.where(cam['caixas']); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    Lq = cv2.cvtColor(np.ascontiguousarray(q[y0:y1, x0:x1]), cv2.COLOR_RGB2LAB).astype(np.float32)
    dE = cv2.GaussianBlur(np.sqrt(((Lq - cam['Lf'][y0:y1, x0:x1]) ** 2).sum(-1)), (0, 0), 1.5)   # fumaça é mais ESCURA que a parede, chama mais clara
    ex = np.clip((dE - 4) / 10, 0, 1) * cam['caixas'][y0:y1, x0:x1]
    a = a.copy(); a[y0:y1, x0:x1] = np.maximum(a[y0:y1, x0:x1], cv2.GaussianBlur(ex, (0, 0), 1.2))
    return a

def _mapa(d, W, H0, W0, t, modo, forca):
    H = int(round(W * H0 / W0 / 2)) * 2; M = 1.12; Wb, Hb = int(W * M), int(H * M)
    db = cv2.resize(d, (Wb, Hb), interpolation=cv2.INTER_AREA)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32); ox, oy = (Wb - W) / 2, (Hb - H) / 2
    X, Y = xs + ox, ys + oy; dd = cv2.remap(db, X, Y, cv2.INTER_LINEAR); cx, cy = Wb / 2, Hb / 2; k = forca
    if modo == 'dolly': zoom, dx, dy = 1 + 0.06 * k * t, 0.0, 0.0
    elif modo == 'orbita': zoom, dx, dy = 1 + 0.02 * k * t, (t - 0.5) * 0.05 * k * W, 0.0
    else: zoom, dx, dy = 1 + 0.03 * k * t, 0.0, (0.5 - t) * 0.04 * k * H
    z = 1 + (zoom - 1) * (0.25 + 1.5 * dd)
    sx = (cx + (X - cx) / z - dx * (0.15 + 1.2 * dd)).astype(np.float32)
    sy = (cy + (Y - cy) / z - dy * (0.15 + 1.2 * dd)).astype(np.float32)
    return sx, sy, (Wb, Hb), H

def camera_camadas(q, cam, t, modo, W, forca=1.0):
    """q = quadro da pintura (já com rosto/chama animados); cam = saída de prepara(). Figura sobre fundo, cada um com sua profundidade."""
    H0, W0 = q.shape[:2]
    # o fundo recebe os pixels do quadro atual fora da figura (chama, fumaça, poeira continuam vivos) e o preenchimento dentro
    fora = (cam['figura'] == 0) & ~cam['buraco'] & ~cam['caixas']
    fundo = np.where(fora[..., None], q, cam['fundo'])
    sxf, syf, tam, H = _mapa(cam['prof_fundo'], W, H0, W0, t, modo, forca)
    sxp, syp, _, _ = _mapa(cam['prof_fig'], W, H0, W0, t, modo, forca)
    rf = cv2.remap(cv2.resize(fundo, tam, interpolation=cv2.INTER_AREA), sxf, syf, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    rp = cv2.remap(cv2.resize(q, tam, interpolation=cv2.INTER_AREA), sxp, syp, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    ra = cv2.remap(cv2.resize(alfa_quadro(q, cam), tam, interpolation=cv2.INTER_AREA), sxp, syp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = np.clip(ra, 0, 1)[..., None]
    return (rp.astype(np.float32) * a + rf.astype(np.float32) * (1 - a)).astype(np.uint8)
