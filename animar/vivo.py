# Rosto vivo: dá movimento CORRETO ao rosto de uma pintura (piscar, respirar, leve giro da cabeça e do olhar)
# com o LivePortrait (licença MIT, Kuaishou). Nada é inventado: o modelo só move os pontos do próprio rosto.
# Sem o detector insightface (que é de uso não comercial): os 5 pontos do rosto (olho esq., olho dir., nariz,
# canto esq. da boca, canto dir.) são dados à mão; o marcador de pontos do próprio LivePortrait (MIT) refina.
# Depois, a câmera anda "dentro" do quadro pela profundidade (Depth Anything V2 Small, Apache 2.0).
# Uso no Colab (T4):  !wget -q <url>/animar/vivo.py -O vivo.py   e   %run vivo.py
import os, sys, subprocess, math, time
os.chdir('/content')
if not os.path.exists('LivePortrait'):
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/KwaiVGI/LivePortrait.git'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tyro', 'onnxruntime', 'imageio', 'imageio-ffmpeg'], check=True)
from huggingface_hub import snapshot_download
snapshot_download('KlingTeam/LivePortrait', local_dir='LivePortrait/pretrained_weights', allow_patterns=['liveportrait/*'])
if not os.path.exists('da2s.onnx'):
    subprocess.run(['wget', '-q', '-O', 'da2s.onnx',
                    'https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx'], check=True)
sys.path.insert(0, '/content/LivePortrait')
import numpy as np, cv2, torch, imageio, onnxruntime as ort
from urllib.request import urlopen
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.human_landmark_runner import LandmarkRunner
from src.utils.crop import crop_image, prepare_paste_back, paste_back
from src.utils.camera import get_rotation_matrix
from src.utils.retargeting_utils import calc_eye_close_ratio
from IPython.display import HTML, display
from base64 import b64encode

BASE = 'https://raw.githubusercontent.com/mortos-para-o-pecado/canal-midias./main/animar/'
FPS, SEG = 24, 5.0
# pts: olho esq., olho dir., nariz, canto esq. e dir. da boca (pixels na imagem original)
# cabeca: amplitude em graus (giro de lado, inclinação, aceno da respiração); olhar: desvio do olhar (0 = parado)
CASOS = {
    'lutero_cranach':  dict(pts=[(556, 428), (734, 409), (690, 547), (615, 644), (777, 619)], piscadas=[1.2, 3.7],
                            giro=2.5, incl=0.8, resp=1.0, olhar=4, camera='dolly'),
    'paulo_rembrandt': dict(pts=[(261, 214), (305, 214), (288, 242), (277, 261), (302, 261)], piscadas=[1.6, 4.1],
                            giro=2.0, incl=0.6, resp=1.2, olhar=3, camera='orbita'),
    'paulo_valentin':  dict(pts=[(345, 292), (395, 290), (362, 330), (345, 352), (385, 350)], piscadas=[2.0],
                            giro=1.5, incl=0.8, resp=1.2, olhar=0, camera='orbita'),
    'latour_vela':     dict(pts=[(478, 232), (455, 235), (462, 270), (442, 292), (458, 293)], piscadas=[2.2],
                            giro=1.2, incl=0.5, resp=1.0, olhar=0, camera='orbita', vela=(0.66, 0.30)),
}
SO = os.environ.get('SO')            # para rodar um só: %env SO=lutero_cranach
os.makedirs('vivo', exist_ok=True)

cfg = InferenceConfig(); cfg.flag_use_half_precision = True
LP = LivePortraitWrapper(cfg)
LM = LandmarkRunner(ckpt_path=CropConfig().landmark_ckpt_path, onnx_provider='cpu')
DEP = ort.InferenceSession('da2s.onnx', providers=['CPUExecutionProvider'])

def baixa(nome):
    for t in range(5):
        try:
            a = np.frombuffer(urlopen(BASE + nome + '.jpg', timeout=60).read(), np.uint8)
            return cv2.cvtColor(cv2.imdecode(a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        except Exception as ex:
            print('  nova tentativa', nome, type(ex).__name__); time.sleep(4)
    raise RuntimeError('não baixou ' + nome)

def suave(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)
def piscada(t, t0):   # fecha em 0,09 s, segura 0,03 s, abre em 0,15 s (piscar natural)
    d = t - t0
    if d < 0 or d > 0.27: return 0.0
    if d < 0.09: return suave(d / 0.09)
    if d < 0.12: return 1.0
    return 1.0 - suave((d - 0.12) / 0.15)

def olhar_delta(delta, gx):   # mesma conta do LivePortrait (gradio_pipeline) para o olhar de lado
    if gx > 0: delta[0, 11, 0] += gx * 0.0007; delta[0, 15, 0] += gx * 0.001
    else:      delta[0, 11, 0] += gx * 0.001;  delta[0, 15, 0] += gx * 0.0007
    return delta

def profundidade(img):
    S = 518; x = cv2.resize(img, (S, S), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
    d = DEP.run(None, {DEP.get_inputs()[0].name: x})[0].squeeze()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

def camera(quadros, d, modo, larg=720, vela=None, forca=1.0):
    # a câmera anda dentro do quadro: o perto se move mais que o fundo; nada é redesenhado
    H0, W0 = quadros[0].shape[:2]; W = larg; H = int(round(W * H0 / W0 / 2)) * 2; M = 1.12
    Wb, Hb = int(W * M), int(H * M)
    d = cv2.GaussianBlur(cv2.resize(d, (Wb, Hb)), (0, 0), 3)
    d = cv2.GaussianBlur(np.maximum(d, cv2.dilate(d, np.ones((9, 9), np.uint8)) * 0.98), (0, 0), 2)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32); ox, oy = (Wb - W) / 2, (Hb - H) / 2
    X, Y = xs + ox, ys + oy; dd = cv2.remap(d, X, Y, cv2.INTER_LINEAR); cx, cy = Wb / 2, Hb / 2
    peso = None
    if vela: peso = np.exp(-((np.hypot(X - vela[0] * Wb, Y - vela[1] * Hb) / W) / 0.35) ** 2)[..., None]
    rng = np.random.default_rng(3); N = len(quadros)
    treme = np.convolve(0.5 * np.sin(np.arange(N) * 0.9) + 0.5 * rng.normal(0, 1, N), np.ones(3) / 3, 'same')
    out = []
    for i, q in enumerate(quadros):
        t = 0.5 - 0.5 * math.cos(math.pi * i / (N - 1)); k = forca
        if modo == 'dolly': zoom, dx, dy = 1 + 0.06 * k * t, 0.0, 0.0
        elif modo == 'orbita': zoom, dx, dy = 1 + 0.02 * k * t, (t - 0.5) * 0.05 * k * W, 0.0
        else: zoom, dx, dy = 1 + 0.03 * k * t, 0.0, (0.5 - t) * 0.04 * k * H
        z = 1 + (zoom - 1) * (0.25 + 1.5 * dd)
        sx = (cx + (X - cx) / z - dx * (0.15 + 1.2 * dd)).astype(np.float32)
        sy = (cy + (Y - cy) / z - dy * (0.15 + 1.2 * dd)).astype(np.float32)
        src = cv2.resize(q, (Wb, Hb), interpolation=cv2.INTER_AREA).astype(np.float32)
        r = cv2.remap(src, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        if peso is not None: r = r * (1 + 0.035 * treme[i] * peso)
        out.append(np.clip(r, 0, 255).astype(np.uint8))
    return out

def grava(quadros, arq):
    w = imageio.get_writer(arq, fps=FPS, codec='libx264', quality=8, macro_block_size=2, pixelformat='yuv420p')
    for q in quadros: w.append_data(q)
    w.close()

def mostra(arq, rotulo):
    d = b64encode(open(arq, 'rb').read()).decode()
    display(HTML(f'<p>{rotulo}</p><video width=360 controls loop autoplay muted src="data:video/mp4;base64,{d}"></video>'))

@torch.no_grad()
def anima(nome, c):
    img = baixa(nome); h, w = img.shape[:2]
    p5 = np.array(c['pts'], np.float32)
    l1 = LM.run(img, p5); lmk = LM.run(img, l1)                     # 203 pontos do rosto, refinados duas vezes
    ci = crop_image(img, lmk, dsize=512, scale=2.3, vx_ratio=0, vy_ratio=-0.125, flag_do_rot=True)
    I_s = LP.prepare_source(cv2.resize(ci['img_crop'], (256, 256), interpolation=cv2.INTER_AREA))
    mask = prepare_paste_back(cfg.mask_crop, ci['M_c2o'], dsize=(w, h))
    info = LP.get_kp_info(I_s); f_s = LP.extract_feature_3d(I_s); x_s = LP.transform_keypoint(info)
    p0, y0, r0 = info['pitch'], info['yaw'], info['roll']
    olho0 = float(calc_eye_close_ratio(lmk[None]).mean())
    print(f'{nome}: pose original pitch {p0.item():.1f} yaw {y0.item():.1f} roll {r0.item():.1f} | olho aberto {olho0:.2f}')
    N = int(SEG * FPS); D = SEG; quadros = []
    for i in range(N):
        t = i / FPS
        dy = c['giro'] * math.sin(2 * math.pi * t / D)
        dr = c['incl'] * math.sin(2 * math.pi * t / D + 1.0)
        dp = c['resp'] * math.sin(4 * math.pi * t / D)           # duas respirações (aceno leve)
        R_d = get_rotation_matrix(p0 + dp, y0 + dy, r0 + dr)
        delta = info['exp'].clone()
        if c['olhar']: delta = olhar_delta(delta, c['olhar'] * math.sin(2 * math.pi * t / D + 0.6))
        x_d = info['scale'] * (info['kp'] @ R_d + delta) + info['t']
        b = max([piscada(t, t0) for t0 in c['piscadas']] + [0.0])
        alvo = olho0 * (1 - 0.95 * b)
        x_d = x_d + LP.retarget_eye(x_s, LP.calc_combined_eye_ratio([[alvo]], lmk))
        x_d = LP.stitching(x_s, x_d)
        out = LP.parse_output(LP.warp_decode(f_s, x_s, x_d)['out'])[0]
        quadros.append(paste_back(out, ci['M_c2o'], img, mask))
    return img, quadros

for nome, c in CASOS.items():
    if SO and nome != SO: continue
    t0 = time.time()
    try:
        img, qs = anima(nome, c)
        grava([cv2.resize(q, (720, int(round(720 * q.shape[0] / q.shape[1] / 2)) * 2), interpolation=cv2.INTER_AREA) for q in qs], f'vivo/{nome}_rosto.mp4')
        d = profundidade(img)
        grava(camera(qs, d, c['camera'], vela=c.get('vela')), f'vivo/{nome}_cam.mp4')
    except Exception as ex:
        import traceback; traceback.print_exc(); print('ERRO', nome, type(ex).__name__, ex); continue
    print(f'{nome}: pronto em {time.time() - t0:.0f} s', flush=True)
    mostra(f'vivo/{nome}_rosto.mp4', nome + ' · só o rosto'); mostra(f'vivo/{nome}_cam.mp4', nome + ' · rosto + câmera')
