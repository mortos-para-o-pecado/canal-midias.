# Rosto vivo no Kaggle (placa T4): LivePortrait (MIT) move só o rosto da pintura (piscar, respirar, leve giro e olhar).
# Mesmo método de vivo.py (5 pontos dados à mão, sem o detector insightface, máscara só do contorno do rosto),
# mas a saída é só o RECORTE do rosto em resolução original + a máscara: o resto da pintura não muda,
# então o arquivo fica pequeno e a colagem final é feita fora daqui, na pintura original nítida.
# Saídas em /kaggle/working: <nome>_rosto.mp4 (recorte, 24 q/s), <nome>_mascara.png, <nome>.json (caixa), <nome>_tira.jpg
import os, sys, subprocess, math, time, json
W = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('saida_rosto')
os.makedirs(W, exist_ok=True); os.chdir('/tmp')
LOG = open(os.path.join(W, 'log_rosto.txt'), 'a', buffering=1)
def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')
if not os.path.exists('LivePortrait'):
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/KwaiVGI/LivePortrait.git'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tyro', 'onnxruntime', 'rich', 'pyyaml'], check=False)
from huggingface_hub import snapshot_download
snapshot_download('KlingTeam/LivePortrait', local_dir='LivePortrait/pretrained_weights', allow_patterns=['liveportrait/*'])
sys.path.insert(0, '/tmp/LivePortrait')
import numpy as np, cv2, torch
from urllib.request import urlopen
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.human_landmark_runner import LandmarkRunner
from src.utils.crop import crop_image, prepare_paste_back, paste_back
from src.utils.camera import get_rotation_matrix
from src.utils.retargeting_utils import calc_eye_close_ratio

RAW = 'https://raw.githubusercontent.com/mortos-para-o-pecado/canal-midias./main/'
FPS, SEG = 24, 6.0
# pts (pixels na imagem original): olho esq., olho dir., nariz, canto esq. e dir. da boca
CASOS = {
    'agostinho_champaigne': dict(img='animar/agostinho_champaigne.jpg', pts=[(683, 322), (728, 337), (695, 368), (668, 388), (698, 392)],
                                 piscadas=[1.6, 4.4], giro=0.8, incl=0.4, resp=0.7, olhar=2),
    'maria_sassoferrato':   dict(img='animar/maria_sassoferrato.jpg', pts=[(573, 546), (712, 570), (623, 662), (588, 673), (660, 683)],
                                 piscadas=[2.6], giro=0.6, incl=0.6, resp=0.9, olhar=0),
    'lutero_cranach':       dict(img='animar/lutero_cranach.jpg', pts=[(556, 428), (734, 409), (690, 547), (615, 644), (777, 619)],
                                 piscadas=[1.2, 4.2], giro=1.2, incl=0.5, resp=0.8, olhar=4),
    # 28/10: rostos das pinturas novas do EP03/EP04
    'pedro_lagrimas':       dict(img='animar/pedro_lagrimas.jpg', pts=[(729, 584), (856, 546), (818, 649), (799, 730), (910, 723)],
                                 piscadas=[1.8, 4.6], giro=0.6, incl=0.4, resp=0.8, olhar=0),
    'scheffer_monica':      dict(img='animar/scheffer_monica.jpg', pts=[(842, 624), (929, 637), (916, 690), (878, 729), (929, 734)],
                                 piscadas=[2.4], giro=0.5, incl=0.4, resp=0.7, olhar=0),
    'murillo_imaculada':    dict(img='animar/murillo_imaculada.jpg', pts=[(1022, 428), (1089, 449), (1050, 482), (1022, 508), (1057, 515)],
                                 piscadas=[1.4, 4.3], giro=0.5, incl=0.5, resp=0.8, olhar=0),
}
SO = [x for x in sys.argv[1:]]

cfg = InferenceConfig(); cfg.flag_use_half_precision = torch.cuda.is_available()
if not torch.cuda.is_available(): cfg.flag_force_cpu = True
log('rodando em', 'GPU' if torch.cuda.is_available() else 'CPU')
LP = LivePortraitWrapper(cfg)
LM = LandmarkRunner(ckpt_path=CropConfig().landmark_ckpt_path, onnx_provider='cpu')

def baixa(rel):
    for t in range(5):
        try:
            a = np.frombuffer(urlopen(RAW + rel, timeout=60).read(), np.uint8)
            return cv2.cvtColor(cv2.imdecode(a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        except Exception as ex:
            log('  nova tentativa', rel, type(ex).__name__); time.sleep(4)
    raise RuntimeError('não baixou ' + rel)

def suave(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)
def piscada(t, t0):
    d = t - t0
    if d < 0 or d > 0.27: return 0.0
    if d < 0.09: return suave(d / 0.09)
    if d < 0.12: return 1.0
    return 1.0 - suave((d - 0.12) / 0.15)
def olhar_delta(delta, gx):
    if gx > 0: delta[0, 11, 0] += gx * 0.0007; delta[0, 15, 0] += gx * 0.001
    else:      delta[0, 11, 0] += gx * 0.001;  delta[0, 15, 0] += gx * 0.0007
    return delta

@torch.no_grad()
def anima(nome, c):
    img = baixa(c['img']); h, w = img.shape[:2]
    p5 = np.array(c['pts'], np.float32)
    l1 = LM.run(img, p5); lmk = LM.run(img, l1)
    ci = crop_image(img, lmk, dsize=512, scale=2.3, vx_ratio=0, vy_ratio=-0.125, flag_do_rot=True)
    I_s = LP.prepare_source(cv2.resize(ci['img_crop'], (256, 256), interpolation=cv2.INTER_AREA))
    mask = prepare_paste_back(cfg.mask_crop, ci['M_c2o'], dsize=(w, h))
    info = LP.get_kp_info(I_s); f_s = LP.extract_feature_3d(I_s); x_s = LP.transform_keypoint(info)
    p0, y0, r0 = info['pitch'], info['yaw'], info['roll']
    olho0 = float(calc_eye_close_ratio(lmk[None]).mean())
    log(f'{nome}: pose pitch {p0.item():.1f} yaw {y0.item():.1f} roll {r0.item():.1f} | olho aberto {olho0:.2f}')
    hull = cv2.convexHull(lmk.astype(np.float32)).reshape(-1, 2); cen = hull.mean(0)
    hull = cen + (hull - cen) * np.array([c.get('larg_mascara', 1.18), 1.22]); hull[:, 1] -= 0.04 * (hull[:, 1].max() - hull[:, 1].min())
    fm = np.zeros((h, w), np.float32); cv2.fillPoly(fm, [hull.astype(np.int32)], 1.0)
    tam = hull[:, 0].max() - hull[:, 0].min(); fm = cv2.GaussianBlur(fm, (0, 0), max(2.0, tam * 0.05))
    mask = mask * fm[..., None]
    m1 = mask[..., 0] if mask.ndim == 3 else mask
    ys, xs = np.where(m1 > 0.004); pad = 6
    x0, y0b = max(0, xs.min() - pad), max(0, ys.min() - pad); x1, y1 = min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1)
    if (x1 - x0) % 2: x1 = x1 - 1 if x1 - 1 > x0 else x1 + 1
    if (y1 - y0b) % 2: y1 = y1 - 1 if y1 - 1 > y0b else y1 + 1
    N = int(SEG * FPS); D = SEG; out_frames = []
    for i in range(N):
        t = i / FPS
        dy = c['giro'] * math.sin(2 * math.pi * t / D)
        dr = c['incl'] * math.sin(2 * math.pi * t / D + 1.0)
        dp = c['resp'] * math.sin(4 * math.pi * t / D)
        R_d = get_rotation_matrix(p0 + dp, y0 + dy, r0 + dr)
        delta = info['exp'].clone()
        if c['olhar']: delta = olhar_delta(delta, c['olhar'] * math.sin(2 * math.pi * t / D + 0.6))
        x_d = info['scale'] * (info['kp'] @ R_d + delta) + info['t']
        b = max([piscada(t, t0) for t0 in c['piscadas']] + [0.0])
        x_d = x_d + LP.retarget_eye(x_s, LP.calc_combined_eye_ratio([[olho0 * (1 - 0.95 * b)]], lmk))
        x_d = LP.stitching(x_s, x_d)
        out = LP.parse_output(LP.warp_decode(f_s, x_s, x_d)['out'])[0]
        q = paste_back(out, ci['M_c2o'], img, mask)
        out_frames.append(q[y0b:y1, x0:x1])
    return img, out_frames, (int(x0), int(y0b), int(x1), int(y1)), m1[y0b:y1, x0:x1]

for nome, c in CASOS.items():
    if SO and nome not in SO: continue
    t0 = time.time()
    try:
        img, qs, caixa, m = anima(nome, c)
        x0, y0, x1, y1 = caixa
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{x1 - x0}x{y1 - y0}', '-r', str(FPS),
                              '-i', '-', '-c:v', 'libx264', '-crf', '12', '-preset', 'slow', '-pix_fmt', 'yuv444p', f'{W}/{nome}_rosto.mp4'], stdin=subprocess.PIPE)
        for q in qs: p.stdin.write(np.ascontiguousarray(q).tobytes())
        p.stdin.close(); p.wait()
        cv2.imwrite(f'{W}/{nome}_mascara.png', (np.clip(m, 0, 1) * 255).astype(np.uint8))
        json.dump(dict(caixa=caixa, tamanho=[img.shape[1], img.shape[0]], fps=FPS, n=len(qs), piscadas=c['piscadas']), open(f'{W}/{nome}.json', 'w'))
        sel = [0, int(c['piscadas'][0] * FPS) + 3, len(qs) // 2, len(qs) - 1]
        tira = np.hstack([cv2.resize(qs[i], (240, int(240 * (y1 - y0) / (x1 - x0)))) for i in sel])
        cv2.imwrite(f'{W}/{nome}_tira.jpg', cv2.cvtColor(tira, cv2.COLOR_RGB2BGR))
        log(f'{nome}: pronto em {time.time() - t0:.0f} s, caixa {caixa}, {os.path.getsize(f"{W}/{nome}_rosto.mp4") // 1024} KB')
    except Exception as ex:
        import traceback; traceback.print_exc(); log('ERRO', nome, type(ex).__name__, ex)
log('FIM')
