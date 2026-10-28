# Teste do Wan 2.2 (imagem -> vídeo, 14B, licença Apache 2.0) com a MESMA pintura como primeiro e último quadro,
# rodando no Kaggle (placa T4 grátis). A pintura fica presa nas duas pontas; o modelo só pode criar o movimento do meio.
# Usa ComfyUI (GPL-3, só como ferramenta) + ComfyUI-GGUF (modelos comprimidos que cabem na T4) + LoRA Lightx2v de 4 passos.
# Saídas em /kaggle/working: <nome>.mp4, <nome>_tira.jpg (quadros), <nome>_rosto.txt (quanto o rosto mudou), log.txt.
# Nenhuma chave ou senha é usada: todos os arquivos vêm de repositórios públicos.
import os, sys, re, json, time, glob, shutil, subprocess, urllib.request, base64

OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('saida_wan')
BASE = '/tmp/wan'                     # modelos ficam fora de /kaggle/working (não entram na saída salva)
COMFY = os.path.join(BASE, 'ComfyUI')
RAW = 'https://raw.githubusercontent.com/mortos-para-o-pecado/canal-midias./main/animar/'
os.makedirs(OUT, exist_ok=True); os.makedirs(BASE, exist_ok=True)
LOG = open(os.path.join(OUT, 'log.txt'), 'a', buffering=1)

def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n')

def sh(cmd, ok_fail=False):
    log('$', cmd[:160])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode and not ok_fail:
        log('ERRO', r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
    return r.stdout

NEG = ('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，'
       '多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, '
       'morphing face, changing face, different person, deformed hands, extra fingers, camera movement, zoom, text, watermark, photorealistic, 3d render')

CENAS = [
    dict(nome='latour_wan', img='latour_vela.jpg', w=480, h=608, seed=42,
         face=(380, 170, 520, 330),   # caixa do rosto na pintura original (x0,y0,x1,y1)
         pos=('A classical oil painting by Georges de La Tour comes gently to life. A young woman sits still in quiet contemplation, '
              'her chin resting on her hand, looking at the candle. The candle flame flickers softly, a thin wisp of smoke rises and sways, '
              'warm candlelight breathes gently on her face, her white sleeves and the wall. She breathes slowly, her shoulders rise very slightly. '
              'Static camera, no zoom. Preserve the original oil painting texture, brushstrokes and canvas grain.')),
    dict(nome='paulo_wan', img='paulo_valentin.jpg', w=512, h=608, seed=7,
         face=(270, 200, 460, 440),
         pos=('A classical oil painting by Rembrandt comes gently to life. An elderly apostle with a long white beard stands at his writing desk, '
              'his hand resting on an open book, deep in thought, looking down. He breathes slowly, his chest rises gently, the folds of his '
              'brown robe and his sash shift very slightly, warm sunlight shimmers softly on the wall, fine dust floats in the light. '
              'Static camera, no zoom. Preserve the original oil painting texture, brushstrokes and canvas grain.')),
]

def prepara():
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv', ok_fail=True)
    log(sh('df -h /tmp /kaggle/working 2>/dev/null; free -g', ok_fail=True))
    if not os.path.isdir(COMFY):
        sh(f'git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI {COMFY}')
        sh(f'git clone -q --depth 1 https://github.com/city96/ComfyUI-GGUF {COMFY}/custom_nodes/ComfyUI-GGUF')
    sh(f'pip install -q -r {COMFY}/requirements.txt gguf hf_transfer huggingface_hub 2>&1 | tail -3', ok_fail=True)
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    from huggingface_hub import list_repo_files, hf_hub_download
    def pega(repo, padrao, destino, evita=None):
        fs = [f for f in list_repo_files(repo) if re.search(padrao, f, re.I) and not (evita and re.search(evita, f, re.I))]
        if not fs: log('NÃO ACHEI', repo, padrao); raise SystemExit(1)
        f = sorted(fs, key=len)[0]; d = os.path.join(COMFY, 'models', destino); os.makedirs(d, exist_ok=True)
        alvo = os.path.join(d, os.path.basename(f))
        if not os.path.exists(alvo):
            t = time.time(); p = hf_hub_download(repo, f, local_dir=os.path.join(BASE, 'hf'))
            shutil.move(p, alvo); log('baixado', f, '%.1f GB' % (os.path.getsize(alvo) / 1e9), '%.0fs' % (time.time() - t))
        return os.path.basename(alvo)
    m = {}
    m['alto'] = pega('QuantStack/Wan2.2-I2V-A14B-GGUF', r'high.*q4_k_m\.gguf$', 'unet')
    m['baixo'] = pega('QuantStack/Wan2.2-I2V-A14B-GGUF', r'low.*q4_k_m\.gguf$', 'unet')
    rep = 'Comfy-Org/Wan_2.2_ComfyUI_Repackaged'
    m['texto'] = pega(rep, r'text_encoders/umt5_xxl_fp8_e4m3fn_scaled\.safetensors$', 'text_encoders')
    m['vae'] = pega(rep, r'vae/wan_2\.1_vae\.safetensors$', 'vae')
    m['lora_alto'] = pega(rep, r'loras/.*i2v.*lightx2v.*4steps.*high', 'loras')
    m['lora_baixo'] = pega(rep, r'loras/.*i2v.*lightx2v.*4steps.*low', 'loras')
    log('modelos', json.dumps(m))
    for c in CENAS:
        dst = os.path.join(COMFY, 'input', c['img'])
        if not os.path.exists(dst): urllib.request.urlretrieve(RAW + c['img'], dst)
    return m

def fluxo(m, c):
    return {
        '1': {'class_type': 'UnetLoaderGGUF', 'inputs': {'unet_name': m['alto']}},
        '2': {'class_type': 'UnetLoaderGGUF', 'inputs': {'unet_name': m['baixo']}},
        '3': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['1', 0], 'lora_name': m['lora_alto'], 'strength_model': 1.0}},
        '4': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['2', 0], 'lora_name': m['lora_baixo'], 'strength_model': 1.0}},
        '5': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': ['3', 0], 'shift': 5.0}},
        '6': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': ['4', 0], 'shift': 5.0}},
        '7': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': m['texto'], 'type': 'wan', 'device': 'default'}},
        '8': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['7', 0], 'text': c['pos']}},
        '9': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['7', 0], 'text': NEG}},
        '10': {'class_type': 'VAELoader', 'inputs': {'vae_name': m['vae']}},
        '11': {'class_type': 'LoadImage', 'inputs': {'image': c['img']}},
        '12': {'class_type': 'WanFirstLastFrameToVideo', 'inputs': {'positive': ['8', 0], 'negative': ['9', 0], 'vae': ['10', 0],
               'width': c['w'], 'height': c['h'], 'length': 81, 'batch_size': 1, 'start_image': ['11', 0], 'end_image': ['11', 0]}},
        '13': {'class_type': 'KSamplerAdvanced', 'inputs': {'model': ['5', 0], 'add_noise': 'enable', 'noise_seed': c['seed'], 'steps': 4,
               'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'positive': ['12', 0], 'negative': ['12', 1],
               'latent_image': ['12', 2], 'start_at_step': 0, 'end_at_step': 2, 'return_with_leftover_noise': 'enable'}},
        '14': {'class_type': 'KSamplerAdvanced', 'inputs': {'model': ['6', 0], 'add_noise': 'disable', 'noise_seed': c['seed'], 'steps': 4,
               'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'positive': ['12', 0], 'negative': ['12', 1],
               'latent_image': ['13', 0], 'start_at_step': 2, 'end_at_step': 4, 'return_with_leftover_noise': 'disable'}},
        '15': {'class_type': 'VAEDecode', 'inputs': {'samples': ['14', 0], 'vae': ['10', 0]}},
        '16': {'class_type': 'SaveImage', 'inputs': {'images': ['15', 0], 'filename_prefix': c['nome']}},
    }

def servidor():
    p = subprocess.Popen(f'cd {COMFY} && CUDA_VISIBLE_DEVICES=0 python main.py --listen 127.0.0.1 --port 8188 --lowvram '
                         f'--preview-method none --disable-auto-launch > {OUT}/comfy_log.txt 2>&1', shell=True)
    for _ in range(180):
        try: urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=2); log('ComfyUI no ar'); return p
        except Exception: time.sleep(2)
    log('ComfyUI não subiu'); log(open(f'{OUT}/comfy_log.txt').read()[-3000:]); raise SystemExit(1)

def gera(m, c):
    t = time.time()
    req = urllib.request.Request('http://127.0.0.1:8188/prompt', data=json.dumps({'prompt': fluxo(m, c)}).encode(),
                                 headers={'Content-Type': 'application/json'})
    try: pid = json.loads(urllib.request.urlopen(req).read())['prompt_id']
    except urllib.error.HTTPError as e: log('fluxo recusado', e.read().decode()[:3000]); return False
    while True:
        time.sleep(15)
        h = json.loads(urllib.request.urlopen(f'http://127.0.0.1:8188/history/{pid}').read())
        if pid in h:
            st = h[pid].get('status', {})
            if st.get('status_str') == 'error': log('ERRO na geração', json.dumps(st)[:3000]); return False
            log(c['nome'], 'pronto em %.1f min' % ((time.time() - t) / 60)); return True
        if int(time.time() - t) % 300 < 15: log(c['nome'], 'gerando... %.0f min' % ((time.time() - t) / 60))

def finaliza(c):
    import numpy as np, cv2
    fs = sorted(glob.glob(f'{COMFY}/output/{c["nome"]}_*.png'))
    if not fs: log('sem quadros', c['nome']); return
    mp4 = f'{OUT}/{c["nome"]}.mp4'
    sh(f'ffmpeg -v error -y -framerate 16 -pattern_type glob -i "{COMFY}/output/{c["nome"]}_*.png" -c:v libx264 -crf 16 '
       f'-pix_fmt yuv420p -movflags +faststart {mp4}')
    q = [cv2.imread(f) for f in fs]; h, w = q[0].shape[:2]
    orig = cv2.imread(f'{COMFY}/input/{c["img"]}'); H0, W0 = orig.shape[:2]
    # a pintura é ajustada ao tamanho do vídeo com recorte central: converte a caixa do rosto
    esc = max(w / W0, h / H0); ox, oy = (W0 * esc - w) / 2, (H0 * esc - h) / 2
    x0, y0, x1, y1 = [int(v) for v in (c['face'][0] * esc - ox, c['face'][1] * esc - oy, c['face'][2] * esc - ox, c['face'][3] * esc - oy)]
    ref = cv2.resize(orig, (int(W0 * esc), int(H0 * esc)), interpolation=cv2.INTER_AREA)[int(oy):int(oy) + h, int(ox):int(ox) + w]
    linhas = []
    for i, f in enumerate(q):
        d = cv2.absdiff(f[y0:y1, x0:x1], ref[y0:y1, x0:x1]).mean(); dt = cv2.absdiff(f, ref).mean()
        linhas.append(f'{i} rosto {d:.2f} quadro {dt:.2f}')
    open(f'{OUT}/{c["nome"]}_rosto.txt', 'w').write('\n'.join(linhas))
    dif = [float(l.split()[2]) for l in linhas]
    log(c['nome'], 'diferença do rosto para a pintura (0-255): média %.2f, máxima %.2f no quadro %d' % (np.mean(dif), max(dif), int(np.argmax(dif))))
    sel = [0, 20, 40, 60, 80] if len(q) > 80 else list(range(0, len(q), max(1, len(q) // 5)))
    tira = np.hstack([cv2.putText(q[i].copy(), str(i), (8, 30), 0, 1, (0, 255, 255), 2) for i in sel])
    rost = np.hstack([cv2.resize(q[i][y0:y1, x0:x1], (240, int(240 * (y1 - y0) / (x1 - x0)))) for i in sel])
    cv2.imwrite(f'{OUT}/{c["nome"]}_tira.jpg', tira, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(f'{OUT}/{c["nome"]}_rostos.jpg', rost, [cv2.IMWRITE_JPEG_QUALITY, 92])
    open(f'{OUT}/{c["nome"]}.mp4.b64', 'w').write(base64.b64encode(open(mp4, 'rb').read()).decode())

if __name__ == '__main__':
    so = sys.argv[1:]  # opcional: nomes das cenas
    m = prepara(); srv = servidor()
    for c in CENAS:
        if so and c['nome'] not in so: continue
        if gera(m, c): finaliza(c)
    srv.terminate(); log('FIM')
