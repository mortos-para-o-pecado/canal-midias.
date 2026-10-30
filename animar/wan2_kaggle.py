# Teste 2 do Wan 2.2 (imagem -> vídeo, 14B, Apache 2.0) no Kaggle (T4 grátis): ANIMAÇÃO DE VERDADE.
# O teste 1 (wan_kaggle.py) prendeu a mesma pintura no primeiro e no último quadro e mandou "quase parado": saiu parado.
# Aqui: só o primeiro quadro (a pintura), um gesto claro por cena (levantar a cabeça, erguer a mão, a chama dançar,
# o tecido se mexer), câmera lenta permitida, e três amostradores: os 2 primeiros passos do modelo de ruído alto SEM o LoRA
# de 4 passos (com CFG 3,5, que é o que dá movimento) e depois o LoRA Lightx2v para acabar rápido.
# Versão 3 (29/10, depois do teste 2): o teste 2 deu MOVIMENTO DE VERDADE (La Tour: a mão sai do queixo e vai à caveira,
# a cabeça se ergue e vira para a vela), mas levou 60 min por clipe a 656x832x81 e o ComfyUI morreu por falta de memória no
# segundo clipe. Agora: lado maior 640, 49 quadros (3 s), 4 passos (1 sem LoRA + 3 com LoRA), câmera parada (a pintura viva
# é colada sobre a pintura nítida, só onde se mexe) e o ComfyUI reiniciado a cada clipe.
# Versão 4 (29/10 noite, estudo no Gemini Pro; regra do autor: "se o estudo diz o contrário, fica o estudo"): shift 5,
# CFG 2,5 no passo sem LoRA, LoRA a 0,7, negativo sem "pintura/obra/estilo", textura de óleo no positivo, VAE em ladrilhos.
# Saídas em /kaggle/working: <nome>.mp4 (16 qps), <nome>_tira.jpg, <nome>_mov.txt (quanto cada quadro se mexeu), log.txt.
# Nenhuma chave ou senha: tudo vem de repositórios públicos.
import os, sys, re, json, time, glob, shutil, subprocess, urllib.request

OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('saida_wan2')
BASE = '/tmp/wan'
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

# negativo padrão do Wan (inclui "estático", "parado", "imagem parada") + o que estraga pintura.
# 29/10 (versão 4): TIRADOS do negativo padrão "风格，作品，画作，画面" (estilo, obra, PINTURA, quadro): o negativo oficial manda o
# modelo FUGIR de pintura, e isso empurra o rosto para "foto" (foi o que se viu no La Tour da versão 6).
NEG = ('色调艳丽，过曝，静态，细节模糊不清，字幕，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，'
       '多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, '
       'morphing face, changing face, different person, deformed hands, extra fingers, melting, warping, text, watermark, '
       'photorealistic photo, photograph, smooth plastic skin, 3d render, cartoon, frozen still image, camera movement, zoom, pan, '
       'darkening, fading to black, flicker, hyperrealistic skin, smooth skin, dark frame, color fade, digital art, blur')
ESTILO = (' Keep it a classical oil painting with oil painting texture, visible brushstrokes and impasto: same colors, same '
          'brushstrokes, same canvas texture, same face and same person. '
          'Natural, smooth, clearly visible motion of the figures, same lighting. The camera does not move.')

CENAS = [
    dict(nome='pedro_negacao_vida', img='pedro_negacao.jpg', seed=51,
         pos='Rembrandt painting of the Denial of Peter comes to life. The servant girl holds up the candle and points at Peter, '
             'accusing him. Peter lifts his open hand and slowly shakes his head in denial, his eyes widen. The soldier in armor turns '
             'his head toward Peter. The candlelight flickers warmly on their faces and on the white cloak.'),
    dict(nome='paulo_prisao_vida', img='paulo_prisao.jpg', seed=61,
         pos='Rembrandt painting of the old apostle Paul in prison comes to life. Paul lowers his hand from his white beard, takes '
             'the quill and writes on the page of the open book, his head nods slowly as he thinks. The beam of sunlight from the '
             'window shimmers on the wall and dust floats in the light.'),
    dict(nome='moises_vida', img='moises_lei.jpg', seed=71,
         pos='Rembrandt painting of Moses with the Ten Commandments comes to life. Moses slowly raises the two stone tablets of the '
             'Law higher above his head with both hands, his face fills with grief, his arms tremble under the weight, his white '
             'sleeves and his beard move.'),
    dict(nome='belsazar_vida', img='belsazar.jpg', seed=81,
         pos='Rembrandt painting of Belshazzar\'s Feast comes to life. The glowing hand on the dark wall moves as it writes, the '
             'shining letters glow brighter. The king turns in terror and pulls back, his golden robe shimmers. The guests recoil '
             'in fear and wine spills from the golden cup.'),
    dict(nome='mateus_vida', img='caravaggio_mateus.jpg', seed=91,
         pos='Caravaggio painting of the Calling of Saint Matthew comes to life. Christ, on the right, extends his arm and points '
             'at Matthew. The bearded Matthew at the table raises his head and points at himself in surprise. The young men at the '
             'table turn to look. The beam of light on the wall slowly brightens and coins glint on the table.'),
    dict(nome='paulo_cartas_vida', img='valentin_epistolas.jpg', seed=101,
         pos='Baroque painting of the apostle Paul writing his letters comes to life. Paul writes with the quill across the page, '
             'his hand moves steadily, he lifts his eyes to think and then looks down again at the letter. His red cloak moves '
             'gently and warm light shimmers on the papers.'),
    # feito no teste 2 (versão 6): latour_vida
    dict(nome='latour_vida_x', img='latour_vela.jpg', seed=11,
         pos='Georges de La Tour candlelit painting comes to life. The young woman slowly lifts her head from her hand and turns her eyes '
             'toward the candle, her fingers move on the skull in her lap. The candle flame dances and flickers strongly, a thin line of '
             'smoke rises and curls, warm light pulses on her face, sleeves and the wall.'),
    dict(nome='paulo_vida', img='paulo_valentin.jpg', seed=21,
         pos='Rembrandt painting of the elderly apostle Paul comes to life. The old man slowly raises his head from the book and looks '
             'up toward the window light, thinking, then his hand turns a page of the open book on the desk. His brown robe and sash '
             'shift as he moves, his white beard moves slightly, sunlight shimmers on the wall and dust floats in the light.'),
    dict(nome='agostinho_vida', img='agostinho_champaigne.jpg', seed=31,
         pos='Baroque painting of Saint Augustine comes to life. The saint slowly lifts the burning heart higher toward the heavenly '
             'light, the flames on the heart flicker and rise, he gazes upward with devotion, his golden embroidered cope shimmers and '
             'its folds move, the quill in his other hand trembles above the book, rays of light from above slowly brighten.'),
    dict(nome='pedro_vida', img='pedro_lagrimas.jpg', seed=41,
         pos='El Greco painting of the repentant Saint Peter comes to life. The old apostle lifts his tearful eyes to heaven, a tear '
             'rolls down his cheek, he presses his clasped hands together, he breathes deeply, his blue robe and yellow cloak move gently, '
             'the ivy leaves behind him tremble in a light breeze.'),
]

def dims(path, lado=640):
    import cv2
    h, w = cv2.imread(path).shape[:2]
    if w >= h: W, H = lado, int(round(lado * h / w / 16)) * 16
    else: H, W = lado, int(round(lado * w / h / 16)) * 16
    return W, H

def prepara():
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv', ok_fail=True)
    log(sh('df -h /tmp /kaggle/working 2>/dev/null; free -g', ok_fail=True))
    if not os.path.isdir(COMFY):
        sh(f'git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI {COMFY}')
        sh(f'git clone -q --depth 1 https://github.com/city96/ComfyUI-GGUF {COMFY}/custom_nodes/ComfyUI-GGUF')
    sh(f'pip install -q -r {COMFY}/requirements.txt gguf hf_transfer huggingface_hub 2>&1 | tail -3', ok_fail=True)
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    from huggingface_hub import list_repo_files, hf_hub_download
    def pega(repo, padrao, destino):
        fs = [f for f in list_repo_files(repo) if re.search(padrao, f, re.I)]
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
        c['w'], c['h'] = dims(dst)
        log(c['nome'], 'tamanho', c['w'], 'x', c['h'])
    return m

def fluxo(m, c):
    S = 4   # passos: 0-1 alto SEM LoRA (CFG 3,5: movimento), 1-2 alto COM LoRA, 2-4 baixo COM LoRA (detalhe)
    return {
        '1': {'class_type': 'UnetLoaderGGUF', 'inputs': {'unet_name': m['alto']}},
        '2': {'class_type': 'UnetLoaderGGUF', 'inputs': {'unet_name': m['baixo']}},
        '3': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['1', 0], 'lora_name': m['lora_alto'], 'strength_model': c.get('lora', 0.7)}},
        '4': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['2', 0], 'lora_name': m['lora_baixo'], 'strength_model': c.get('lora', 0.7)}},
        '5': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': ['3', 0], 'shift': c.get('shift', 5.0)}},
        '6': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': ['4', 0], 'shift': c.get('shift', 5.0)}},
        '17': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': ['1', 0], 'shift': c.get('shift', 5.0)}},
        '7': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': m['texto'], 'type': 'wan', 'device': 'default'}},
        '8': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['7', 0], 'text': c['pos'] + ESTILO}},
        '9': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['7', 0], 'text': NEG}},
        '10': {'class_type': 'VAELoader', 'inputs': {'vae_name': m['vae']}},
        '11': {'class_type': 'LoadImage', 'inputs': {'image': c['img']}},
        '12': {'class_type': 'WanImageToVideo', 'inputs': {'positive': ['8', 0], 'negative': ['9', 0], 'vae': ['10', 0],
               'width': c['w'], 'height': c['h'], 'length': 49, 'batch_size': 1, 'start_image': ['11', 0]}},
        '13': {'class_type': 'KSamplerAdvanced', 'inputs': {'model': ['17', 0], 'add_noise': 'enable', 'noise_seed': c['seed'], 'steps': S,
               'cfg': c.get('cfg', 2.5), 'sampler_name': 'euler', 'scheduler': 'simple', 'positive': ['12', 0], 'negative': ['12', 1],
               'latent_image': ['12', 2], 'start_at_step': 0, 'end_at_step': 1, 'return_with_leftover_noise': 'enable'}},
        '14': {'class_type': 'KSamplerAdvanced', 'inputs': {'model': ['5', 0], 'add_noise': 'disable', 'noise_seed': c['seed'], 'steps': S,
               'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'positive': ['12', 0], 'negative': ['12', 1],
               'latent_image': ['13', 0], 'start_at_step': 1, 'end_at_step': 2, 'return_with_leftover_noise': 'enable'}},
        '15': {'class_type': 'KSamplerAdvanced', 'inputs': {'model': ['6', 0], 'add_noise': 'disable', 'noise_seed': c['seed'], 'steps': S,
               'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'positive': ['12', 0], 'negative': ['12', 1],
               'latent_image': ['14', 0], 'start_at_step': 2, 'end_at_step': S, 'return_with_leftover_noise': 'disable'}},
        '16': ({'class_type': 'VAEDecodeTiled', 'inputs': {'samples': ['15', 0], 'vae': ['10', 0], 'tile_size': 512, 'overlap': 64,
                'temporal_size': 64, 'temporal_overlap': 8}} if c.get('ladrilho', True) else
               {'class_type': 'VAEDecode', 'inputs': {'samples': ['15', 0], 'vae': ['10', 0]}}),
        '18': {'class_type': 'SaveImage', 'inputs': {'images': ['16', 0], 'filename_prefix': c['nome']}},
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
    except urllib.error.HTTPError as e:
        log('fluxo recusado', e.read().decode()[:3000])
        if c.get('ladrilho', True):   # o nó de VAE em ladrilhos mudou de nome ou de entradas: tenta o simples
            c['ladrilho'] = False; log('tentando de novo com VAEDecode simples'); return gera(m, c)
        return False
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
    sh(f'ffmpeg -v error -y -framerate 16 -pattern_type glob -i "{COMFY}/output/{c["nome"]}_*.png" -c:v libx264 -crf 15 '
       f'-pix_fmt yuv420p -movflags +faststart {mp4}')
    q = [cv2.imread(f) for f in fs]
    # quanto se mexeu: fluxo ótico médio entre o 1o quadro e cada quadro (em pixels)
    g0 = cv2.cvtColor(q[0], cv2.COLOR_BGR2GRAY); mov = []
    for i, f in enumerate(q):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        fl = cv2.calcOpticalFlowFarneback(g0, g, None, .5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(fl, axis=2); mov.append((i, float(mag.mean()), float(np.percentile(mag, 95))))
    open(f'{OUT}/{c["nome"]}_mov.txt', 'w').write('\n'.join(f'{i} media {a:.2f} p95 {b:.2f}' for i, a, b in mov))
    log(c['nome'], 'movimento até o fim: média %.1f px, p95 %.1f px' % (mov[-1][1], mov[-1][2]))
    sel = [0, 12, 24, 36, 48] if len(q) == 49 else list(range(0, len(q), max(1, len(q) // 5)))
    tira = np.hstack([cv2.putText(q[i].copy(), str(i), (8, 30), 0, 1, (0, 255, 255), 2) for i in sel])
    cv2.imwrite(f'{OUT}/{c["nome"]}_tira.jpg', tira, [cv2.IMWRITE_JPEG_QUALITY, 85])

if __name__ == '__main__':
    so = sys.argv[1:]
    m = prepara(); T0 = time.time()
    for c in CENAS:
        if so and c['nome'] not in so: continue
        if c['nome'].endswith('_x'): continue
        if time.time() - T0 > 3.2 * 3600: log('limite de tempo desta versão: parando antes de', c['nome']); break
        srv = servidor()    # um ComfyUI novo por clipe (no teste 2 o segundo clipe morreu por falta de memória)
        try:
            if gera(m, c): finaliza(c)
        except Exception as e: log('falhou', c['nome'], repr(e)[:500])
        srv.terminate()
        try: srv.wait(timeout=60)
        except Exception: pass
        sh('pkill -f "[m]ain.py --listen" || true', ok_fail=True); time.sleep(5)
    log('FIM')
