# Teste de animação real de pinturas (LTX-Video, licença OpenRail-M/Apache) numa GPU T4 gratuita do Google Colab.
# Uso no Colab (uma célula):  !wget -q <url>/animar/animar.py -O animar.py   e depois   %run animar.py
# Gera saida/<nome>.mp4 (4 s, 24 fps) e mostra no notebook. O texto (T5) roda primeiro e é descartado,
# para caber na memória do Colab gratuito.
import os, subprocess, sys, gc, time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'diffusers>=0.33', 'transformers', 'accelerate', 'sentencepiece', 'imageio', 'imageio-ffmpeg'], check=True)
import torch
import torch.nn.functional as F
# A T4 não tem, em bf16, o atalho de memória da atenção: o PyTorch montaria a matriz inteira de uma vez (4 a 6 GB).
# Aqui a atenção é feita em fatias de 1024 posições: mesmo resultado, pouca memória.
_sdpa = F.scaled_dot_product_attention
def _sdpa_fatiado(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    n = q.shape[-2]
    if n <= 1024 or is_causal:
        return _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    out = []
    for i in range(0, n, 1024):
        m = attn_mask[..., i:i + 1024, :] if (attn_mask is not None and attn_mask.dim() >= 2 and attn_mask.shape[-2] == n) else attn_mask
        out.append(_sdpa(q[..., i:i + 1024, :], k, v, attn_mask=m, dropout_p=dropout_p, scale=scale, **kw))
    return torch.cat(out, dim=-2)
F.scaled_dot_product_attention = _sdpa_fatiado
from IPython.display import HTML, display
from base64 import b64encode
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

BASE = 'https://raw.githubusercontent.com/mortos-para-o-pecado/canal-midias./main/animar/'
NEG = ('worst quality, inconsistent motion, blurry, jittery, distorted, deformed face, deformed hands, extra fingers, '
       'morphing, melting, text, watermark, camera shake, fast motion')
CENAS = {
    'latour_vela': ('A classical oil painting comes to life: the candle flame flickers softly and its warm light trembles on the young woman\'s face, '
                    'her hand and the old books; she breathes slowly and blinks once; the smoke rises gently. Static camera, very subtle natural motion, '
                    'baroque chiaroscuro oil painting texture preserved.', 512, 672, .5),
    'paulo_valentin': ('A classical oil painting comes to life: an old bearded apostle sits at his desk, he breathes slowly, his head turns slightly '
                       'toward the light, the dust floats in the warm light from the window. Static camera, very subtle natural motion, '
                       'Rembrandt oil painting texture preserved.', 576, 672, .35),
    'pentecostes': ('A classical oil painting comes to life: tongues of fire flicker above the heads of the apostles, the light from the dove above '
                    'glows and pulses softly, robes move slightly. Slow gentle push-in camera, subtle natural motion, El Greco oil painting texture preserved.', 512, 768, .2),
}
os.makedirs('saida', exist_ok=True)
dt = torch.bfloat16

# 1) texto -> embeddings num processo separado (quando ele termina, a GPU fica 100% livre para o vídeo)
TEXTO = r"""
import torch, json, sys
from transformers import T5EncoderModel, T5Tokenizer
dt = torch.bfloat16
tok = T5Tokenizer.from_pretrained('Lightricks/LTX-Video', subfolder='tokenizer')
enc = T5EncoderModel.from_pretrained('Lightricks/LTX-Video', subfolder='text_encoder', dtype=dt, device_map='cuda', low_cpu_mem_usage=True)
txts = json.load(open('textos.json')); out = {}
for k, txt in txts.items():
    t = tok(txt, padding='max_length', max_length=128, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad(): e = enc(t.input_ids, attention_mask=t.attention_mask)[0]
    out[k] = (e.to(dt).cpu(), t.attention_mask.cpu())
torch.save(out, 'emb.pt'); print('textos prontos', flush=True)
"""
import json
json.dump(dict({k: v[0] for k, v in CENAS.items()}, _neg=NEG), open('textos.json', 'w'))
open('texto.py', 'w').write(TEXTO)
r = subprocess.run([sys.executable, 'texto.py'], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-600:] if r.returncode else '', flush=True)
EMB = torch.load('emb.pt')
E = {k: (EMB[k][0].to('cuda'), EMB[k][1].to('cuda')) for k in CENAS}; EN = (EMB['_neg'][0].to('cuda'), EMB['_neg'][1].to('cuda'))

# 2) imagem -> vídeo
pipe = LTXImageToVideoPipeline.from_pretrained('Lightricks/LTX-Video', text_encoder=None, tokenizer=None, torch_dtype=dt)
pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
def recorta(img, w, h, fy):
    # recorta na proporção do vídeo (sem esticar), com o centro vertical em fy (0 = topo)
    W, H = img.size; alvo = w / h
    if W / H > alvo: nw = int(H * alvo); x0 = (W - nw) // 2; img = img.crop((x0, 0, x0 + nw, H))
    else:
        nh = int(W / alvo); y0 = int(min(max(0, fy * H - nh / 2), H - nh)); img = img.crop((0, y0, W, y0 + nh))
    return img.resize((w, h))
def mostra(nome):
    d = b64encode(open(f'saida/{nome}.mp4', 'rb').read()).decode()
    display(HTML(f'<p>{nome}</p><video width=360 controls loop autoplay muted src="data:video/mp4;base64,{d}"></video>'))
for nome, (txt, w, h, fy) in CENAS.items():
    if os.path.exists(f'saida/{nome}.mp4'): mostra(nome); continue
    t0 = time.time()
    print(nome, '| GPU livre:', round(torch.cuda.mem_get_info()[0] / 2**30, 1), 'GB', flush=True)
    img = recorta(load_image(BASE + nome + '.jpg'), w, h, fy)
    (pe, pm), (ne, nm) = E[nome], EN
    try:
      v = pipe(image=img, prompt_embeds=pe, prompt_attention_mask=pm, negative_prompt_embeds=ne, negative_prompt_attention_mask=nm,
             width=w, height=h, num_frames=97, num_inference_steps=40, guidance_scale=3.0,
             generator=torch.Generator('cuda').manual_seed(7)).frames[0]
    except Exception as ex:
        print('ERRO', nome, type(ex).__name__, str(ex)[:300], 'GPU livre:', round(torch.cuda.mem_get_info()[0] / 2**30, 1), 'GB', flush=True); torch.cuda.empty_cache(); continue
    export_to_video(v, f'saida/{nome}.mp4', fps=24)
    print(f'{nome}: pronto em {time.time() - t0:.0f} s', flush=True); mostra(nome)

